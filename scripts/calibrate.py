"""Phase D — calibrate a trained deployment checkpoint on Val-Realistic.

Implements the calibration procedure that gates the one-shot Test-Realistic
(training.md §12, inference.md §7). Order matters (training.md §12, lines on
calibration-deployment parity §4.6): **TTA selection → temperature → threshold**,
all on Val-Realistic, all with the precision/scales the deployment will use.

What it does, for one or more deploy checkpoints (best_deployment.pth):
  1. Collect raw per-tile val logits (tta=none) once per checkpoint.
  2. TTA selection: compare none vs minimal(2×) vs full(D4 8×) on val
     PR-AUC-geomean + precision@0.5; adopt the cheapest config gaining
     >= `--tta-gain` PR-AUC AND dropping precision <= `--tta-prec-drop`
     (defaults from the plan: +1% / -0.5%). Default stays "none".
  3. Temperature: fit a single T minimizing NLL on val logits (sigmoid(logit/T)).
  4. Threshold: PR curve at the configured realistic ratios; pick the smallest
     threshold reaching `--target-precision` (spec default 0.8) at the reference
     ratio; report recall + precision/recall at every ratio.
  5. Single-best vs N-seed ensemble: if >1 checkpoint, also score the
     mean-probability ensemble; recommend it only on a measured PR-AUC gain.

Writes a calibration report JSON and (with --write-deployment) the learned
temperature + threshold + tta back into configs/deployment.yaml (training.md §4.6).
It does NOT touch Test-Realistic — that is a separate one-shot (evaluate_test.py).

Run (in the training Docker image, GPU):
    python scripts/calibrate.py \
        --config configs/aug_trivialaugment_deploy.yaml \
        --checkpoint seed42=/outputs/v1.0/runs/aug_trivialaugment_deploy/checkpoints/best_deployment.pth \
        --checkpoint seed43=/outputs/v1.0/runs/aug_trivialaugment_deploy_seed43/checkpoints/best_deployment.pth \
        --checkpoint seed44=/outputs/v1.0/runs/aug_trivialaugment_deploy_seed44/checkpoints/best_deployment.pth \
        --out /outputs/v1.0/calibration/effb5_trivialaug
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import RTSDataset, parse_extra_spec  # noqa: E402
from data.normalization import load_stats  # noqa: E402
from data.splits import get_tile_ids, load_metadata, load_splits_yaml  # noqa: E402
from data.transforms import build_eval_transforms  # noqa: E402
from inference.predictor import TTA_PASSES, predict_probs  # noqa: E402
from models import build_model  # noqa: E402
from training.metrics import _geomean  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Val loader (mirrors scripts/train.py:_setup_data val branch — same RTSDataset
# construction so boundary handling + normalization are identical; parity-critical)
# ---------------------------------------------------------------------------


def build_val_loader(cfg: dict) -> tuple[DataLoader, list[str]]:
    """Build the Val-Realistic loader exactly as training does (parity)."""
    data_root = cfg["data"]["data_root"]
    tile_size = int(cfg["data"]["tile_size"])
    ignore_idx = int(cfg["data"]["label_ignore_index"])
    boundary = cfg["loss"]["boundary_handling"]
    boundary_w = int(cfg["loss"].get("boundary_ignore_width", 3))

    metadata = load_metadata(Path(data_root) / cfg["data"]["metadata_csv"])
    splits = load_splits_yaml(Path(data_root) / cfg["data"]["splits_yaml"])
    extra_channels = parse_extra_spec(cfg["channels"].get("extra", []))
    stats_path = cfg["data"]["normalization_stats_path"]
    try:
        stats = load_stats(stats_path)
    except FileNotFoundError:
        stats = None
        logger.warning("Normalization stats not found at %s", stats_path)

    val_ids = get_tile_ids("val_realistic", metadata, splits)
    logger.info("Val-Realistic tiles: %d", len(val_ids))

    val_ds = RTSDataset(
        tile_ids=val_ids,
        metadata=metadata,
        data_root=data_root,
        rgb_dir=cfg["data"]["rgb_dir"],
        extra_dir=cfg["data"]["extra_dir"],
        labels_dir=cfg["data"]["labels_dir"],
        extra_channels=extra_channels,
        norm_stats_path=stats_path if stats is not None else None,
        transform=build_eval_transforms(),
        tile_size=tile_size,
        label_ignore_index=ignore_idx,
        boundary_handling=boundary,
        boundary_ignore_width=boundary_w,
        nodata_handling=cfg["data"].get("nodata_handling", False),
        aug_cfg=None,
        seed=int(cfg.get("seed", 42)),
    )
    bs = int(cfg["training"]["batch_size"])
    nw = int(cfg["training"]["num_workers"])
    loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, drop_last=False,
        num_workers=nw, pin_memory=True,
        prefetch_factor=(int(cfg["training"]["prefetch_factor"]) if nw > 0 else None),
        persistent_workers=False,
    )
    return loader, val_ids


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------


def load_checkpoint(model_cfg: dict, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Build the model and load deploy (EMA) weights from best_deployment.pth."""
    model = build_model(model_cfg)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd["model_state_dict"] if "model_state_dict" in sd else sd
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def collect_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    tta: str,
    temperature: float,
    precision: str,
    ignore_index: int,
) -> dict[str, dict]:
    """Run val once; return {tile_id: {probs, labels, is_pos}} valid-pixel-only.

    Uses inference.predictor.predict_probs so TTA + temperature fusion is
    bit-identical to the deployment path (training.md §4.6 parity).
    """
    out: dict[str, dict] = {}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].numpy()
        tile_ids = batch["tile_id"]
        probs = predict_probs(model, images, temperature=temperature,
                              tta=tta, precision=precision).cpu().numpy()
        for i, tid in enumerate(tile_ids):
            valid = labels[i] != ignore_index
            gt = (labels[i] == 1) & valid
            out[tid] = {
                "probs": probs[i][valid].astype(np.float32),
                "labels": (labels[i][valid] == 1).astype(np.uint8),
                "is_pos": bool(gt.any()),
            }
    return out


# ---------------------------------------------------------------------------
# Metrics on collected per-tile probabilities (reuse training prevalence logic)
# ---------------------------------------------------------------------------


def pr_auc_geomean(
    per_tile: dict[str, dict], ratios: list[int], seed: int = 42, max_pix: int = 10_000_000,
) -> tuple[float, dict[int, float]]:
    """PR-AUC geomean over prevalence ratios — mirrors training.metrics logic
    (subsample negatives without replacement, proportional pixel cap)."""
    rng = np.random.default_rng(seed)
    pos = [t for t in per_tile.values() if t["is_pos"]]
    neg = [t for t in per_tile.values() if not t["is_pos"]]
    per_ratio: dict[int, float] = {}
    if not pos or not neg:
        return 0.0, {r: 0.0 for r in ratios}
    for r in ratios:
        needed = r * len(pos)
        replace = len(neg) < needed
        idx = rng.choice(len(neg), size=needed, replace=replace)
        tiles = pos + [neg[i] for i in idx]
        total = sum(len(t["probs"]) for t in tiles)
        if total > max_pix:
            frac = max_pix / total
            P, L = [], []
            for t in tiles:
                n = max(1, int(round(len(t["probs"]) * frac)))
                si = rng.choice(len(t["probs"]), size=n, replace=False)
                P.append(t["probs"][si]); L.append(t["labels"][si])
            probs = np.concatenate(P); labs = np.concatenate(L)
        else:
            probs = np.concatenate([t["probs"] for t in tiles])
            labs = np.concatenate([t["labels"] for t in tiles])
        per_ratio[r] = float(average_precision_score(labs, probs)) if labs.max() > 0 else 0.0
    return _geomean([per_ratio[r] for r in ratios]), per_ratio


def precision_at_threshold(per_tile: dict[str, dict], thr: float) -> float:
    """Global pixel precision at a probability threshold (natural val prevalence)."""
    tp = fp = 0
    for t in per_tile.values():
        pred = t["probs"] >= thr
        tp += int(np.logical_and(pred, t["labels"] == 1).sum())
        fp += int(np.logical_and(pred, t["labels"] == 0).sum())
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


# ---------------------------------------------------------------------------
# Temperature + threshold
# ---------------------------------------------------------------------------


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, max_pix: int = 10_000_000, seed: int = 42,
) -> float:
    """T minimizing BCE NLL of sigmoid(logit/T). Bounded scalar search [0.25, 5].

    Fitting a single scalar needs only a sample; subsample to `max_pix` pixels
    (preserving the natural prevalence) so the search stays fast on full-val arrays.
    """
    if logits.size > max_pix:
        rng = np.random.default_rng(seed)
        si = rng.choice(logits.size, size=max_pix, replace=False)
        logits, labels = logits[si], labels[si]
    lg = logits.astype(np.float64); y = labels.astype(np.float64)

    def nll(T: float) -> float:
        z = lg / T
        # numerically stable BCE-with-logits
        return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))

    res = minimize_scalar(nll, bounds=(0.25, 5.0), method="bounded")
    return float(res.x)


def select_threshold(
    per_tile: dict[str, dict], ratio: int, target_precision: float, seed: int = 42,
) -> dict:
    """PR curve at a prevalence ratio; smallest threshold with precision >= target.

    Falls back to the max-F1 threshold if the target is unreachable.
    """
    rng = np.random.default_rng(seed)
    pos = [t for t in per_tile.values() if t["is_pos"]]
    neg = [t for t in per_tile.values() if not t["is_pos"]]
    needed = ratio * len(pos)
    idx = rng.choice(len(neg), size=needed, replace=len(neg) < needed)
    tiles = pos + [neg[i] for i in idx]
    probs = np.concatenate([t["probs"] for t in tiles])
    labs = np.concatenate([t["labels"] for t in tiles])
    max_pix = 10_000_000
    if probs.size > max_pix:  # a sample is plenty to locate the PR operating point
        si = rng.choice(probs.size, size=max_pix, replace=False)
        probs, labs = probs[si], labs[si]
    prec, rec, thr = precision_recall_curve(labs, probs)
    # prec/rec have len = len(thr)+1; align by dropping the last point.
    prec_t, rec_t = prec[:-1], rec[:-1]
    ok = np.where(prec_t >= target_precision)[0]
    if len(ok) > 0:
        j = ok[np.argmax(rec_t[ok])]  # highest recall meeting the precision target
        return {"threshold": float(thr[j]), "precision": float(prec_t[j]),
                "recall": float(rec_t[j]), "target_met": True}
    f1 = 2 * prec_t * rec_t / np.maximum(prec_t + rec_t, _EPS)
    j = int(np.argmax(f1))
    return {"threshold": float(thr[j]), "precision": float(prec_t[j]),
            "recall": float(rec_t[j]), "target_met": False}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="training/deploy config (model + data + ratios)")
    p.add_argument("--checkpoint", action="append", required=True,
                   metavar="NAME=PATH", help="deploy checkpoint; repeatable for an ensemble")
    p.add_argument("--out", required=True, help="output dir for calibration_report.json")
    p.add_argument("--deployment-yaml", default="configs/deployment.yaml")
    p.add_argument("--write-deployment", action="store_true",
                   help="write learned temperature/threshold/tta into the deployment yaml")
    p.add_argument("--target-precision", type=float, default=0.8)
    p.add_argument("--ref-ratio", type=int, default=None,
                   help="prevalence ratio for threshold selection (default: max configured)")
    p.add_argument("--tta-candidates", default="none,minimal,full")
    p.add_argument("--tta-gain", type=float, default=0.01, help="min PR-AUC gain to adopt TTA")
    p.add_argument("--tta-prec-drop", type=float, default=0.005, help="max precision drop for TTA")
    p.add_argument("--ensemble-gain", type=float, default=0.005,
                   help="min PR-AUC geomean gain to recommend the ensemble over best-single")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=str(out_dir / "calibrate.log"))

    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    precision = load_config(args.deployment_yaml).get("precision", "bf16")
    ignore_idx = int(cfg["data"]["label_ignore_index"])
    ratios = [int(r) for r in cfg["metrics"]["pr_auc_ratios"]]
    ref_ratio = args.ref_ratio or max(ratios)
    ckpts = dict(c.split("=", 1) for c in args.checkpoint)
    logger.info("Calibrating %d checkpoint(s): %s | precision=%s ratios=%s ref=%s",
                len(ckpts), list(ckpts), precision, ratios, ref_ratio)

    loader, _ = build_val_loader(cfg)

    # --- collect raw logits (tta=none, T=1) per checkpoint --------------------
    # T=1 => predict_probs returns sigmoid(logit); recover logit for temperature fit.
    per_ckpt: dict[str, dict[str, dict]] = {}
    for name, path in ckpts.items():
        logger.info("Forward pass (tta=none): %s", name)
        model = load_checkpoint(cfg, path, device)
        per_ckpt[name] = collect_probs(model, loader, device, tta="none",
                                       temperature=1.0, precision=precision,
                                       ignore_index=ignore_idx)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report: dict = {"config": args.config, "precision": precision, "ratios": ratios,
                    "ref_ratio": ref_ratio, "target_precision": args.target_precision,
                    "checkpoints": {}}

    # --- per-checkpoint: baseline PR-AUC, temperature, threshold --------------
    for name, per_tile in per_ckpt.items():
        gm, per_r = pr_auc_geomean(per_tile, ratios)
        logits = np.concatenate([_logit(t["probs"]) for t in per_tile.values()])
        labels = np.concatenate([t["labels"] for t in per_tile.values()])
        T = fit_temperature(logits, labels)
        # threshold on temperature-scaled probabilities
        scaled = {tid: {**t, "probs": _sigmoid(_logit(t["probs"]) / T)}
                  for tid, t in per_tile.items()}
        thr = select_threshold(scaled, ref_ratio, args.target_precision)
        report["checkpoints"][name] = {
            "pr_auc_geomean_T1": gm, "pr_auc_per_ratio_T1": per_r,
            "temperature": T, "threshold": thr,
        }
        logger.info("%s: PR-AUC-geomean=%.4f T=%.3f thr=%.4f (P=%.3f R=%.3f met=%s)",
                    name, gm, T, thr["threshold"], thr["precision"], thr["recall"],
                    thr["target_met"])

    # --- TTA selection (on the best single checkpoint) ------------------------
    best_name = max(per_ckpt, key=lambda n: report["checkpoints"][n]["pr_auc_geomean_T1"])
    base_gm = report["checkpoints"][best_name]["pr_auc_geomean_T1"]
    base_prec = precision_at_threshold(per_ckpt[best_name], 0.5)
    tta_table = {"none": {"pr_auc_geomean": base_gm, "precision_at_0.5": base_prec}}
    chosen_tta = "none"
    model = load_checkpoint(cfg, ckpts[best_name], device)
    for tta in [t for t in args.tta_candidates.split(",") if t not in ("none", "")]:
        if tta not in TTA_PASSES:
            logger.warning("Unknown TTA %r, skipping", tta); continue
        logger.info("TTA eval on %s: %s (%dx)", best_name, tta, len(TTA_PASSES[tta]))
        pt = collect_probs(model, loader, device, tta=tta, temperature=1.0,
                           precision=precision, ignore_index=ignore_idx)
        gm, _ = pr_auc_geomean(pt, ratios)
        prc = precision_at_threshold(pt, 0.5)
        tta_table[tta] = {"pr_auc_geomean": gm, "precision_at_0.5": prc}
        if (gm - base_gm) >= args.tta_gain and (base_prec - prc) <= args.tta_prec_drop:
            chosen_tta = tta  # cheapest-first order in candidates
            break
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    report["tta_selection"] = {"table": tta_table, "chosen": chosen_tta,
                               "gate": {"min_gain": args.tta_gain, "max_prec_drop": args.tta_prec_drop}}
    logger.info("TTA chosen: %s | table=%s", chosen_tta, tta_table)

    # --- single-best vs ensemble (mean prob over seeds) -----------------------
    recommendation = {"mode": "single", "checkpoint": best_name}
    if len(per_ckpt) > 1:
        common = set.intersection(*[set(pt) for pt in per_ckpt.values()])
        ens = {}
        for tid in common:
            P = np.mean([per_ckpt[n][tid]["probs"] for n in per_ckpt], axis=0)
            ens[tid] = {"probs": P.astype(np.float32),
                        "labels": next(iter(per_ckpt.values()))[tid]["labels"],
                        "is_pos": next(iter(per_ckpt.values()))[tid]["is_pos"]}
        ens_gm, ens_per_r = pr_auc_geomean(ens, ratios)
        report["ensemble"] = {"pr_auc_geomean_T1": ens_gm, "pr_auc_per_ratio_T1": ens_per_r,
                              "n_models": len(per_ckpt)}
        logger.info("Ensemble (%d-seed mean prob): PR-AUC-geomean=%.4f vs best-single %.4f (Δ=%+.4f)",
                    len(per_ckpt), ens_gm, base_gm, ens_gm - base_gm)
        if (ens_gm - base_gm) >= args.ensemble_gain:
            # calibrate the ensemble too
            ens_logits = np.concatenate([_logit(t["probs"]) for t in ens.values()])
            ens_labels = np.concatenate([t["labels"] for t in ens.values()])
            T = fit_temperature(ens_logits, ens_labels)
            scaled = {tid: {**t, "probs": _sigmoid(_logit(t["probs"]) / T)} for tid, t in ens.items()}
            thr = select_threshold(scaled, ref_ratio, args.target_precision)
            report["ensemble"].update({"temperature": T, "threshold": thr})
            recommendation = {"mode": "ensemble", "n_models": len(per_ckpt),
                              "note": f"+{ens_gm - base_gm:.4f} PR-AUC over best-single at {len(per_ckpt)}x cost"}
        else:
            recommendation = {"mode": "single", "checkpoint": best_name,
                              "note": f"ensemble gain {ens_gm - base_gm:+.4f} < {args.ensemble_gain} → not worth {len(per_ckpt)}x cost"}
    report["recommendation"] = recommendation

    # --- write report + (optional) deployment.yaml ----------------------------
    (out_dir / "calibration_report.json").write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s", out_dir / "calibration_report.json")

    if args.write_deployment:
        dep_path = Path(args.deployment_yaml)
        dep = yaml.safe_load(dep_path.read_text())
        if recommendation["mode"] == "ensemble":
            T = report["ensemble"]["temperature"]; thr = report["ensemble"]["threshold"]["threshold"]
        else:
            c = report["checkpoints"][recommendation["checkpoint"]]
            T = c["temperature"]; thr = c["threshold"]["threshold"]
        dep["temperature"] = round(float(T), 6)
        dep["threshold"] = round(float(thr), 6)
        dep["tta"] = chosen_tta
        dep_path.write_text(yaml.safe_dump(dep, sort_keys=False))
        logger.info("Wrote calibration into %s: T=%.4f thr=%.4f tta=%s",
                    dep_path, T, thr, chosen_tta)

    print(json.dumps({"recommendation": recommendation,
                      "tta": chosen_tta,
                      "checkpoints": {n: {"pr_auc_geomean": report["checkpoints"][n]["pr_auc_geomean_T1"],
                                          "temperature": report["checkpoints"][n]["temperature"],
                                          "threshold": report["checkpoints"][n]["threshold"]["threshold"]}
                                      for n in report["checkpoints"]}}, indent=2))
    return 0


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


if __name__ == "__main__":
    raise SystemExit(main())

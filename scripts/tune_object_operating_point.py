"""Tier-1 object operating-point tuning on Val-Realistic — report-only.

The deployed model (3-seed EffB5 ensemble, configs/deployment.yaml) has its
threshold tuned for *pixel* precision (calibrate.py); object-F1 was never
optimized. Good pixel ranking (PR-AUC-geomean 0.9393) implies a better OBJECT
operating point exists. This script sweeps `threshold x min_blob_size x
morphological-close-radius` on cached val predictions and reports the
obj-F1-optimal operating point plus an object-error decomposition — WITHOUT
retraining.

FACTS ONLY: counts, metric tables, and the computed obj-F1 argmax. No
recommendations about other levers (overlap/boundary losses + copy-paste are
already tested elsewhere). REPORT-ONLY: does NOT modify configs/deployment.yaml
or post-inference code. Val-Realistic ONLY — never touches Test-Realistic (held
for the one-shot).

Parity: the object machinery (min-size filter + greedy IoU>=0.3 matching) is the
exact `training.metrics` code, so at the defaults (thr 0.5, min_blob 10, no
morph) the obj counts match `ValidationAccumulator`.

Run (training Docker image; GPU for the one-time forward pass, sweep is CPU):
    python scripts/tune_object_operating_point.py \
        --config configs/aug_trivialaugment_deploy.yaml \
        --checkpoint seed42=/mnt/outputs/v1.0/runs/aug_trivialaugment_deploy/checkpoints/best_deployment.pth \
        --checkpoint seed43=/mnt/outputs/v1.0/runs/aug_trivialaugment_deploy_seed43/checkpoints/best_deployment.pth \
        --checkpoint seed44=/mnt/outputs/v1.0/runs/aug_trivialaugment_deploy_seed44/checkpoints/best_deployment.pth \
        --out /mnt/outputs/v1.0/object_operating_point/effb5_ensemble
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.predictor import predict_probs  # noqa: E402
from scripts.calibrate import build_val_loader, load_checkpoint  # noqa: E402
from training.metrics import _filter_small_blobs, _match_objects, _safe_div  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Prediction collection (2-D maps — unlike calibrate.collect_probs which
# flattens to valid pixels; object metrics need the spatial structure)
# ---------------------------------------------------------------------------


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@torch.no_grad()
def collect_ensemble_probs(
    ckpts: dict[str, str],
    cfg: dict,
    device: torch.device,
    *,
    tta: str,
    temperature: float,
    precision: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Mean-prob fuse the checkpoints over Val-Realistic, keeping 2-D maps.

    Each model is run at T=1 (per-seed sigmoid probs); the mean is then
    temperature-scaled by ``temperature`` so the probability scale matches the
    deployed ensemble (calibrate.py:384-404). One model is held in memory at a
    time; only the running prob-sum + labels are kept across seeds.

    Returns:
        (tile_ids, probs, labels): probs (N, H, W) float32 in [0, 1] on the
        deployed scale; labels (N, H, W) int16 in {0, 1, ignore_index}.
    """
    loader, _ = build_val_loader(cfg)
    prob_sum: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    order: list[str] = []

    for name, path in ckpts.items():
        logger.info("Forward pass (tta=%s): %s", tta, name)
        model = load_checkpoint(cfg, path, device)
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            lbl = batch["label"].numpy()
            probs = predict_probs(model, images, temperature=1.0,
                                  tta=tta, precision=precision).cpu().numpy()
            for i, tid in enumerate(batch["tile_id"]):
                if tid not in prob_sum:
                    prob_sum[tid] = probs[i].astype(np.float64)
                    labels[tid] = lbl[i].astype(np.int16)
                    order.append(tid)
                else:
                    prob_sum[tid] += probs[i]
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n = len(ckpts)
    probs = np.stack([_sigmoid(_logit(prob_sum[t] / n) / temperature).astype(np.float32)
                      for t in order])
    labs = np.stack([labels[t] for t in order])
    return order, probs, labs


# ---------------------------------------------------------------------------
# Object operating point — reuses the exact training.metrics machinery
# ---------------------------------------------------------------------------


def _pred_mask(prob: np.ndarray, valid: np.ndarray, thr: float, morph_r: int) -> np.ndarray:
    """Binary prediction at (thr, morph close radius), restricted to valid."""
    pred = (prob >= thr) & valid
    if morph_r > 0:
        struct = ndimage.iterate_structure(
            ndimage.generate_binary_structure(2, 1), morph_r)
        pred = ndimage.binary_closing(pred, structure=struct) & valid
    return pred


def object_counts(
    prob: np.ndarray, label: np.ndarray, ignore_index: int,
    thr: float, min_blob: int, morph_r: int, iou_thr: float,
) -> tuple[int, int, int, int, int, int]:
    """One tile → (obj_tp, obj_fp, obj_fn, pix_tp, pix_fp, pix_fn).

    At (thr=0.5, min_blob=10, morph_r=0) this reproduces
    `ValidationAccumulator._accumulate_tile` exactly (parity test).
    """
    valid = label != ignore_index
    gt = (label == 1) & valid
    pred = _pred_mask(prob, valid, thr, morph_r)

    pix_tp = int(np.logical_and(pred, gt).sum())
    pix_fp = int(np.logical_and(pred, np.logical_not(gt) & valid).sum())
    pix_fn = int(np.logical_and(np.logical_not(pred) & valid, gt).sum())

    pred_filt = _filter_small_blobs(pred.astype(np.uint8), min_blob)
    pred_labels, n_pred = ndimage.label(pred_filt)
    gt_labels, n_gt = ndimage.label(gt.astype(np.uint8))
    conf = (np.array(ndimage.mean(prob, pred_labels, index=np.arange(1, n_pred + 1)),
                     dtype=np.float64) if n_pred > 0 else np.zeros(0))
    tp, fp, fn = _match_objects(pred_labels, n_pred, gt_labels, n_gt, conf, iou_thr)
    return tp, fp, fn, pix_tp, pix_fp, pix_fn


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    return p, r, _safe_div(2 * p * r, p + r)


def evaluate_grid(
    probs: np.ndarray, labels: np.ndarray, ignore_index: int, iou_thr: float,
    thresholds: list[float], min_blobs: list[int], morphs: list[int],
) -> list[dict]:
    """Aggregate obj/pixel metrics over all tiles for every (thr, min_blob, morph).

    Equivalent to summing ``object_counts`` over tiles per cell, but the
    expensive work — morphological close + connected-component labelling — is
    done once per (threshold, morph) and reused across ``min_blob`` (which only
    filters predictions by blob size). GT components are labelled once (they do
    not depend on any knob). Pixel metrics depend only on (thr, morph).
    """
    # GT labelling — independent of thr/morph/min_blob.
    gt_cache = []
    for label in labels:
        valid = label != ignore_index
        gt = (label == 1) & valid
        gl, ng = ndimage.label(gt.astype(np.uint8))
        gt_cache.append((valid, gt, gl, ng))

    rows: list[dict] = []
    for morph_r in morphs:
        for thr in thresholds:
            ptp = pfp = pfn = 0
            otp = {m: 0 for m in min_blobs}
            ofp = {m: 0 for m in min_blobs}
            ofn = {m: 0 for m in min_blobs}
            for prob, (valid, gt, gl, ng) in zip(probs, gt_cache):
                pred = _pred_mask(prob, valid, thr, morph_r)
                ptp += int(np.logical_and(pred, gt).sum())
                pfp += int(np.logical_and(pred, np.logical_not(gt) & valid).sum())
                pfn += int(np.logical_and(np.logical_not(pred) & valid, gt).sum())
                pl, npred = ndimage.label(pred.astype(np.uint8))
                sizes = np.bincount(pl.ravel())[1:] if npred else np.zeros(0, dtype=int)
                conf = (np.array(ndimage.mean(prob, pl, index=np.arange(1, npred + 1)))
                        if npred else np.zeros(0))
                for m in min_blobs:
                    if npred == 0:
                        ofn[m] += ng
                        continue
                    keep = sizes >= m if m > 1 else np.ones(npred, dtype=bool)
                    k = int(keep.sum())
                    if k == 0:
                        ofn[m] += ng
                        continue
                    remap = np.zeros(npred + 1, dtype=np.int64)
                    remap[np.nonzero(keep)[0] + 1] = np.arange(1, k + 1)
                    tp, fp, fn = _match_objects(remap[pl], k, gl, ng, conf[keep], iou_thr)
                    otp[m] += tp; ofp[m] += fp; ofn[m] += fn
            pix_p, pix_r = _safe_div(ptp, ptp + pfp), _safe_div(ptp, ptp + pfn)
            for m in min_blobs:
                op, orc, of1 = _prf(otp[m], ofp[m], ofn[m])
                rows.append({
                    "threshold": round(float(thr), 4), "min_blob_size": int(m),
                    "morph_close_radius": int(morph_r),
                    "obj_tp": otp[m], "obj_fp": ofp[m], "obj_fn": ofn[m],
                    "obj_precision": round(op, 4), "obj_recall": round(orc, 4),
                    "obj_f1": round(of1, 4),
                    "pixel_precision": round(pix_p, 4), "pixel_recall": round(pix_r, 4),
                })
    return rows


# ---------------------------------------------------------------------------
# Object-error decomposition (facts only; run at the reported points)
# ---------------------------------------------------------------------------


def decompose(
    probs: np.ndarray, labels: np.ndarray, ignore_index: int,
    thr: float, min_blob: int, morph_r: int, iou_thr: float,
) -> dict:
    """Categorise unmatched predictions (FP) and missed GT (FN) over all tiles.

    FP: no_overlap (best IoU 0) · low_iou (0<IoU<match) · fragment (overlaps a GT
        already matched by a higher-confidence pred).
    FN: missed (no pred pixel overlaps the GT) · low_iou (overlaps but IoU<match).
    Also reports FP-blob size percentiles. Counts only — no interpretation.
    """
    fp_no, fp_low, fp_frag = 0, 0, 0
    fn_missed, fn_low = 0, 0
    fp_sizes: list[int] = []

    for prob, label in zip(probs, labels):
        valid = label != ignore_index
        gt = (label == 1) & valid
        pred = _pred_mask(prob, valid, thr, morph_r)
        pred_filt = _filter_small_blobs(pred.astype(np.uint8), min_blob)
        pred_labels, n_pred = ndimage.label(pred_filt)
        gt_labels, n_gt = ndimage.label(gt.astype(np.uint8))
        if n_pred == 0 and n_gt == 0:
            continue

        # IoU of every pred blob vs every gt blob.
        iou = np.zeros((n_pred, n_gt))
        for p in range(1, n_pred + 1):
            pm = pred_labels == p
            for g in range(1, n_gt + 1):
                gm = gt_labels == g
                inter = int(np.logical_and(pm, gm).sum())
                if inter:
                    iou[p - 1, g - 1] = inter / int(np.logical_or(pm, gm).sum())

        # Greedy 1-to-1 by descending confidence (mirrors _match_objects).
        conf = (np.array(ndimage.mean(prob, pred_labels, index=np.arange(1, n_pred + 1)))
                if n_pred else np.zeros(0))
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        for p in np.argsort(conf)[::-1] if n_pred else []:
            row = iou[p].copy()
            for g in matched_gt:
                row[g] = 0.0
            g = int(np.argmax(row)) if n_gt else -1
            if g >= 0 and row[g] >= iou_thr:
                matched_gt.add(g); matched_pred.add(int(p))

        for p in range(n_pred):
            if p in matched_pred:
                continue
            best = int(np.argmax(iou[p])) if n_gt else -1
            best_iou = iou[p, best] if best >= 0 else 0.0
            fp_sizes.append(int((pred_labels == p + 1).sum()))
            if best_iou <= 0.0:
                fp_no += 1
            elif best in matched_gt:
                fp_frag += 1
            else:
                fp_low += 1

        for g in range(n_gt):
            if g in matched_gt:
                continue
            if n_pred == 0 or iou[:, g].max() <= 0.0:
                fn_missed += 1
            else:
                fn_low += 1

    sizes = np.array(fp_sizes) if fp_sizes else np.zeros(1)
    return {
        "fp_no_overlap": fp_no, "fp_low_iou": fp_low, "fp_fragment": fp_frag,
        "fn_missed": fn_missed, "fn_low_iou": fn_low,
        "fp_blob_size_px_p50": int(np.percentile(sizes, 50)),
        "fp_blob_size_px_p90": int(np.percentile(sizes, 90)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="training/deploy config (model + data)")
    p.add_argument("--checkpoint", action="append", required=True,
                   metavar="NAME=PATH", help="deploy checkpoint; repeatable → mean-prob ensemble")
    p.add_argument("--out", required=True, help="output dir for the report + prob cache")
    p.add_argument("--deployment-yaml", default="configs/deployment.yaml")
    p.add_argument("--precision-floor", type=float, default=0.8,
                   help="pixel-precision floor the obj-F1 argmax must satisfy (natural val prevalence)")
    p.add_argument("--thresholds", default=None,
                   help="comma list; default arange(0.05,0.95,0.05) + the deployed threshold")
    p.add_argument("--min-blobs", default="1,5,10,20,40,80")
    p.add_argument("--morph-radii", default="0,1,2")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=str(out_dir / "tune_object_operating_point.log"))

    cfg = load_config(args.config)
    dep = load_config(args.deployment_yaml)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ignore_index = int(cfg["data"]["label_ignore_index"])
    iou_thr = float(cfg.get("metrics", {}).get("object_iou_threshold", 0.3))
    precision = dep.get("precision", "bf16")
    tta = dep.get("tta", "none")
    temperature = float(dep.get("temperature", 1.0))
    deployed_thr = float(dep["threshold"])
    ckpts = dict(c.split("=", 1) for c in args.checkpoint)

    # --- predictions (cached) -------------------------------------------------
    cache = out_dir / "val_probs.npz"
    if cache.exists():
        logger.info("Loading cached val predictions: %s", cache)
        z = np.load(cache, allow_pickle=True)
        tids, probs, labels = list(z["tids"]), z["probs"], z["labels"]
    else:
        logger.info("Collecting %d-ckpt ensemble val predictions (T=%.4f, tta=%s)",
                    len(ckpts), temperature, tta)
        tids, probs, labels = collect_ensemble_probs(
            ckpts, cfg, device, tta=tta, temperature=temperature, precision=precision)
        np.savez_compressed(cache, tids=np.array(tids), probs=probs, labels=labels)
        logger.info("Cached %d val tiles → %s", len(tids), cache)

    # --- grids ----------------------------------------------------------------
    if args.thresholds:
        thresholds = sorted({round(float(x), 4) for x in args.thresholds.split(",")})
    else:
        thresholds = sorted({round(float(x), 4) for x in np.arange(0.05, 0.95, 0.05)}
                            | {round(deployed_thr, 4)})
    min_blobs = [int(x) for x in args.min_blobs.split(",")]
    morphs = [int(x) for x in args.morph_radii.split(",")]
    logger.info("Sweep: %d thr × %d min_blob × %d morph = %d cells over %d tiles",
                len(thresholds), len(min_blobs), len(morphs),
                len(thresholds) * len(min_blobs) * len(morphs), len(tids))

    grid = evaluate_grid(probs, labels, ignore_index, iou_thr, thresholds, min_blobs, morphs)

    # current deployed operating point (deployed thr, training defaults min10/no-morph)
    deployed = next(r for r in grid
                    if r["threshold"] == round(deployed_thr, 4)
                    and r["min_blob_size"] == 10 and r["morph_close_radius"] == 0)
    # global obj-F1 argmax, and argmax subject to the pixel-precision floor
    best = max(grid, key=lambda r: r["obj_f1"])
    feasible = [r for r in grid if r["pixel_precision"] >= args.precision_floor]
    best_floored = max(feasible, key=lambda r: r["obj_f1"]) if feasible else None

    report = {
        "config": args.config, "checkpoints": ckpts, "n_val_tiles": len(tids),
        "object_iou_threshold": iou_thr, "temperature": temperature,
        "precision_floor": args.precision_floor,
        "deployed_operating_point": {**deployed, "note": "deployed threshold, min_blob=10, no morph"},
        "obj_f1_argmax_unconstrained": best,
        "obj_f1_argmax_at_precision_floor": best_floored,
        "decomposition": {
            "deployed": decompose(probs, labels, ignore_index, deployed_thr, 10, 0, iou_thr),
            "argmax_at_floor": (decompose(
                probs, labels, ignore_index, best_floored["threshold"],
                best_floored["min_blob_size"], best_floored["morph_close_radius"], iou_thr)
                if best_floored else None),
        },
        "grid": grid,
    }
    (out_dir / "object_operating_point_report.json").write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s", out_dir / "object_operating_point_report.json")

    print(json.dumps({
        "deployed_obj_f1": deployed["obj_f1"],
        "best_obj_f1_unconstrained": {k: best[k] for k in
            ("threshold", "min_blob_size", "morph_close_radius", "obj_f1", "pixel_precision")},
        "best_obj_f1_at_precision_floor": ({k: best_floored[k] for k in
            ("threshold", "min_blob_size", "morph_close_radius", "obj_f1", "pixel_precision")}
            if best_floored else None),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

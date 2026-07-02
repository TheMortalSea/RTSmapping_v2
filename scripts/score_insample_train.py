"""In-sample train inference pass for the bias/variance diagnosis (plan: Phase 0B).

Runs the frozen N-seed ensemble (deployed recipe) over a **region-stratified sample
of TRAIN positive tiles** and caches the fused probabilities to ``*_probs.npz`` — the
same format ``scripts/object_scorecard.py`` consumes. Scoring these in-sample tiles
and comparing their perception-invisible floor (F_in) against held-out val (F_held)
separates a representation/label bias gap from a generalization gap.

DELIBERATELY SEPARATE from ``scripts/evaluate_test.py``: that file is the sacred
one-shot Test-Realistic evaluator (training.md §10.3). This one never touches
test_realistic and reuses the identical fusion building blocks
(``inference.predictor.predict_probs_ensemble``).

IN-SAMPLE / DIAGNOSTIC ONLY: these tiles were in training, so the scorecard built
from this cache measures *fit*, not generalization. The relative F_in-vs-F_held gap
is the signal — never report these as model performance.

Run (on the L4 VM with GPU + the v2 seed checkpoints):
    python scripts/score_insample_train.py \
        --checkpoint seed42=gs://.../seed42 --checkpoint seed1=... --checkpoint seed2=... \
        --config configs/aug_trivialaugment_deploy.yaml \
        --deployment-yaml configs/deployment.yaml \
        --per-region-cap 40 \
        --cache-probs /mnt/outputs/v1.0/diagnostics/insample_train_probs.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.splits import get_tile_ids, load_metadata, load_splits_yaml  # noqa: E402
from utils.config import load_config, resolve_path  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def sample_region_stratified(
    metadata: pd.DataFrame,
    candidate_tids: list[str],
    per_region_cap: int,
    seed: int = 42,
) -> list[str]:
    """Region-stratified sample: up to ``per_region_cap`` tiles per RegionName.

    Pure + deterministic (seeded) so it is unit-testable without GPU/data. Covers
    every region present among ``candidate_tids`` (no region dropped), capping the
    dense ones so no single region dominates the in-sample floor estimate.
    """
    cand = set(candidate_tids)
    sub = metadata[metadata["Tile_ID"].isin(cand)]
    rng = np.random.default_rng(seed)
    picked: list[str] = []
    for region in sorted(sub["RegionName"].unique()):
        tids = sorted(sub.loc[sub["RegionName"] == region, "Tile_ID"].tolist())
        if len(tids) > per_region_cap:
            idx = rng.choice(len(tids), size=per_region_cap, replace=False)
            tids = [tids[i] for i in sorted(idx)]
        picked.extend(tids)
    return sorted(picked)


def _build_cache(
    checkpoints: dict[str, str],
    config: Path,
    deployment_yaml: Path,
    cache_probs: Path,
    per_region_cap: int,
    seed: int,
    device: str | None,
) -> None:
    """Run ensemble inference over the sampled train tiles; write the probs cache.

    Mirrors ``evaluate_test_ensemble``'s fusion (per-seed sigmoid at T=1 → mean →
    temperature on the fused prob), but on the in-sample train sample. GPU/data path
    — exercised on the L4 VM, not in the unit suite.
    """
    import torch  # local import: keep the sampler unit-testable without torch
    import yaml

    from data.dataset import RTSDataset, parse_extra_spec
    from data.transforms import build_eval_transforms
    from inference.predictor import predict_probs_ensemble
    from scripts.calibrate import load_checkpoint

    cfg = load_config(config)
    dep_cfg = yaml.safe_load(Path(deployment_yaml).read_text())
    if dep_cfg.get("threshold") is None or dep_cfg.get("temperature") is None:
        raise ValueError("deployment.yaml has null threshold/temperature — calibrate first.")
    temperature = float(dep_cfg["temperature"])
    precision = dep_cfg.get("precision", "bf16")
    tta = dep_cfg.get("tta", "none")
    _EPS = 1e-6

    metadata = load_metadata(resolve_path(cfg["data"]["data_root"], cfg["data"]["metadata_csv"]))
    splits = load_splits_yaml(resolve_path(cfg["data"]["data_root"], cfg["data"]["splits_yaml"]))
    train_pos = get_tile_ids("train", metadata, splits, class_filter="positive")
    sample_ids = sample_region_stratified(metadata, train_pos, per_region_cap, seed)
    n_regions = metadata.loc[metadata["Tile_ID"].isin(set(sample_ids)), "RegionName"].nunique()
    logger.info("In-sample train pass: %d/%d train-positive tiles across %d regions (cap=%d)",
                len(sample_ids), len(train_pos), n_regions, per_region_cap)

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = RTSDataset(
        tile_ids=sample_ids, metadata=metadata, data_root=cfg["data"]["data_root"],
        rgb_dir=cfg["data"]["rgb_dir"], extra_dir=cfg["data"]["extra_dir"],
        labels_dir=cfg["data"]["labels_dir"],
        extra_channels=parse_extra_spec(cfg["channels"].get("extra", [])),
        norm_stats_path=cfg["data"]["normalization_stats_path"],
        transform=build_eval_transforms(), tile_size=int(cfg["data"]["tile_size"]),
        label_ignore_index=int(cfg["data"]["label_ignore_index"]),
        boundary_handling="none",
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]), pin_memory=True,
    )
    models = [load_checkpoint(cfg, path, device_t) for path in checkpoints.values()]

    rprobs, rlabels, rtids = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device_t, non_blocking=True)
            probs = predict_probs_ensemble(models, images, temperature=temperature,
                                           tta=tta, precision=precision).float()
            probs = probs.clamp(_EPS, 1 - _EPS)
            rprobs.append(probs.cpu().numpy().astype(np.float32))
            rlabels.append(batch["label"].numpy().astype(np.int16))
            rtids.extend(batch["tile_id"])

    cache_probs.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_probs, tids=np.array(rtids),
                        probs=np.concatenate(rprobs, axis=0),
                        labels=np.concatenate(rlabels, axis=0))
    logger.info("Cached in-sample train predictions → %s (%d tiles)", cache_probs, len(rtids))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH",
                   help="ensemble member checkpoint; repeatable")
    p.add_argument("--config", type=Path, required=True, help="deploy/training config (model+data)")
    p.add_argument("--deployment-yaml", type=Path, default=Path("configs/deployment.yaml"))
    p.add_argument("--cache-probs", type=Path, required=True, help="output *_probs.npz path")
    p.add_argument("--per-region-cap", type=int, default=40,
                   help="max train-positive tiles sampled per region")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda|cpu; default auto")
    args = p.parse_args()

    setup_logging(level="INFO", log_file=str(args.cache_probs.parent / "score_insample_train.log"))
    checkpoints = dict(kv.split("=", 1) for kv in args.checkpoint)
    _build_cache(checkpoints, args.config, args.deployment_yaml, args.cache_probs,
                 args.per_region_cap, args.seed, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

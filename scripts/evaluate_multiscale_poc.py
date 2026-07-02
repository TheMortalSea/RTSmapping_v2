"""Multiscale-POC evaluation — gates 2 and 3 of ledger family M.

For each run checkpoint (best_deployment.pth, EMA weights):
  * evaluate val_realistic at **1x** (primary root) and at **0.5x** (the
    v1.0_scale05 root) with the training ValidationAccumulator — gate 2 is
    0.5x val F1 / geomean vs the same model's 1x numbers;
  * **fuse**: for every 1x val tile covered by a 0.5x val tile (positive
    blocks are quad-aligned, so a 1x grid tile maps to one block quadrant),
    bilinear-upsample that 256px quadrant of the 0.5x probability map to 512
    and average with the 1x map (inference.md §7.3 arithmetic mean over valid
    scales; uncovered tiles keep the 1x map — the §6.3 graceful degradation).
    Gate 3 compares fused vs 1x-only object recall / F1.

Facts only in the output JSON — the verdict lives in the ledger.

Run (inside rts-train:v2, one GPU):
  python scripts/evaluate_multiscale_poc.py \
      --run-dir /outputs/multiscale_poc_seed42 \
      --config configs/multiscale_poc_seed42.yaml \
      --out /outputs/multiscale_poc_eval/seed42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import RTSDataset, parse_extra_spec  # noqa: E402
from data.splits import get_tile_ids, load_metadata, load_splits_yaml  # noqa: E402
from data.transforms import build_eval_transforms  # noqa: E402
from models import build_model  # noqa: E402
from training import metrics as metrics_mod  # noqa: E402
from utils.config import load_config, resolve_path  # noqa: E402

logger = logging.getLogger("evaluate_multiscale_poc")


def _build_loader(cfg: dict, data_root: str, tile_ids, metadata) -> torch.utils.data.DataLoader:
    ds = RTSDataset(
        tile_ids=tile_ids,
        metadata=metadata,
        data_root=data_root,
        rgb_dir=cfg["data"]["rgb_dir"],
        extra_dir=cfg["data"]["extra_dir"],
        labels_dir=cfg["data"]["labels_dir"],
        extra_channels=parse_extra_spec(cfg["channels"].get("extra", [])),
        norm_stats_path=cfg["data"]["normalization_stats_path"],
        transform=build_eval_transforms(),
        tile_size=int(cfg["data"]["tile_size"]),
        label_ignore_index=int(cfg["data"]["label_ignore_index"]),
        boundary_handling="none",
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False, num_workers=4, pin_memory=True,
    )


@torch.no_grad()
def _run_split(model, loader, device, autocast_ctx, cfg) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Returns (metrics, probs_by_tile, labels_by_tile)."""
    acc = metrics_mod.ValidationAccumulator(cfg)
    probs: dict[str, np.ndarray] = {}
    labels_map: dict[str, np.ndarray] = {}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with autocast_ctx:
            logits = model(images)
        logits = logits.float()
        acc.update(logits, labels, batch["tile_id"])
        p = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)
        lab = labels.cpu().numpy().astype(np.uint8)
        for i, tid in enumerate(batch["tile_id"]):
            probs[tid] = p[i]
            labels_map[tid] = lab[i]
    return acc.compute(), probs, labels_map


def _bounds(root: str, sub: str, tid: str) -> tuple[float, float, float, float]:
    with rasterio.open(f"{root}/{sub}/{tid}.tif") as src:
        b = src.bounds
    return (b.left, b.bottom, b.right, b.top)


def _fuse(
    probs_1x: dict[str, np.ndarray],
    bounds_1x: dict[str, tuple],
    probs_05: dict[str, np.ndarray],
    bounds_05: dict[str, tuple],
) -> tuple[dict[str, np.ndarray], int]:
    """§7.3 average fusion on the 1x val grid. Returns (fused probs, n_covered)."""
    # Index 0.5x tiles by their bounds for containment lookup.
    fused: dict[str, np.ndarray] = {}
    n_cov = 0
    tiles05 = list(bounds_05.items())
    for tid, p1 in probs_1x.items():
        minx, miny, maxx, maxy = bounds_1x[tid]
        match = None
        for tid5, (m5x, m5y, M5x, M5y) in tiles05:
            if m5x - 1e-3 <= minx and maxx <= M5x + 1e-3 \
                    and m5y - 1e-3 <= miny and maxy <= M5y + 1e-3:
                match = (tid5, (m5x, m5y, M5x, M5y))
                break
        if match is None:
            fused[tid] = p1
            continue
        tid5, (m5x, m5y, M5x, M5y) = match
        p5 = probs_05[tid5]
        h, w = p5.shape
        # pixel window of the 1x tile inside the 0.5x tile (row 0 = top/maxy)
        resx = (M5x - m5x) / w
        resy = (M5y - m5y) / h
        c0 = int(round((minx - m5x) / resx))
        r0 = int(round((M5y - maxy) / resy))
        c1 = int(round((maxx - m5x) / resx))
        r1 = int(round((M5y - miny) / resy))
        quad = p5[r0:r1, c0:c1]
        if quad.size == 0:
            fused[tid] = p1
            continue
        up = F.interpolate(
            torch.from_numpy(quad)[None, None], size=p1.shape,
            mode="bilinear", align_corners=False,
        )[0, 0].numpy()
        fused[tid] = (p1 + up) / 2.0
        n_cov += 1
    return fused, n_cov


def _metrics_from_probs(probs: dict[str, np.ndarray], labels: dict[str, np.ndarray],
                        cfg: dict) -> dict:
    """Object/pixel metrics via the training accumulator, feeding logit(prob)."""
    acc = metrics_mod.ValidationAccumulator(cfg)
    eps = 1e-6
    for tid, p in probs.items():
        logit = np.log((p + eps) / (1 - p + eps)).astype(np.float32)
        acc.update(torch.from_numpy(logit)[None, None],
                   torch.from_numpy(labels[tid].astype(np.int64))[None], [tid])
    return acc.compute()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="training run dir containing checkpoints/best_deployment.pth")
    ap.add_argument("--config", required=True, type=Path, help="the run's training config")
    ap.add_argument("--scale05-root", default="/outputs/v1.0_scale05/data_local")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(cfg).to(device).eval()
    ckpt = torch.load(args.run_dir / "checkpoints" / "best_deployment.pth",
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("loaded %s (epoch %s)", args.run_dir.name, ckpt.get("epoch"))

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and torch.cuda.is_bf16_supported() else nullcontext()
    )

    result: dict = {"run": args.run_dir.name, "best_epoch": ckpt.get("epoch")}

    # --- 1x val ---
    root1 = cfg["data"]["data_root"]
    md1 = load_metadata(resolve_path(root1, cfg["data"]["metadata_csv"]))
    splits = load_splits_yaml(resolve_path(root1, cfg["data"]["splits_yaml"]))
    ids1 = get_tile_ids("val_realistic", md1, splits)
    m1, probs1, labels1 = _run_split(model, _build_loader(cfg, root1, ids1, md1),
                                     device, autocast_ctx, cfg)
    result["val_1x"] = m1
    logger.info("1x val: geomean %.4f f1 %.4f obj_recall %.4f",
                m1["val_realistic_pr_auc_geomean"], m1["object_f1"], m1["object_recall"])

    # --- 0.5x val ---
    root5 = args.scale05_root
    md5 = load_metadata(f"{root5}/metadata.csv")
    splits5 = load_splits_yaml(f"{root5}/splits.yaml")
    ids5 = get_tile_ids("val_realistic", md5, splits5)
    m5, probs5, labels5 = _run_split(model, _build_loader(cfg, root5, ids5, md5),
                                     device, autocast_ctx, cfg)
    result["val_05x"] = m5
    logger.info("0.5x val: geomean %.4f f1 %.4f obj_recall %.4f",
                m5["val_realistic_pr_auc_geomean"], m5["object_f1"], m5["object_recall"])

    # --- fusion (gate 3) ---
    b1 = {t: _bounds(root1, cfg["data"]["labels_dir"], t)
          if md1.set_index("Tile_ID").loc[t, "TrainClass"] == "positive"
          else _bounds(root1, cfg["data"]["rgb_dir"], t) for t in ids1}
    b5 = {t: _bounds(root5, "labels" if md5.set_index("Tile_ID").loc[t, "TrainClass"] == "positive"
                     else "PLANET-RGB", t) for t in ids5}
    fused, n_cov = _fuse(probs1, b1, probs5, b5)
    result["fusion_coverage"] = {"covered_1x_tiles": n_cov, "total_1x_tiles": len(ids1)}
    result["val_fused"] = _metrics_from_probs(fused, labels1, cfg)
    result["val_1x_reference"] = _metrics_from_probs(probs1, labels1, cfg)
    logger.info("fusion: %d/%d 1x tiles covered | fused obj_recall %.4f vs 1x %.4f",
                n_cov, len(ids1), result["val_fused"]["object_recall"],
                result["val_1x_reference"]["object_recall"])

    np.savez_compressed(args.out / "probs_1x.npz", **probs1)
    np.savez_compressed(args.out / "probs_05x.npz", **probs5)
    with open(args.out / "poc_eval.json", "w") as f:
        json.dump(result, f, indent=1, default=float)
    logger.info("wrote %s", args.out / "poc_eval.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

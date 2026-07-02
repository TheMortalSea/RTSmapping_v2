"""QC contact sheet of the perception-invisible objects (D1 label audit aid). Report-only.

Given a ``*_probs.npz`` cache + metadata + the RGB tile dir, this finds every GT object
the model is invisible on (max ensemble prob in its footprint < ``invisible_thr``), crops
the RGB around each with the GT outline drawn, and lays them out on paginated PNG contact
sheets + a CSV manifest. The point is to make the D1 audit fast: for each invisible object,
a human decides "real RTS the model missed" vs "label noise / not actually a visible slump".

FACTS ONLY: no verdict — it just surfaces the objects and their stats for review.

Run:
    python scripts/make_invisible_contact_sheet.py \
        --cache insample_train_probs.npz \
        --metadata /mnt/outputs/v1.0/data_local/metadata.csv \
        --rgb-dir /mnt/outputs/v1.0/data_local/PLANET-RGB \
        --out /mnt/outputs/.../diagnostics --tag insample_train
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from scipy import ndimage  # noqa: E402

# matplotlib is imported lazily in render() so the selection logic
# (find_invisible_objects) stays importable/testable without the plotting dep.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.splits import load_metadata  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def find_invisible_objects(
    probs: np.ndarray, labels: np.ndarray, tids: list[str],
    *, ignore_index: int, invisible_thr: float,
) -> list[dict]:
    """Per invisible GT object: tile_id, its footprint bbox, area, max/mean prob."""
    out: list[dict] = []
    for prob, label, tid in zip(probs, labels, tids):
        valid = label != ignore_index
        gt = (label == 1) & valid
        gt_labels, n = ndimage.label(gt.astype(np.uint8))
        for g in range(1, n + 1):
            fp = gt_labels == g
            max_p = float(prob[fp].max())
            if max_p >= invisible_thr:
                continue
            ys, xs = np.where(fp)
            out.append({
                "tile_id": tid, "comp": g,
                "area_px": int(fp.sum()),
                "max_prob": round(max_p, 4),
                "mean_prob": round(float(prob[fp].mean()), 4),
                "bbox": (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())),
                "_gt_labels": gt_labels,
            })
    return out


def _stretch(rgb: np.ndarray) -> np.ndarray:
    """Percentile 2–98 contrast stretch to [0,1] for display (per crop)."""
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(rgb.shape[2]):
        ch = rgb[:, :, c].astype(np.float32)
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        out[:, :, c] = np.clip((ch - lo) / (hi - lo + 1e-6), 0, 1) if hi > lo else 0.0
    return out


def _crop_with_outline(rgb_dir: Path, obj: dict, pad: int) -> np.ndarray:
    """RGB crop around the object bbox (+pad) with the GT outline drawn red."""
    tid = obj["tile_id"]
    with rasterio.open(rgb_dir / f"{tid}.tif") as src:
        rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
    H, W = rgb.shape[:2]
    y0, y1, x0, x1 = obj["bbox"]
    y0, y1 = max(0, y0 - pad), min(H, y1 + pad + 1)
    x0, x1 = max(0, x0 - pad), min(W, x1 + pad + 1)
    crop = _stretch(rgb[y0:y1, x0:x1])
    comp = (obj["_gt_labels"][y0:y1, x0:x1] == obj["comp"])
    outline = comp ^ ndimage.binary_erosion(comp)
    crop[outline] = [1.0, 0.0, 0.0]  # red GT boundary
    return crop


def render(objects: list[dict], rgb_dir: Path, out_dir: Path, tag: str,
           *, tid_region: dict[str, str], pad: int, per_page: int, cols: int) -> list[Path]:
    """Paginated contact-sheet PNGs; returns the written page paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pages: list[Path] = []
    objects = sorted(objects, key=lambda o: (tid_region.get(o["tile_id"], ""), -o["area_px"]))
    for pg, start in enumerate(range(0, len(objects), per_page)):
        chunk = objects[start:start + per_page]
        rows = (len(chunk) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for i, obj in enumerate(chunk):
            ax = axes[i]
            try:
                ax.imshow(_crop_with_outline(rgb_dir, obj, pad))
            except Exception as e:  # missing/corrupt tile — surface, don't crash the sheet
                ax.text(0.5, 0.5, f"[read err]\n{obj['tile_id']}", ha="center", va="center", fontsize=6)
                logger.warning("crop failed for %s: %s", obj["tile_id"], e)
            reg = tid_region.get(obj["tile_id"], "?")[:16]
            ax.set_title(f"{obj['tile_id'][:8]} {reg}\nA={obj['area_px']} mx={obj['max_prob']}",
                         fontsize=5.5)
        fig.suptitle(f"invisible objects — {tag} — page {pg + 1} "
                     f"({start + 1}–{start + len(chunk)} of {len(objects)})", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = out_dir / f"invisible_contact_{tag}_p{pg + 1:02d}.png"
        fig.savefig(path, dpi=130); plt.close(fig)
        pages.append(path)
        logger.info("wrote %s", path)
    return pages


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--rgb-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default="insample_train")
    p.add_argument("--invisible-thr", type=float, default=0.30)
    p.add_argument("--ignore-index", type=int, default=255)
    p.add_argument("--pad", type=int, default=48)
    p.add_argument("--per-page", type=int, default=30)
    p.add_argument("--cols", type=int, default=6)
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=str(out_dir / f"contact_sheet_{args.tag}.log"))

    z = np.load(args.cache, allow_pickle=True)
    tids = [str(t) for t in z["tids"]]
    objects = find_invisible_objects(
        z["probs"], z["labels"], tids,
        ignore_index=args.ignore_index, invisible_thr=args.invisible_thr,
    )
    meta = load_metadata(args.metadata)
    tid_region = dict(zip(meta["Tile_ID"], meta["RegionName"]))
    logger.info("%d invisible objects (max_prob < %.2f)", len(objects), args.invisible_thr)

    manifest = out_dir / f"invisible_manifest_{args.tag}.csv"
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile_id", "region", "area_px", "max_prob", "mean_prob", "bbox_y0y1x0x1"])
        for o in sorted(objects, key=lambda o: (tid_region.get(o["tile_id"], ""), -o["area_px"])):
            w.writerow([o["tile_id"], tid_region.get(o["tile_id"], "?"), o["area_px"],
                        o["max_prob"], o["mean_prob"], "_".join(map(str, o["bbox"]))])
    logger.info("wrote %s", manifest)

    render(objects, Path(args.rgb_dir), out_dir, args.tag,
           tid_region=tid_region, pad=args.pad, per_page=args.per_page, cols=args.cols)
    print(f"{len(objects)} invisible objects → contact sheets + {manifest.name} in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

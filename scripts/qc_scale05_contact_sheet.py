"""QC contact sheet for the 0.5x-scale re-stage (data.md §3.5).

Samples positive 0.5x tiles (biased toward tiles containing auto-converted
ignore features), renders RGB + label overlay (green=positive, red=ignore,
yellow outline=auto-converted ignore features), and reports label-composition
stats. For the user eyeball gate before the multiscale POC training runs.

Usage (inside rts-train:v2 with ADC):
  python scripts/qc_scale05_contact_sheet.py \
      --root gs://rts-mapping-v2/training/v1.0_scale05 \
      --out /outputs/staging_scale05_qc --n 30
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from shapely.geometry import box  # noqa: E402
from shapely.strtree import STRtree  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.scale05_tile_creation import DEFAULT_VECTORS, LabelVectors, _read_3857  # noqa: E402

logger = logging.getLogger("qc_scale05")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(args.out, exist_ok=True)

    md = pd.read_csv(f"{args.root}/metadata.csv", dtype={"Tile_ID": str})
    pos = md[md.TrainClass == "positive"]
    logger.info("metadata: %d tiles (%s)", len(md), md.TrainClass.value_counts().to_dict())

    # Re-classify vectors to find auto-converted ignore geometries.
    positive = [g for uri in DEFAULT_VECTORS["positive"] for g in _read_3857(uri).geometry]
    ignore = [g for uri in DEFAULT_VECTORS["ignore"] for g in _read_3857(uri).geometry]
    v = LabelVectors.classify(positive, ignore, arts_positive=[])
    conv_tree = STRtree(v.ignore_convert)

    # Which sampled tiles contain a converted feature? (bias half the sample there)
    rng = np.random.default_rng(args.seed)
    rows, conv_flags = [], []
    for _, r in pos.iterrows():
        rows.append(r)
    stats = {"with_converted": 0}
    tile_bounds: dict[str, tuple] = {}
    with_conv, without_conv = [], []
    for r in rows:
        # cheap test on the tile bbox from the raster header
        try:
            with rasterio.open(f"{args.root}/labels/{r.Tile_ID}.tif") as src:
                b = tuple(src.bounds)
        except rasterio.errors.RasterioIOError:
            continue
        tile_bounds[r.Tile_ID] = b
        if len(conv_tree.query(box(*b), predicate="intersects")):
            with_conv.append(r)
        else:
            without_conv.append(r)
    stats["with_converted"] = len(with_conv)
    logger.info("positive tiles containing auto-converted ignore: %d / %d",
                len(with_conv), len(with_conv) + len(without_conv))

    take = min(args.n // 2, len(with_conv))
    sample = list(rng.choice(len(with_conv), take, replace=False)) if take else []
    picked = [with_conv[i] for i in sample]
    take2 = args.n - len(picked)
    sample2 = list(rng.choice(len(without_conv), min(take2, len(without_conv)), replace=False))
    picked += [without_conv[i] for i in sample2]

    ncol = 5
    nrow = int(np.ceil(len(picked) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    axes = np.atleast_2d(axes)
    comp_rows = []
    for ax, r in zip(axes.ravel(), picked):
        tid = r.Tile_ID
        with rasterio.open(f"{args.root}/PLANET-RGB/{tid}.tif") as src:
            rgb = src.read().transpose(1, 2, 0)
        with rasterio.open(f"{args.root}/labels/{tid}.tif") as src:
            lab = src.read(1)
            b = src.bounds
        ax.imshow(rgb)
        overlay = np.zeros((*lab.shape, 4), dtype=np.float32)
        overlay[lab == 1] = [0, 1, 0, 0.35]
        overlay[lab == 255] = [1, 0, 0, 0.35]
        ax.imshow(overlay)
        # outline converted features
        for i in conv_tree.query(box(*b), predicate="intersects"):
            g = v.ignore_convert[int(i)]
            xs, ys = g.exterior.xy if g.geom_type == "Polygon" else g.geoms[0].exterior.xy
            px = [(x - b.left) / (b.right - b.left) * lab.shape[1] for x in xs]
            py = [(b.top - y) / (b.top - b.bottom) * lab.shape[0] for y in ys]
            ax.plot(px, py, color="yellow", lw=1.2)
        n1, n255 = int((lab == 1).sum()), int((lab == 255).sum())
        ax.set_title(f"{tid}\n1:{n1}px 255:{n255}px", fontsize=7)
        ax.axis("off")
        comp_rows.append({"Tile_ID": tid, "pos_px": n1, "ignore_px": n255,
                          "has_converted": bool(len(conv_tree.query(box(*b), predicate='intersects')))})
    for ax in axes.ravel()[len(picked):]:
        ax.axis("off")
    fig.suptitle("0.5x re-stage QC — green=positive, red=ignore, yellow=auto-converted ignore",
                 fontsize=12)
    fig.tight_layout()
    out_png = os.path.join(args.out, "scale05_contact_sheet.png")
    fig.savefig(out_png, dpi=140)
    pd.DataFrame(comp_rows).to_csv(os.path.join(args.out, "scale05_contact_sheet.csv"), index=False)
    logger.info("wrote %s", out_png)

    # Dataset-level label composition
    tot = {"tiles": 0, "pos_px": 0, "ign_px": 0}
    sub = pos.sample(min(300, len(pos)), random_state=args.seed)
    for tid in sub.Tile_ID:
        try:
            with rasterio.open(f"{args.root}/labels/{tid}.tif") as src:
                lab = src.read(1)
        except rasterio.errors.RasterioIOError:
            continue
        tot["tiles"] += 1
        tot["pos_px"] += int((lab == 1).sum())
        tot["ign_px"] += int((lab == 255).sum())
    denom = tot["tiles"] * 512 * 512
    logger.info("label composition over %d sampled positives: positive %.3f%%, ignore %.3f%%",
                tot["tiles"], 100 * tot["pos_px"] / denom, 100 * tot["ign_px"] / denom)


if __name__ == "__main__":
    main()

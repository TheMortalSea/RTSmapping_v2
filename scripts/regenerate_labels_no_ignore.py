"""Regenerate training labels WITHOUT the manual ignore-region polygons (ablation).

First principle (user): the manual ignore-region polygons mask out high-uncertainty
areas so the model is not forced to learn from them. This ablation removes that mask to
test whether it *helps* learning. We keep every background(0) and positive(1) pixel
bit-identical to the deploy baseline and reclassify ONLY the on-disk 255 (manual-ignore)
pixels to their true class:

    new_label = where(old == 255, positive_union_raster, old)

so a formerly-ignored pixel becomes 1 iff some positive (RTS-truth) polygon delineates it,
else 0. Output tiles therefore contain NO 255 (the loss-time boundary_ignore_w2 dilation is
applied separately at load time and is unchanged — it is NOT a manual ignore region).

Outputs:
  - <out_dir>/labels/<id>.tif        regenerated single-band uint8 labels (values in {0,1})
  - <out_dir>/regen_report.json      per-run stats (tiles, 255 px, 255->1, 255->0)
  - <out_dir>/previews/*.png         side-by-side RGB | with-ignore | without-ignore samples
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("regen")

POSITIVE_LAYERS = [
    "batch2_positive_labels.geojson",
    "batch3_positive_labels.geojson",
    "training_positive_labels_03_03_2026.geojson",
]
IGNORE_LAYERS = [
    "batch2_ignore.json",
    "batch3_ignore_regions.geojson",
    "ignore_regions_3Mar2026.geojson",
]


def load_union(geojson_dir: Path, names: list[str]) -> gpd.GeoDataFrame:
    """Concatenate vector layers into one GeoDataFrame in EPSG:4326 with a spatial index."""
    parts = []
    for name in names:
        path = geojson_dir / name
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        gdf = gdf.to_crs(4326)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        gdf["geometry"] = gdf.geometry.buffer(0)  # fix invalid rings
        parts.append(gdf[["geometry"]])
        log.info("  %s: %d features", name, len(gdf))
    union = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)
    union.sindex  # build index
    return union


def rasterize_layer_on_tile(union_4326: gpd.GeoDataFrame, src, value: int) -> np.ndarray:
    """Rasterize the layer onto a tile's exact grid (src CRS+transform), burning `value`."""
    tile_bounds_native = box(*src.bounds)
    tile_bounds_4326 = (
        gpd.GeoSeries([tile_bounds_native], crs=src.crs).to_crs(4326).iloc[0]
    )
    idx = list(union_4326.sindex.query(tile_bounds_4326, predicate="intersects"))
    if not idx:
        return np.zeros((src.height, src.width), dtype=np.uint8)
    subset = union_4326.iloc[idx].to_crs(src.crs)
    subset = subset[subset.intersects(tile_bounds_native)]
    if subset.empty:
        return np.zeros((src.height, src.width), dtype=np.uint8)
    return rasterize(
        [(geom, value) for geom in subset.geometry],
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype=np.uint8,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/outputs/v1.0/data_local")
    ap.add_argument("--geojson-dir", default="/outputs/v1.0/staging/ignore_ablation/geojsons")
    ap.add_argument("--out-dir", default="/outputs/v1.0/staging/ignore_ablation")
    ap.add_argument("--n-previews", type=int, default=12)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    geojson_dir = Path(args.geojson_dir)
    out_dir = Path(args.out_dir)
    labels_in = data_root / "labels"
    labels_out = out_dir / "labels"
    previews = out_dir / "previews"
    labels_out.mkdir(parents=True, exist_ok=True)
    previews.mkdir(parents=True, exist_ok=True)

    log.info("Loading positive union ...")
    pos_union = load_union(geojson_dir, POSITIVE_LAYERS)
    log.info("Positive union: %d features", len(pos_union))

    tiles = sorted(labels_in.glob("*.tif"))
    log.info("Found %d label tiles", len(tiles))

    stats = {
        "n_tiles": len(tiles),
        "n_tiles_with_255": 0,
        "n_255_px": 0,
        "n_255_to_1": 0,
        "n_255_to_0": 0,
        "n_tiles_residual_255": 0,
    }
    preview_candidates = []  # (n_255, n_flip_to_1, tile_id)

    for i, tpath in enumerate(tiles):
        tid = tpath.stem
        with rasterio.open(tpath) as src:
            old = src.read(1)
            profile = src.profile.copy()
            ignore_mask = old == 255
            n255 = int(ignore_mask.sum())
            if n255 == 0:
                new = old.copy()
            else:
                pos_raster = rasterize_layer_on_tile(pos_union, src, 1)
                new = old.copy()
                flip_to_1 = ignore_mask & (pos_raster == 1)
                new[ignore_mask] = 0
                new[flip_to_1] = 1
                stats["n_tiles_with_255"] += 1
                stats["n_255_px"] += n255
                stats["n_255_to_1"] += int(flip_to_1.sum())
                stats["n_255_to_0"] += int(n255 - flip_to_1.sum())
                preview_candidates.append((n255, int(flip_to_1.sum()), tid))

        if int((new == 255).sum()) > 0:
            stats["n_tiles_residual_255"] += 1
        profile.update(dtype="uint8", count=1, nodata=None)
        with rasterio.open(labels_out / f"{tid}.tif", "w", **profile) as dst:
            dst.write(new[np.newaxis, :, :])
            dst.set_band_description(1, "Mask: 0=background 1=positive (no manual ignore)")
        if (i + 1) % 200 == 0:
            log.info("  processed %d/%d", i + 1, len(tiles))

    with open(out_dir / "regen_report.json", "w") as f:
        json.dump(stats, f, indent=2)
    log.info("STATS: %s", json.dumps(stats, indent=2))

    # ---- previews: prefer tiles that exercise BOTH outcomes (some 255->1 and some 255->0)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    # label cmap: 0 black, 1 red, 255 yellow
    lab_cmap = ListedColormap(["#101010", "#e02020", "#f2e000"])
    lab_norm = BoundaryNorm([-0.5, 0.5, 1.5, 255.5], lab_cmap.N)

    mixed = sorted([c for c in preview_candidates if c[1] > 0 and c[0] - c[1] > 0],
                   key=lambda c: -min(c[1], c[0] - c[1]))
    pure = sorted([c for c in preview_candidates if c not in mixed],
                  key=lambda c: -c[0])
    chosen = (mixed + pure)[: args.n_previews]

    rgb_dir = data_root / "PLANET-RGB"
    for n255, nflip, tid in chosen:
        with rasterio.open(labels_in / f"{tid}.tif") as s:
            old = s.read(1)
        with rasterio.open(labels_out / f"{tid}.tif") as s:
            new = s.read(1)
        rgb_path = rgb_dir / f"{tid}.tif"
        if rgb_path.exists():
            with rasterio.open(rgb_path) as s:
                rgb = s.read([1, 2, 3]).transpose(1, 2, 0).astype(np.float32)
            p2, p98 = np.percentile(rgb, (2, 98))
            rgb = np.clip((rgb - p2) / max(p98 - p2, 1e-6), 0, 1)
        else:
            rgb = np.zeros((old.shape[0], old.shape[1], 3), dtype=np.float32)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
        axes[0].imshow(rgb)
        axes[0].set_title(f"RGB\n{tid}")
        axes[1].imshow(old, cmap=lab_cmap, norm=lab_norm, interpolation="nearest")
        axes[1].set_title(f"WITH ignore (deploy)\n255px={n255}  (yellow=ignore)")
        axes[2].imshow(new, cmap=lab_cmap, norm=lab_norm, interpolation="nearest")
        axes[2].set_title(f"WITHOUT ignore (ablation)\n255->1: {nflip}   255->0: {n255 - nflip}")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(previews / f"{tid}.png", dpi=90, bbox_inches="tight")
        plt.close(fig)
    log.info("Wrote %d previews to %s", len(chosen), previews)


if __name__ == "__main__":
    main()

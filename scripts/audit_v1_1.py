"""v1.1 data audit — the checks the 2026-06-13 QC did NOT run. Report-only.

`qc_full_dataset.py` certified v1.0 labels "flawless" but only checked empty/full-frame/
invalid-value/duplicates. This adds the latent-problem checks for the data-v1.1 re-stage:

  A. **Sub-MMU slivers** — positive blob-size distribution; how many blobs/tiles a given
     MMU cutoff removes (via the shared `data.label_cleaning` primitive). Pixels are
     4.777 m → area_m2 = px * 22.82.
  B. **Mostly-ignore / empty positives** — positive tiles with 0 RTS pixels, or a tiny
     RTS fraction of valid pixels (little usable signal).
  C. **Negative-pool contamination** — negative tiles whose footprint intersects a
     refined RTS polygon (should be positive/ignore, not a clean negative).
  D. **Duplicates** — duplicate Tile_ID or centroid in metadata (re-confirm).

Emits `known_issues_v1.1.json` (lists + summary). FACTS ONLY — no fixes here.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.label_cleaning import clean_positive_label  # noqa: E402
from data.splits import load_metadata  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

M2_PER_PX = 4.77731426716 ** 2  # ≈ 22.82 m² per pixel (EPSG:3857, PlanetScope basemap)


def audit_slivers(labels_dir: str, pos_tids: list[str], cutoffs: list[int]) -> dict:
    """Blob-size distribution + per-cutoff removal counts across positive labels."""
    sizes: list[int] = []
    per_tile_blobs: dict[str, list[int]] = {}
    for tid in pos_tids:
        p = os.path.join(labels_dir, f"{tid}.tif")
        if not os.path.exists(p):
            continue
        with rasterio.open(p) as s:
            lab = s.read(1)
        gt = lab == 1
        if not gt.any():
            continue
        lbl, n = ndimage.label(gt)
        bs = ndimage.sum(gt, lbl, index=np.arange(1, n + 1)).astype(int).tolist()
        sizes.extend(bs)
        per_tile_blobs[tid] = bs
    a = np.array(sizes)
    pct = {p: int(np.percentile(a, p)) for p in (1, 5, 10, 25, 50, 90)} if a.size else {}
    removal = {}
    for t in cutoffs:
        removed = int((a < t).sum())
        # tiles that would lose ALL positive blobs at this cutoff (become empty positives)
        emptied = sum(1 for bs in per_tile_blobs.values() if max(bs) < t)
        removal[t] = {"blobs_removed": removed, "pct_of_blobs": round(100 * removed / a.size, 2),
                      "tiles_emptied": emptied, "area_m2_below": round(t * M2_PER_PX, 1)}
    return {"n_blobs": int(a.size), "n_positive_tiles": len(per_tile_blobs),
            "size_percentiles_px": pct, "removal_by_cutoff": removal}


def audit_mostly_ignore(labels_dir: str, pos_tids: list[str], ignore_dom_frac: float) -> dict:
    """Positive tiles with 0 RTS px (empty) or ignore-dominated (ignore_frac > threshold).

    'Ignore-dominated' is the real problem signal (a positive tile mostly masked out);
    a small RTS *fraction* is normal (slumps are small), so it is NOT flagged.
    """
    empty_positive, ignore_dominated = [], []
    n = 512 * 512
    for tid in pos_tids:
        p = os.path.join(labels_dir, f"{tid}.tif")
        if not os.path.exists(p):
            continue
        with rasterio.open(p) as s:
            lab = s.read(1)
        if int((lab == 1).sum()) == 0:
            empty_positive.append(tid)
        elif (lab == 255).sum() / n > ignore_dom_frac:
            ignore_dominated.append(tid)
    return {"empty_positive_tiles": empty_positive,
            "ignore_dominated_tiles": ignore_dominated,
            "ignore_dom_frac": ignore_dom_frac,
            "n_empty_positive": len(empty_positive),
            "n_ignore_dominated": len(ignore_dominated)}


def audit_contamination(metadata, vectors_dir: str, half_m: float) -> dict:
    """Negative tiles whose footprint intersects a refined RTS polygon."""
    import geopandas as gpd
    from shapely.geometry import box

    polys = []
    for p in sorted(glob.glob(os.path.join(vectors_dir, "*.geojson"))):
        g = gpd.read_file(p).to_crs(3857)
        polys.append(g[["geometry"]])
    pos = gpd.GeoDataFrame(pd_concat(polys), crs=3857)

    neg = metadata[metadata["TrainClass"] == "negative"].copy()
    pts = gpd.GeoDataFrame(
        neg[["Tile_ID"]],
        geometry=gpd.points_from_xy(neg["centroid_lon"], neg["centroid_lat"]),
        crs=4326,
    ).to_crs(3857)
    pts["geometry"] = pts.geometry.apply(lambda p: box(p.x - half_m, p.y - half_m,
                                                        p.x + half_m, p.y + half_m))
    hit = gpd.sjoin(pts, pos, predicate="intersects", how="inner")
    tids = sorted(set(hit["Tile_ID"]))
    return {"n_negatives_checked": len(neg), "n_contaminated": len(tids),
            "contaminated_negative_tiles": tids, "footprint_half_m": half_m}


def pd_concat(frames):
    import pandas as pd
    return pd.concat(frames, ignore_index=True)


def audit_duplicates(metadata) -> dict:
    dup_id = metadata["Tile_ID"][metadata["Tile_ID"].duplicated()].tolist()
    ll = metadata[["centroid_lat", "centroid_lon"]].round(6)
    dup_ll = metadata["Tile_ID"][ll.duplicated()].tolist()
    return {"n_dup_tile_id": len(dup_id), "n_dup_centroid": len(dup_ll),
            "dup_tile_ids": dup_id[:50], "dup_centroid_tids": dup_ll[:50]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--metadata", required=True)
    p.add_argument("--labels-dir", required=True)
    p.add_argument("--positive-vectors", required=True, help="dir of refined RTS *.geojson")
    p.add_argument("--out", required=True)
    p.add_argument("--cutoffs", default="1,2,5,10,20,50")
    p.add_argument("--ignore-dom-frac", type=float, default=0.5)
    p.add_argument("--footprint-half-m", type=float, default=1223.0)  # 2446 m tile / 2
    p.add_argument("--skip-contamination", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=str(out_dir / "audit_v1_1.log"))

    meta = load_metadata(args.metadata)
    pos_tids = meta.loc[meta["TrainClass"] == "positive", "Tile_ID"].tolist()
    cutoffs = [int(x) for x in args.cutoffs.split(",")]

    logger.info("Auditing %d positive tiles + %d total", len(pos_tids), len(meta))
    slivers = audit_slivers(args.labels_dir, pos_tids, cutoffs)
    mostly = audit_mostly_ignore(args.labels_dir, pos_tids, args.ignore_dom_frac)
    dups = audit_duplicates(meta)
    contam = ({} if args.skip_contamination
              else audit_contamination(meta, args.positive_vectors, args.footprint_half_m))

    report = {
        "_generated_for": "data-v1.1 re-stage audit (checks omitted by qc_full_dataset.py)",
        "m2_per_px": round(M2_PER_PX, 2),
        "slivers": slivers,
        "mostly_ignore": mostly,
        "negative_pool_contamination": contam,
        "duplicates": dups,
    }
    outp = out_dir / "known_issues_v1.1.json"
    outp.write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s", outp)

    print(json.dumps({
        "slivers_removal_by_cutoff": slivers["removal_by_cutoff"],
        "n_empty_positive": mostly["n_empty_positive"],
        "n_ignore_dominated": mostly["n_ignore_dominated"],
        "n_contaminated_negatives": contam.get("n_contaminated"),
        "n_dup_id": dups["n_dup_tile_id"], "n_dup_centroid": dups["n_dup_centroid"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

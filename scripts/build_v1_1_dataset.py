"""Build the v1.1 dataset metadata + the vjn7 promoted label (data-correctness track).

v1.1 applies three *unambiguous* row-level corrections on top of the v1.0 snapshot
metadata (the small-blob question is handled separately by the Minimum Mapping Unit
metric fix, `data.apply_min_mapping_unit` — NOT by deleting labels):

  1. restore 28 wrongly-dropped positives (rows taken from the region-hotfixed source
     metadata so their RegionName is the corrected ecoregion → correct split),
  2. drop 49 all-black negatives (zero information),
  3. promote `vjn7wxyufczs` (a negative whose footprint contains a real ~13 ha slump)
     → positive, and rasterize its RTS label from the refined positive polygons.

Exact tile lists live in `known_issues_v1.0.json` (`next_restage_actions`). Writes
`metadata_v1_1.csv`, `labels/vjn7wxyufczs.tif`, and `manifest.json` under --out. Does
NOT re-split, recompute normalization, upload to GCS, or retrain — those are the
version-bump ritual steps run separately once a v1.1 retrain is commissioned
(`create_splits.py`, `compute_normalization_stats.py`, snapshot upload, `train.py`).

Run inside rts-train:v2 with ADC (google.cloud.storage) + the refined positive
GeoJSONs available locally.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger("build_v1_1")

SRC_BUCKET = "abrupt_thaw"
SRC_PREFIX = "RTS_MODEL_V2/DATA/TRAINING_DATA/"
PROJECT = "pdg-project-406720"
CONTAMINATED_NEG = "vjn7wxyufczs"


def _download(client, bucket: str, blob: str, dst: str) -> str:
    client.bucket(bucket).blob(blob).download_to_filename(dst)
    return dst


def build(
    v1_metadata_csv: str,
    known_issues_json: str,
    vectors: list[str],
    out_dir: str,
) -> dict:
    """Build v1.1 metadata + the vjn7 label. Returns the manifest dict."""
    from google.cloud import storage

    out = Path(out_dir)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp()
    client = storage.Client(project=PROJECT)

    ki = json.loads(Path(known_issues_json).read_text())["next_restage_actions"]
    restore_pos = ki["restore_positives_wrongly_dropped"]
    black = ki["drop_negatives_black"]

    # Region-hotfixed source metadata (corrected ecoregion → correct split for the restores).
    hf_path = _download(client, SRC_BUCKET, SRC_PREFIX + "metadata_region_hotfix.csv", tmp + "/hf.csv")
    hf = pd.read_csv(hf_path, dtype={"Tile_ID": str, "UIDs": str})

    base = pd.read_csv(v1_metadata_csv, dtype={"Tile_ID": str, "UIDs": str})
    n0 = len(base)
    base = base[~base["Tile_ID"].isin(black)].copy()                 # (2) drop 49 black
    base.loc[base["Tile_ID"] == CONTAMINATED_NEG, "TrainClass"] = "positive"  # (3) promote
    base.loc[base["Tile_ID"] == CONTAMINATED_NEG, "Version"] = "1.1"
    add = hf[hf["Tile_ID"].isin(restore_pos)].copy()                 # (1) restore 28
    add["Version"] = "1.1"
    add = add[base.columns.tolist()]
    v11 = pd.concat([base, add], ignore_index=True)
    assert v11["Tile_ID"].duplicated().sum() == 0, "duplicate Tile_ID in v1.1 metadata"
    v11.to_csv(out / "metadata_v1_1.csv", index=False)

    # Rasterize the vjn7 RTS label from the refined positive polygons onto its tile grid.
    rgb_path = _download(client, SRC_BUCKET, SRC_PREFIX + f"PLANET-RGB/{CONTAMINATED_NEG}.tif",
                         tmp + "/vjn7_rgb.tif")
    polys = gpd.GeoDataFrame(
        pd.concat([gpd.read_file(v).to_crs(3857)[["geometry"]] for v in vectors], ignore_index=True),
        crs=3857,
    )
    with rasterio.open(rgb_path) as s:
        transform, (H, W), bounds = s.transform, s.shape, s.bounds
    hit = polys[polys.intersects(box(*bounds))]
    lab = rasterize([(g, 1) for g in hit.geometry], out_shape=(H, W), transform=transform,
                    fill=0, dtype="uint8", all_touched=False)
    prof = dict(count=1, dtype="uint8", height=H, width=W, crs="EPSG:3857",
                transform=transform, compress="deflate", nodata=None)
    with rasterio.open(out / "labels" / f"{CONTAMINATED_NEG}.tif", "w", **prof) as d:
        d.write(lab, 1)

    pos = int((v11["TrainClass"] == "positive").sum())
    neg = int((v11["TrainClass"] == "negative").sum())
    manifest = {
        "base_rows": int(n0), "v1_1_rows": int(len(v11)),
        "restored_positives": len(restore_pos), "dropped_black_negatives": len(black),
        "promoted_contaminated_neg": [CONTAMINATED_NEG],
        "counts": {"positive": pos, "negative": neg, "total": pos + neg},
        "restored_regions": add["RegionName"].value_counts().to_dict(),
        "vjn7_label": {"rts_px": int((lab == 1).sum()),
                       "rts_ha": round(float((lab == 1).sum()) * 4.77731426716 ** 2 / 1e4, 2),
                       "intersecting_polygons": int(len(hit))},
        "note": "metadata + vjn7 label only; re-split / norm / GCS snapshot / retrain run separately.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("v1.1: %d rows (%d pos / %d neg); vjn7 label %d px",
                len(v11), pos, neg, int((lab == 1).sum()))
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--v1-metadata", required=True, help="v1.0 snapshot metadata.csv (local)")
    p.add_argument("--known-issues", required=True, help="known_issues_v1.0.json")
    p.add_argument("--vectors", nargs="+", required=True, help="refined positive *.geojson files")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    setup_logging(level="INFO", log_file=str(Path(args.out) / "build_v1_1.log"))
    manifest = build(args.v1_metadata, args.known_issues, args.vectors, args.out)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

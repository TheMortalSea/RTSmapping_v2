# PLANET IS BGR not RGB — bands are 1=Blue, 2=Green, 3=Red

import os
import sys
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box, Point
from shapely.ops import transform as shapely_transform
from shapely.strtree import STRtree
from google.colab import auth
from google.cloud import storage
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import pyproj

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def require_env(name):
    val = os.environ.get(name)
    if val is None:
        print(f"ERROR: Required environment variable '{name}' is not set.")
        sys.exit(1)
    return val

BUCKET               = require_env("BUCKET")
INPUT_PREFIX         = require_env("INPUT_PREFIX")
POLYGON_GEOJSON_BLOB = require_env("POLYGON_GEOJSON_BLOB")
DATA_ROOT            = require_env("DATA_ROOT")
METADATA_SUBREGIONS  = require_env("METADATA_SUBREGIONS")
WORK_DIR             = require_env("WORK_DIR")
MAX_WORKERS          = int(require_env("MAX_WORKERS"))

_target      = os.environ.get("TARGET_TILES")
TARGET_TILES = int(_target) if _target else None

TILE_SIZE       = 512
WORKING_CRS     = "EPSG:6933"
TILE_CRS        = "EPSG:3857"   # Planet basemaps are Web Mercator
RGB_PREFIX      = f"{DATA_ROOT}/PLANET-RGB/"
METADATA_PREFIX = f"{DATA_ROOT}/metadata/"

print("Configuration:")
print(f"  BUCKET:        {BUCKET}")
print(f"  INPUT_PREFIX:  {INPUT_PREFIX}")
print(f"  RGB output:    gs://{BUCKET}/{RGB_PREFIX}")
print(f"  MAX_WORKERS:   {MAX_WORKERS}")
print(f"  TARGET_TILES:  {TARGET_TILES or 'all'}")

# ---------------------------------------------------------------------------
# Auth + GCS
# ---------------------------------------------------------------------------

auth.authenticate_user()

os.makedirs(f"{WORK_DIR}/input",  exist_ok=True)
os.makedirs(f"{WORK_DIR}/output", exist_ok=True)

client = storage.Client()
bucket = client.bucket(BUCKET)

# ---------------------------------------------------------------------------
# Load + filter polygons
# ---------------------------------------------------------------------------

print("\nDownloading datasets from GCS...")

polygon_local = f"{WORK_DIR}/input/polygons.geojson"
regions_local = f"{WORK_DIR}/input/regions.geojson"

bucket.blob(POLYGON_GEOJSON_BLOB).download_to_filename(polygon_local)
bucket.blob(METADATA_SUBREGIONS).download_to_filename(regions_local)

gdf_polygons = gpd.read_file(polygon_local)
gdf_polygons = gdf_polygons[gdf_polygons["TrainClass"] == "Negative"].copy()
gdf_polygons = gdf_polygons.to_crs(WORKING_CRS)
gdf_polygons["geometry"] = gdf_polygons["geometry"].buffer(0)  # fix invalid geometries once
print(f"  Negative polygons:  {len(gdf_polygons)}")

gdf_regions       = gpd.read_file(regions_local)
gdf_regions_wgs84 = gdf_regions.to_crs("EPSG:4326")
gdf_regions_work  = gdf_regions.to_crs(WORKING_CRS)

if "ECO_NAME" not in gdf_regions.columns:
    print(f"ERROR: 'ECO_NAME' not found. Available: {list(gdf_regions.columns)}")
    sys.exit(1)

print(f"  Ecoregions:         {len(gdf_regions)}")

# ---------------------------------------------------------------------------
# Assign ecoregions to polygons via spatial join
# ---------------------------------------------------------------------------

print("\nAssigning ecoregions to polygons...")

poly_centroids = gdf_polygons.copy()
poly_centroids["geometry"] = gdf_polygons.geometry.centroid

joined = gpd.sjoin(
    poly_centroids[["geometry"]],
    gdf_regions_work[["geometry", "ECO_NAME"]],
    how="left", predicate="within",
)
gdf_polygons["ECO_NAME"] = joined["ECO_NAME"].values

# Nearest fallback for any polygon centroid that missed all regions
missing = gdf_polygons["ECO_NAME"].isna()
if missing.any():
    region_tree = STRtree(gdf_regions_work.geometry.centroid.values)
    for idx in gdf_polygons[missing].index:
        nearest_i = region_tree.nearest(gdf_polygons.loc[idx, "geometry"].centroid)
        gdf_polygons.at[idx, "ECO_NAME"] = gdf_regions_work.iloc[nearest_i]["ECO_NAME"]
    print(f"  {missing.sum()} polygons assigned via nearest centroid")

# ---------------------------------------------------------------------------
# Stratified sampling — equal polygons per ecoregion
# ---------------------------------------------------------------------------

eco_groups   = gdf_polygons.groupby("ECO_NAME")
n_ecoregions = len(eco_groups)
per_region   = max(1, TARGET_TILES // n_ecoregions) if TARGET_TILES else None

print(f"\nSampling: {TARGET_TILES or 'all'} tiles across {n_ecoregions} ecoregions"
      + (f" = {per_region} per region" if per_region else ""))

sampled_parts = []
for _, group in eco_groups:
    if per_region is None or len(group) <= per_region:
        sampled_parts.append(group)
    else:
        sampled_parts.append(group.sample(n=per_region, random_state=42))

gdf_sampled = pd.concat(sampled_parts).reset_index(drop=True)
print(f"  Sampled polygons:   {len(gdf_sampled)}")

sampled_local = f"{WORK_DIR}/input/polygons_sampled.geojson"
gdf_sampled.to_file(sampled_local, driver="GeoJSON")

# ---------------------------------------------------------------------------
# STRtree for fast region lookup (used in main process after each tile)
# ---------------------------------------------------------------------------

_region_tree = STRtree(gdf_regions_wgs84.geometry.centroid.values)

def find_nearest_region(tile_centroid_wgs84: Point) -> str:
    nearest_i = _region_tree.nearest(tile_centroid_wgs84)
    return gdf_regions_wgs84.iloc[nearest_i]["ECO_NAME"]

# ---------------------------------------------------------------------------
# Pre-filter tiles by bbox — skip tiles with no polygon overlap before downloading
#
# Planet tile path structure:
#   .../101/1471/global_quarterly_2024q3_mosaic_101-1471_quad.tif
#   parts[-3] = zoom, parts[-2] = x, filename contains y after last "-"
# ---------------------------------------------------------------------------

EARTH_HALF_CIRC = 20037508.3427892

def tile_path_to_bbox_3857(blob_path: str):
    parts    = blob_path.rstrip("/").split("/")
    zoom     = int(parts[-3])
    x        = int(parts[-2])
    y        = int(parts[-1].split("_quad")[0].split("-")[-1])
    tile_m   = (EARTH_HALF_CIRC * 2) / (2 ** zoom)
    xmin     = -EARTH_HALF_CIRC + x       * tile_m
    xmax     = -EARTH_HALF_CIRC + (x + 1) * tile_m
    ymax     =  EARTH_HALF_CIRC - y       * tile_m
    ymin     =  EARTH_HALF_CIRC - (y + 1) * tile_m
    return box(xmin, ymin, xmax, ymax)


print(f"\nListing source tiles at: gs://{BUCKET}/{INPUT_PREFIX}")

all_tile_blobs = [
    b.name for b in bucket.list_blobs(prefix=INPUT_PREFIX)
    if b.name.endswith("_quad.tif")
]
print(f"  Total tiles found:     {len(all_tile_blobs)}")

sampled_3857 = gdf_sampled.to_crs(TILE_CRS)
sampled_tree = STRtree(sampled_3857.geometry.values)

tiles_intersecting = []
for blob_path in all_tile_blobs:
    try:
        bbox = tile_path_to_bbox_3857(blob_path)
        if sampled_tree.query(bbox, predicate="intersects").size > 0:
            tiles_intersecting.append(blob_path)
    except Exception:
        continue

print(f"  After bbox pre-filter: {len(tiles_intersecting)} tiles overlap sampled polygons")

# ---------------------------------------------------------------------------
# Resume: skip already-processed tiles
# ---------------------------------------------------------------------------

metadata_blob_path = f"{METADATA_PREFIX}metadata.csv"
existing_blob      = bucket.blob(metadata_blob_path)
processed_paths    = set()
existing_df        = None

if existing_blob.exists():
    existing_local = f"{WORK_DIR}/input/metadata_existing.csv"
    existing_blob.download_to_filename(existing_local)
    existing_df    = pd.read_csv(existing_local)
    if "source_blob" in existing_df.columns:
        processed_paths = set(existing_df["source_blob"].dropna())

existing_uids = []
if existing_df is not None and "Tile_ID" in existing_df.columns:
    existing_uids = [int(v) for v in existing_df["Tile_ID"] if str(v).isdigit()]
next_uid = max(existing_uids) + 1 if existing_uids else 1

tiles_to_process = [b for b in tiles_intersecting if b not in processed_paths]
print(f"  After resume filter:   {len(tiles_to_process)} tiles remaining")
print(f"  Next UID:              {next_uid:06d}")

if not tiles_to_process:
    print("All tiles already processed.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Worker — init caches GCS client, spatial index, and CRS transformers
# ---------------------------------------------------------------------------

def worker_init(sampled_polygon_path: str):
    global _gdf_polygons, _polygon_sindex
    global _gcs_bucket
    global _project_to_work, _project_to_wgs84

    _gdf_polygons   = gpd.read_file(sampled_polygon_path)  # already WORKING_CRS + buffered
    _polygon_sindex = _gdf_polygons.sindex

    _gcs_client = storage.Client()
    _gcs_bucket = _gcs_client.bucket(BUCKET)

    # Planet tiles are always EPSG:3857 — cache transformers once per worker
    _project_to_work  = pyproj.Transformer.from_crs(3857, WORKING_CRS, always_xy=True).transform
    _project_to_wgs84 = pyproj.Transformer.from_crs(3857, 4326,        always_xy=True).transform


def process_single_tile(blob_path: str, work_dir: str):
    parts     = blob_path.rstrip("/").split("/")
    base_name = "_".join(parts[-3:]).replace(".tif", "")

    local_input   = f"{work_dir}/input/{base_name}.tif"
    local_rgb_out = f"{work_dir}/output/rgb_{base_name}.tif"

    try:
        _gcs_bucket.blob(blob_path).download_to_filename(local_input)

        with rasterio.open(local_input) as src:
            win_w = min(TILE_SIZE, src.width)
            win_h = min(TILE_SIZE, src.height)
            win   = Window(0, 0, win_w, win_h)

            tile_bounds      = rasterio.transform.array_bounds(
                win_h, win_w, rasterio.windows.transform(win, src.transform)
            )
            tile_bbox_native = box(*tile_bounds)
            tile_bbox_work   = shapely_transform(_project_to_work, tile_bbox_native)

            candidate_idx = list(_polygon_sindex.intersection(tile_bbox_work.bounds))
            if not candidate_idx:
                return None
            if not _gdf_polygons.iloc[candidate_idx].intersects(tile_bbox_work).any():
                return None

            try:
                rgb_data = src.read(indexes=[1, 2, 3], window=win)
            except Exception:
                return None

            if src.nodata is not None and (rgb_data == src.nodata).all():
                return None

            chip_tf        = rasterio.windows.transform(win, src.transform)
            centroid_wgs84 = shapely_transform(_project_to_wgs84, tile_bbox_native.centroid)

        profile = dict(
            driver="GTiff", count=3, dtype=rgb_data.dtype,
            width=win_w, height=win_h,
            crs=src.crs, transform=chip_tf, compress="LZW",
        )
        with rasterio.open(local_rgb_out, "w", **profile) as dst:
            dst.write(rgb_data)
            dst.set_band_description(1, "Blue")
            dst.set_band_description(2, "Green")
            dst.set_band_description(3, "Red")

        return (local_rgb_out, centroid_wgs84.y, centroid_wgs84.x)

    finally:
        if os.path.exists(local_input):
            os.remove(local_input)

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

print(f"\nProcessing {len(tiles_to_process)} tiles with {MAX_WORKERS} workers...\n")

success_count = 0
skip_count    = 0
error_count   = 0
metadata_rows = []

with concurrent.futures.ProcessPoolExecutor(
    max_workers=MAX_WORKERS,
    initializer=worker_init,
    initargs=(sampled_local,),
) as executor:

    future_to_blob = {
        executor.submit(process_single_tile, blob, WORK_DIR): blob
        for blob in tiles_to_process
    }

    with tqdm(total=len(tiles_to_process), desc="tiles", unit="tile") as pbar:
        for future in concurrent.futures.as_completed(future_to_blob):
            blob_path = future_to_blob[future]
            try:
                result = future.result()
                if result is not None:
                    local_rgb, centroid_lat, centroid_lon = result
                    uid_str = f"{next_uid:06d}"

                    bucket.blob(f"{RGB_PREFIX}{uid_str}.tif").upload_from_filename(local_rgb)
                    os.remove(local_rgb)

                    region_name = find_nearest_region(Point(centroid_lon, centroid_lat))

                    metadata_rows.append({
                        "Tile_ID":      uid_str,
                        "centroid_lat": round(centroid_lat, 6),
                        "centroid_lon": round(centroid_lon, 6),
                        "TrainClass":   "negative",
                        "RegionName":   region_name,
                        "UIDs":         9999,
                        "source_blob":  blob_path,
                    })

                    next_uid      += 1
                    success_count += 1
                    tqdm.write(f"✓ {uid_str}  {blob_path.split('/')[-1]}")
                else:
                    skip_count += 1

            except Exception as exc:
                error_count += 1
                tqdm.write(f"✗ {blob_path.split('/')[-1]} — {exc}")

            pbar.update(1)

# ---------------------------------------------------------------------------
# Write metadata
# ---------------------------------------------------------------------------

if metadata_rows:
    metadata_df = pd.DataFrame(
        metadata_rows,
        columns=["Tile_ID", "centroid_lat", "centroid_lon",
                 "TrainClass", "RegionName", "UIDs", "source_blob"],
    )
    local_csv = f"{WORK_DIR}/output/metadata.csv"

    if existing_df is not None:
        metadata_df = pd.concat([existing_df, metadata_df], ignore_index=True)
        print(f"\nAppended {len(metadata_rows)} rows → {len(metadata_df)} total")
    else:
        print(f"\nNew metadata CSV with {len(metadata_df)} rows")

    metadata_df.to_csv(local_csv, index=False)
    bucket.blob(metadata_blob_path).upload_from_filename(local_csv)
    os.remove(local_csv)
    print(f"Metadata → gs://{BUCKET}/{metadata_blob_path}")
else:
    print("\nNo successful tiles — metadata not written.")

print(f"\n{success_count} written | {skip_count} skipped | {error_count} errors")
print(f"RGB chips: gs://{BUCKET}/{RGB_PREFIX}")
print(f"Metadata:  gs://{BUCKET}/{METADATA_PREFIX}")
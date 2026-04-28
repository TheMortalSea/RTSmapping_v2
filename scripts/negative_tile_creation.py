
# PLANET IS BGR not RGB read those respectively

import os
import sys
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box, Point
from shapely.ops import transform as shapely_transform
from google.colab import auth
from google.cloud import storage
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import pyproj

# Config (read from environment)

def require_env(name):
    val = os.environ.get(name)
    if val is None:
        print(f"ERROR: Required environment variable '{name}' is not set.")
        sys.exit(1)
    return val

BUCKET                = require_env("BUCKET")
INPUT_PREFIX          = require_env("INPUT_PREFIX") 
POLYGON_GEOJSON_BLOB  = require_env("POLYGON_GEOJSON_BLOB")
DATA_ROOT             = require_env("DATA_ROOT")
METADATA_SUBREGIONS   = require_env("METADATA_SUBREGIONS")
WORK_DIR              = require_env("WORK_DIR")
MAX_WORKERS           = int(require_env("MAX_WORKERS"))

_test_limit = os.environ.get("TEST_LIMIT")
TEST_LIMIT = int(_test_limit) if _test_limit else None

TILE_SIZE       = 512
RGB_PREFIX      = f"{DATA_ROOT}PLANET-RGB/"
METADATA_PREFIX = f"{DATA_ROOT}metadata/"

print("Configuration:")
print(f"  BUCKET:              {BUCKET}")
print(f"  INPUT_PREFIX:        {INPUT_PREFIX}")
print(f"  POLYGON_GEOJSON_BLOB:{POLYGON_GEOJSON_BLOB}")
print(f"  RGB output:          gs://{BUCKET}/{RGB_PREFIX}")
print(f"  Metadata output:     gs://{BUCKET}/{METADATA_PREFIX}")
print(f"  MAX_WORKERS:         {MAX_WORKERS}")
print(f"  TEST_LIMIT:          {TEST_LIMIT or 'None (all tiles)'}")

# Auth — if running in Colab; call auth.authenticate_user() outside this script
# or keep it here if this script is the entrypoint.
auth.authenticate_user()

os.makedirs(f"{WORK_DIR}/input",  exist_ok=True)
os.makedirs(f"{WORK_DIR}/output", exist_ok=True)

client = storage.Client()
bucket = client.bucket(BUCKET)

# Load polygon + region data

print("\nDownloading polygon dataset from GCS...")

polygon_local = f"{WORK_DIR}/input/polygons.geojson"
regions_local = f"{WORK_DIR}/input/regions.geojson"

bucket.blob(POLYGON_GEOJSON_BLOB).download_to_filename(polygon_local)
gdf_polygons = gdf_polygons[gdf_polygons["TrainClass"] == "negative"].copy()
print(f"  Polygons: {len(gdf_polygons)} features (negatives)")

bucket.blob(METADATA_SUBREGIONS).download_to_filename(regions_local)
gdf_regions = gpd.read_file(regions_local)
if "ECO_NAME" not in gdf_regions.columns:
    print(f"ERROR: 'ECO_NAME' column not found in regions GeoJSON.")
    print(f"  Available columns: {list(gdf_regions.columns)}")
    sys.exit(1)
print(f"  Regions: {len(gdf_regions)} features")

# Pre-compute region centroids in WGS84 for fast nearest-region lookup
gdf_regions_wgs84 = gdf_regions.to_crs("EPSG:4326")
region_centroids   = gdf_regions_wgs84.geometry.centroid


def find_nearest_region(tile_centroid_wgs84: Point) -> str:
    """Return ECO_NAME of the region whose centroid is closest to the tile centroid."""
    distances   = region_centroids.distance(tile_centroid_wgs84)
    nearest_idx = distances.idxmin()
    return gdf_regions_wgs84.loc[nearest_idx, "ECO_NAME"]


print(f"\nListing source tiles at: gs://{BUCKET}/{INPUT_PREFIX}")

all_tile_blobs = [
    blob.name
    for blob in bucket.list_blobs(prefix=INPUT_PREFIX)
    if blob.name.endswith("_quad.tif")          # FIX: was ".tif", missed non-quad tifs and could match wrong files
][:TEST_LIMIT]

print(f"  Found: {len(all_tile_blobs)} tiles")

if not all_tile_blobs:
    print("No tiles found — check INPUT_PREFIX and BUCKET.")
    sys.exit(0)

print("Checking for already-processed tiles...")

metadata_blob_path = f"{METADATA_PREFIX}metadata.csv"
existing_blob      = bucket.blob(metadata_blob_path)

processed_source_paths: set[str] = set()
existing_df = None

if existing_blob.exists():
    existing_local = f"{WORK_DIR}/input/metadata_existing.csv"
    existing_blob.download_to_filename(existing_local)
    existing_df = pd.read_csv(existing_local)
    if "source_blob" in existing_df.columns:
        processed_source_paths = set(existing_df["source_blob"].dropna())
    else:
        # Legacy CSV without source_blob — fall back to filename matching
        processed_source_paths = {
            row for row in existing_df.get("Tile_ID", pd.Series(dtype=str))
        }

existing_uids = []
if existing_df is not None and "Tile_ID" in existing_df.columns:
    existing_uids = [
        int(v) for v in existing_df["Tile_ID"]
        if str(v).isdigit()
    ]
next_uid = max(existing_uids) + 1 if existing_uids else 1

tiles_to_process = [
    b for b in all_tile_blobs
    if b not in processed_source_paths
]

print(f"  Already processed: {len(all_tile_blobs) - len(tiles_to_process)}")
print(f"  Remaining:         {len(tiles_to_process)}")
print(f"  Next UID:          {next_uid:06d}")

if not tiles_to_process:
    print("All tiles already processed.")
    sys.exit(0)

# Worker init

def worker_init(polygon_path: str):
    global _gdf_polygons
    _gdf_polygons = gpd.read_file(polygon_path)
    _gdf_polygons = _gdf_polygons[_gdf_polygons["TrainClass"] == "negative"].copy()

def process_single_tile(blob_path: str, bucket_name: str, work_dir: str):
    parts     = blob_path.rstrip("/").split("/")
    # Use last 3 path parts: zoom/x/filename — gives a unique, compact name
    base_name = "_".join(parts[-3:]).replace(".tif", "")

    local_input   = f"{work_dir}/input/{base_name}.tif"
    local_rgb_out = f"{work_dir}/output/rgb_{base_name}.tif"

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    try:
        bucket.blob(blob_path).download_to_filename(local_input)

        with rasterio.open(local_input) as src:
            tile_w   = src.width
            tile_h   = src.height
            tile_crs = src.crs
            tile_tf  = src.transform

            # Crop to TILE_SIZE × TILE_SIZE from top-left
            win_w = min(TILE_SIZE, tile_w)
            win_h = min(TILE_SIZE, tile_h)
            win   = Window(0, 0, win_w, win_h)

            # Check polygon intersection
            gdf = _gdf_polygons
            if gdf.crs != tile_crs:
                gdf = gdf.to_crs(tile_crs)

            tile_bounds = rasterio.transform.array_bounds(
                win_h, win_w, rasterio.windows.transform(win, tile_tf)
            )
            tile_bbox = box(*tile_bounds)

            gdf_valid          = gdf.copy()
            gdf_valid["geometry"] = gdf_valid["geometry"].buffer(0)
            gdf_subset         = gdf_valid[gdf_valid.intersects(tile_bbox)]

            if gdf_subset.empty:
                return None   # no overlap → skip

            # Read RGB bands (bands 1, 2, 3 = R, G, B for Planet)
            try:
                rgb_data = src.read(indexes=[1, 2, 3], window=win)
            except Exception:
                return None

            # Skip fully-nodata chips
            if src.nodata is not None and (rgb_data == src.nodata).all():
                return None

            chip_tf = rasterio.windows.transform(win, tile_tf)

            # Compute tile centroid in WGS84
            tile_centroid_native = tile_bbox.centroid
            project = pyproj.Transformer.from_crs(
                tile_crs.to_epsg(), 4326, always_xy=True
            ).transform
            tile_centroid_wgs84 = shapely_transform(project, tile_centroid_native)
            centroid_lon = tile_centroid_wgs84.x
            centroid_lat = tile_centroid_wgs84.y

        # Write RGB chip
        profile = dict(
            driver    = "GTiff",
            count     = 3,
            dtype     = rgb_data.dtype,
            width     = win_w,
            height    = win_h,
            crs       = tile_crs,
            transform = chip_tf,
            compress  = "LZW",
        )

        with rasterio.open(local_rgb_out, "w", **profile) as dst:
            dst.write(rgb_data)
            dst.set_band_description(1, "Red")
            dst.set_band_description(2, "Green")
            dst.set_band_description(3, "Blue")

        return (local_rgb_out, centroid_lat, centroid_lon)

    finally:
        if os.path.exists(local_input):
            os.remove(local_input)


# Execution

print(f"\nProcessing {len(tiles_to_process)} tiles with {MAX_WORKERS} workers…\n")

success_count = 0
skip_count    = 0
error_count   = 0
metadata_rows = []

with concurrent.futures.ProcessPoolExecutor(
    max_workers  = MAX_WORKERS,
    initializer  = worker_init,
    initargs     = (polygon_local,),
) as executor:

    future_to_blob = {
        executor.submit(process_single_tile, blob, BUCKET, WORK_DIR): blob
        for blob in tiles_to_process
    }

    with tqdm(total=len(tiles_to_process), desc="processing tiles", unit="tile") as pbar:
        for future in concurrent.futures.as_completed(future_to_blob):
            blob_path = future_to_blob[future]
            try:
                result = future.result()
                if result is not None:
                    local_rgb, centroid_lat, centroid_lon = result

                    uid_str = f"{next_uid:06d}"

                    # Upload RGB chip
                    bucket.blob(f"{RGB_PREFIX}{uid_str}.tif").upload_from_filename(local_rgb)
                    os.remove(local_rgb)

                    # Nearest-region lookup
                    tile_point  = Point(centroid_lon, centroid_lat)
                    region_name = find_nearest_region(tile_point)

                    # FIX 2 cont: store source_blob so resume is exact on re-run
                    metadata_rows.append({
                        "Tile_ID":      uid_str,
                        "centroid_lat": round(centroid_lat, 6),
                        "centroid_lon": round(centroid_lon, 6),
                        "TrainClass":   "positive",
                        "RegionName":   region_name,
                        "UIDs":         9999,
                        "source_blob":  blob_path,          # NEW column
                    })

                    next_uid      += 1
                    success_count += 1
                    tqdm.write(f"✓ {uid_str}  {blob_path.split('/')[-1]}")
                else:
                    skip_count += 1

            except Exception as exc:
                error_count += 1
                tqdm.write(f"✗ ERROR: {blob_path.split('/')[-1]} — {exc}")

            pbar.update(1)

# Append metadata to CSV

if metadata_rows:
    metadata_df = pd.DataFrame(
        metadata_rows,
        columns=["Tile_ID", "centroid_lat", "centroid_lon",
                 "TrainClass", "RegionName", "UIDs", "source_blob"],
    )
    local_csv = f"{WORK_DIR}/output/metadata.csv"

    if existing_df is not None:
        # Align columns — existing CSV may lack source_blob
        metadata_df = pd.concat([existing_df, metadata_df], ignore_index=True)
        print(f"\nAppended {len(metadata_rows)} rows → {len(metadata_df)} total in CSV")
    else:
        print(f"\nCreating new metadata CSV with {len(metadata_df)} rows")

    metadata_df.to_csv(local_csv, index=False)
    bucket.blob(metadata_blob_path).upload_from_filename(local_csv)
    os.remove(local_csv)
    print(f"Metadata uploaded → gs://{BUCKET}/{metadata_blob_path}")
else:
    print("\nNo successful tiles — metadata CSV not written.")

# Final SUmmary

print(f"\n{'─'*50}")
print(f"Complete:  {success_count} written | {skip_count} skipped (no overlap) | {error_count} errors")
print(f"RGB chips: gs://{BUCKET}/{RGB_PREFIX}")
print(f"Metadata:  gs://{BUCKET}/{METADATA_PREFIX}")
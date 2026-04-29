# PLANET IS BGR not RGB — read bands respectively

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

# TARGET_TILES replaces TEST_LIMIT — this is the total number of output tiles wanted
_target = os.environ.get("TARGET_TILES")
TARGET_TILES = int(_target) if _target else None

TILE_SIZE       = 512
RGB_PREFIX      = f"{DATA_ROOT}/PLANET-RGB/"
METADATA_PREFIX = f"{DATA_ROOT}/metadata/"

# CRS used for all spatial operations — equal-area so distances/intersections are meaningful
WORKING_CRS = "EPSG:6933"

print("Configuration:")
print(f"  BUCKET:              {BUCKET}")
print(f"  INPUT_PREFIX:        {INPUT_PREFIX}")
print(f"  POLYGON_GEOJSON_BLOB:{POLYGON_GEOJSON_BLOB}")
print(f"  RGB output:          gs://{BUCKET}/{RGB_PREFIX}")
print(f"  Metadata output:     gs://{BUCKET}/{METADATA_PREFIX}")
print(f"  MAX_WORKERS:         {MAX_WORKERS}")
print(f"  TARGET_TILES:        {TARGET_TILES or 'None (all negative tiles)'}")


os.makedirs(f"{WORK_DIR}/input",  exist_ok=True)
os.makedirs(f"{WORK_DIR}/output", exist_ok=True)

client = storage.Client()
bucket = client.bucket(BUCKET)


print("\nDownloading polygon dataset from GCS...")

polygon_local = f"{WORK_DIR}/input/polygons.geojson"
regions_local = f"{WORK_DIR}/input/regions.geojson"

bucket.blob(POLYGON_GEOJSON_BLOB).download_to_filename(polygon_local)
gdf_polygons = gpd.read_file(polygon_local)
print(f"  Polygons loaded:     {len(gdf_polygons)} total features")

# Filter to negative class only
gdf_polygons = gdf_polygons[gdf_polygons["TrainClass"] == "Negative"].copy()
print(f"  Negative class:      {len(gdf_polygons)} features")

# Reproject to working CRS once here — workers inherit the saved file
gdf_polygons = gdf_polygons.to_crs(WORKING_CRS)



bucket.blob(METADATA_SUBREGIONS).download_to_filename(regions_local)
gdf_regions = gpd.read_file(regions_local)

if "ECO_NAME" not in gdf_regions.columns:
    print(f"ERROR: 'ECO_NAME' column not found in regions GeoJSON.")
    print(f"  Available columns: {list(gdf_regions.columns)}")
    sys.exit(1)

print(f"  Regions loaded:      {len(gdf_regions)} ecoregions")

# Reproject regions to working CRS
gdf_regions_wgs84 = gdf_regions.to_crs("EPSG:4326")
gdf_regions_work  = gdf_regions.to_crs(WORKING_CRS)

print("\nAssigning ecoregions to polygons via spatial join...")

# Use centroid of each polygon for the join — faster than full geometry join
polygon_centroids = gdf_polygons.copy()
polygon_centroids["geometry"] = gdf_polygons.geometry.centroid

gdf_polygons_with_region = gpd.sjoin(
    polygon_centroids[["geometry"]],
    gdf_regions_work[["geometry", "ECO_NAME"]],
    how       = "left",
    predicate = "within",
)

# Attach ECO_NAME back to original polygons
gdf_polygons["ECO_NAME"] = gdf_polygons_with_region["ECO_NAME"].values

# Polygons whose centroid didn't fall within any region — assign nearest region
missing_mask = gdf_polygons["ECO_NAME"].isna()
if missing_mask.any():
    print(f"  {missing_mask.sum()} polygons outside all regions — assigning nearest ecoregion...")
    region_centroids_work = gdf_regions_work.geometry.centroid
    region_tree = STRtree(region_centroids_work.values)

    for idx in gdf_polygons[missing_mask].index:
        poly_centroid = gdf_polygons.loc[idx, "geometry"].centroid
        nearest_i     = region_tree.nearest(poly_centroid)
        gdf_polygons.at[idx, "ECO_NAME"] = gdf_regions_work.iloc[nearest_i]["ECO_NAME"]


eco_groups   = gdf_polygons.groupby("ECO_NAME")
n_ecoregions = len(eco_groups)

if TARGET_TILES is not None:
    per_region = max(1, TARGET_TILES // n_ecoregions)
    print(f"\nStratified sampling: {TARGET_TILES} target tiles / "
          f"{n_ecoregions} ecoregions = {per_region} per region")
else:
    per_region = None
    print(f"\nNo TARGET_TILES set — using all {len(gdf_polygons)} negative polygons")

sampled_parts = []
for eco_name, group in eco_groups:
    if per_region is None or len(group) <= per_region:
        sampled_parts.append(group)
    else:
        sampled_parts.append(group.sample(n=per_region, random_state=42))

gdf_sampled = pd.concat(sampled_parts).reset_index(drop=True)
print(f"  Sampled polygons:    {len(gdf_sampled)} "
      f"(across {n_ecoregions} ecoregions)")

# Save sampled polygon file — this is what workers will load
sampled_local = f"{WORK_DIR}/input/polygons_sampled.geojson"
gdf_sampled.to_file(sampled_local, driver="GeoJSON")


region_centroids_wgs84 = gdf_regions_wgs84.geometry.centroid
_region_tree           = STRtree(region_centroids_wgs84.values)

def find_nearest_region(tile_centroid_wgs84: Point) -> str:
    """Return ECO_NAME of the region whose centroid is closest to the tile centroid."""
    nearest_i = _region_tree.nearest(tile_centroid_wgs84)
    return gdf_regions_wgs84.iloc[nearest_i]["ECO_NAME"]

print(f"\nListing source tiles at: gs://{BUCKET}/{INPUT_PREFIX}")

all_tile_blobs = [
    blob.name
    for blob in bucket.list_blobs(prefix=INPUT_PREFIX)
    if blob.name.endswith("_quad.tif")
]

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

existing_uids = []
if existing_df is not None and "Tile_ID" in existing_df.columns:
    existing_uids = [int(v) for v in existing_df["Tile_ID"] if str(v).isdigit()]
next_uid = max(existing_uids) + 1 if existing_uids else 1

tiles_to_process = [b for b in all_tile_blobs if b not in processed_source_paths]

print(f"  Already processed:   {len(all_tile_blobs) - len(tiles_to_process)}")
print(f"  Remaining:           {len(tiles_to_process)}")
print(f"  Next UID:            {next_uid:06d}")

if not tiles_to_process:
    print("All tiles already processed.")
    sys.exit(0)

def worker_init(sampled_polygon_path: str):
    """
    Load the pre-filtered, pre-reprojected, sampled polygon file.
    Build a spatial index for fast intersection checks.
    """
    global _gdf_polygons, _polygon_sindex

    _gdf_polygons = gpd.read_file(sampled_polygon_path)
    # File is already in WORKING_CRS — no reproject needed here

    # Build spatial index once per worker
    _polygon_sindex = _gdf_polygons.sindex


def process_single_tile(blob_path: str, bucket_name: str, work_dir: str):
    parts     = blob_path.rstrip("/").split("/")
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

            win_w = min(TILE_SIZE, tile_w)
            win_h = min(TILE_SIZE, tile_h)
            win   = Window(0, 0, win_w, win_h)

            # Reproject tile bbox to WORKING_CRS for intersection check
            tile_bounds_native = rasterio.transform.array_bounds(
                win_h, win_w, rasterio.windows.transform(win, tile_tf)
            )
            tile_bbox_native = box(*tile_bounds_native)

            project_to_work = pyproj.Transformer.from_crs(
                tile_crs.to_epsg(), WORKING_CRS, always_xy=True
            ).transform
            tile_bbox_work = shapely_transform(project_to_work, tile_bbox_native)

            # Fast intersection using spatial index
            candidate_idx = list(_polygon_sindex.intersection(tile_bbox_work.bounds))
            if not candidate_idx:
                return None

            candidates = _gdf_polygons.iloc[candidate_idx]
            candidates = candidates.copy()
            candidates["geometry"] = candidates["geometry"].buffer(0)
            matches = candidates[candidates.intersects(tile_bbox_work)]

            if matches.empty:
                return None

            # Read BGR bands (Planet band order: 1=B, 2=G, 3=R)
            try:
                rgb_data = src.read(indexes=[1, 2, 3], window=win)
            except Exception:
                return None

            if src.nodata is not None and (rgb_data == src.nodata).all():
                return None

            chip_tf = rasterio.windows.transform(win, tile_tf)

            # Tile centroid → WGS84 for metadata
            tile_centroid_native = tile_bbox_native.centroid
            project_to_wgs84 = pyproj.Transformer.from_crs(
                tile_crs.to_epsg(), 4326, always_xy=True
            ).transform
            tile_centroid_wgs84 = shapely_transform(project_to_wgs84, tile_centroid_native)
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
            dst.set_band_description(1, "Blue")   # Band 1 is Blue in Planet BGR
            dst.set_band_description(2, "Green")
            dst.set_band_description(3, "Red")

        return (local_rgb_out, centroid_lat, centroid_lon)

    finally:
        if os.path.exists(local_input):
            os.remove(local_input)


print(f"\nProcessing {len(tiles_to_process)} tiles with {MAX_WORKERS} workers…\n")

success_count = 0
skip_count    = 0
error_count   = 0
metadata_rows = []

with concurrent.futures.ProcessPoolExecutor(
    max_workers  = MAX_WORKERS,
    initializer  = worker_init,
    initargs     = (sampled_local,),
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

                    bucket.blob(f"{RGB_PREFIX}{uid_str}.tif").upload_from_filename(local_rgb)
                    os.remove(local_rgb)

                    tile_point  = Point(centroid_lon, centroid_lat)
                    region_name = find_nearest_region(tile_point)

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
                tqdm.write(f"✗ ERROR: {blob_path.split('/')[-1]} — {exc}")

            pbar.update(1)

if metadata_rows:
    metadata_df = pd.DataFrame(
        metadata_rows,
        columns=["Tile_ID", "centroid_lat", "centroid_lon",
                 "TrainClass", "RegionName", "UIDs", "source_blob"],
    )
    local_csv = f"{WORK_DIR}/output/metadata.csv"

    if existing_df is not None:
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

print(f"\n{'─'*50}")
print(f"Complete:  {success_count} written | {skip_count} skipped (no overlap) | {error_count} errors")
print(f"RGB chips: gs://{BUCKET}/{RGB_PREFIX}")
print(f"Metadata:  gs://{BUCKET}/{METADATA_PREFIX}")
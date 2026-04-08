import os
import sys
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import box
from google.colab import auth
from google.cloud import storage
import concurrent.futures
from tqdm import tqdm

# CONFIG SETUP FOR ASSOCIATED SHELL SCRIPT (positive_tile_run.sh)

def require_env(name):
    val = os.environ.get(name)
    if val is None:
        print(f"ERROR: Required environment variable '{name}' is not set.")
        print("       Run this script via run_reprocess.sh, not directly.")
        sys.exit(1)
    return val

BUCKET                 = require_env("BUCKET")
INPUT_PREFIX           = require_env("INPUT_PREFIX")
OUTPUT_PREFIX          = require_env("OUTPUT_PREFIX")
POSITIVE_GEOJSON_BLOB  = require_env("POSITIVE_GEOJSON")
IGNORE_GEOJSON_BLOB    = require_env("IGNORE_GEOJSON")
WORK_DIR               = require_env("WORK_DIR")
MAX_WORKERS            = int(require_env("MAX_WORKERS"))

_test_limit = os.environ.get("TEST_LIMIT")
TEST_LIMIT = int(_test_limit) if _test_limit else None

print("config loaded from shell script:")
print(f"  BUCKET:                 {BUCKET}")
print(f"  INPUT_PREFIX:           {INPUT_PREFIX}")
print(f"  OUTPUT_PREFIX:          {OUTPUT_PREFIX}")
print(f"  POSITIVE_GEOJSON_BLOB:  {POSITIVE_GEOJSON_BLOB}")
print(f"  IGNORE_GEOJSON_BLOB:    {IGNORE_GEOJSON_BLOB}")
print(f"  WORK_DIR:               {WORK_DIR}")
print(f"  MAX_WORKERS:            {MAX_WORKERS}")
print(f"  TEST_LIMIT:             {TEST_LIMIT or 'None (all tiles)'}")

# AUTH + LOCAL DIRS
auth.authenticate_user()

os.makedirs(f"{WORK_DIR}/input",  exist_ok=True)
os.makedirs(f"{WORK_DIR}/output", exist_ok=True)

client = storage.Client()
bucket = client.bucket(BUCKET)

# LOAD POLYGON DATASETS
print("\nDownloading polygon datasets from GCS...")

positive_local  = f"{WORK_DIR}/input/positive.geojson"
ignore_local = f"{WORK_DIR}/input/ignore.geojson"

bucket.blob(POSITIVE_GEOJSON_BLOB).download_to_filename(positive_local)
gdf_positive = gpd.read_file(positive_local)
print(f"  Positive polygons:  {len(gdf_positive)} features")

bucket.blob(IGNORE_GEOJSON_BLOB).download_to_filename(ignore_local)
gdf_ignore = gpd.read_file(ignore_local)
print(f"  Ignore polygons: {len(gdf_ignore)} features")


# LIST TILES
print(f"\nfinding imagery tiles: {INPUT_PREFIX}")

all_tile_blobs = [
    blob.name
    for blob in bucket.list_blobs(prefix=INPUT_PREFIX)
    if blob.name.endswith(".tif")
][:TEST_LIMIT]

print(f"  found {len(all_tile_blobs)} tiles")

if len(all_tile_blobs) == 0:
    print("no tiles found — check INPUT_PREFIX and BUCKET in run_reprocess.sh")
    sys.exit(0)


# RESUME SUPPORT
print("checking for already-processed tiles (resumes if process was interrupted)")

def _derive_output_name(blob_path):
    base = blob_path.split("/")[-1].replace(".tif", "")
    return f"{base}_rgbm.tif"

already_done = {
    blob.name.split("/")[-1]
    for blob in bucket.list_blobs(prefix=OUTPUT_PREFIX)
    if blob.name.endswith("_rgbm.tif")
}

tiles_to_process = [
    blob_path for blob_path in all_tile_blobs
    if _derive_output_name(blob_path) not in already_done
]

print(f"  already processed: {len(all_tile_blobs) - len(tiles_to_process)}")
print(f"  remaining:         {len(tiles_to_process)}")

if len(tiles_to_process) == 0:
    print("All tiles already processed")
    sys.exit(0)


# CORE PROCESSING FUNCTION
def worker_init(positive_path, ignore_path):
    global gdf_positive, gdf_ignore
    gdf_positive = gpd.read_file(positive_path)
    gdf_ignore   = gpd.read_file(ignore_path)

def process_single_tile(blob_path, bucket_name, work_dir, output_prefix):
    """
    For one tile:
      1. Download from GCS
      2. Read bands 1-3 as RGB
      3. Rasterize a new mask (255=ignore, 1=positive, 0=background)
      4. Write a single 4-band tiff: bands 1-3 RGB, band 4 mask
      5. Upload to GCS, delete local files
    """
    # uses gdf_positive and gdf_ignore from worker_init
    tile_filename = blob_path.split("/")[-1]
    base_name     = tile_filename.replace(".tif", "")
    parts         = blob_path.split("/")
    col, row      = parts[-3], parts[-2]

    local_input  = f"{work_dir}/input/{col}_{row}_{tile_filename}"
    local_output = f"{work_dir}/output/{base_name}_rgbm.tif"

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    try:
        bucket.blob(blob_path).download_to_filename(local_input)

        with rasterio.open(local_input) as src:
            tile_width     = src.width
            tile_height    = src.height
            tile_crs       = src.crs
            tile_transform = src.transform
            tile_nodata    = src.nodata

            rgb_data = src.read(indexes=[1, 2, 3])

            tile_bounds = rasterio.transform.array_bounds(tile_height, tile_width, tile_transform)
            tile_bbox   = box(*tile_bounds)

            def subset_to_tile(gdf):
                if gdf.crs != tile_crs:
                    gdf = gdf.to_crs(tile_crs)
                gdf = gdf.copy()
                gdf["geometry"] = gdf["geometry"].buffer(0)
                return gdf[gdf.intersects(tile_bbox)]

            pos_subset    = subset_to_tile(gdf_positive)
            ignore_subset = subset_to_tile(gdf_ignore)

            raster_kwargs = dict(
                out_shape=(tile_height, tile_width),
                transform=tile_transform,
                fill=0,
                dtype=np.uint8
            )

            new_mask = np.zeros((tile_height, tile_width), dtype=np.uint8)

            if len(ignore_subset) > 0:
                ignore_shapes = [(geom, 255) for geom in ignore_subset.geometry]
                new_mask      = rasterize(ignore_shapes, **raster_kwargs)

            if len(pos_subset) > 0:
                pos_shapes = [(geom, 1) for geom in pos_subset.geometry]
                pos_raster = rasterize(pos_shapes, **raster_kwargs)
                new_mask[pos_raster == 1] = 1

            if new_mask.max() == 0:
                return None

        profile = {
            "driver":    "GTiff",
            "dtype":     rgb_data.dtype,
            "width":     tile_width,
            "height":    tile_height,
            "count":     4,
            "crs":       tile_crs,
            "transform": tile_transform,
            "compress":  "LZW",
        }
        if tile_nodata is not None:
            profile["nodata"] = tile_nodata

        mask_band = new_mask[np.newaxis, :, :].astype(rgb_data.dtype)
        rgbm_data = np.concatenate([rgb_data, mask_band], axis=0)  # shape: (4, H, W)

        with rasterio.open(local_output, "w", **profile) as dst:
            dst.write(rgbm_data)  # writes all 4 bands at once
            dst.set_band_description(1, "Red")
            dst.set_band_description(2, "Green")
            dst.set_band_description(3, "Blue")
            dst.set_band_description(4, "Mask: 0=background 1=positive 255=ignore")

        gcs_path = f"{output_prefix}{col}/{row}/{base_name}_rgbm.tif"
        bucket.blob(gcs_path).upload_from_filename(local_output)
        os.remove(local_output)

        return f"{col}/{row}/{tile_filename}"

    finally:
        if os.path.exists(local_input):
            os.remove(local_input)


# RUN IN PARALLEL
print(f"\nprocessing {len(tiles_to_process)} tiles with {MAX_WORKERS} workers\n")

success_count = 0
skip_count    = 0
error_count   = 0

with concurrent.futures.ProcessPoolExecutor(
    max_workers=MAX_WORKERS,
    initializer=worker_init,
    initargs=(positive_local, ignore_local)
) as executor:

    tile_args = [
        (blob_path, BUCKET, WORK_DIR, OUTPUT_PREFIX)
        for blob_path in tiles_to_process
    ]

    future_to_blob = {
        executor.submit(process_single_tile, *args): args[0]
        for args in tile_args
    }

    with tqdm(total=len(tiles_to_process), desc="reprocessing tiles", unit="tile") as pbar:
        for future in concurrent.futures.as_completed(future_to_blob):
            blob_path = future_to_blob[future]
            try:
                result = future.result()
                if result is not None:
                    success_count += 1
                    # tqdm.write(f"{result}") # optionally print successful tiles for error check
                else:
                    skip_count += 1
            except Exception as exc:
                error_count += 1
                tqdm.write(f"ERROR: {blob_path.split('/')[-1]} — {exc}")
            pbar.update(1)

# SUMMARY
print(f"\ncomplete {success_count} written, {skip_count} skipped, {error_count} errors")
print(f"output folder = gs://{BUCKET}/{OUTPUT_PREFIX}")
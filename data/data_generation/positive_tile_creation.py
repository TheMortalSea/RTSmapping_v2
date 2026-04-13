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

def require_env(name):
    val = os.environ.get(name)
    if val is None:
        print(f"ERROR: Required environment variable '{name}' is not set.")
        sys.exit(1)
    return val

BUCKET                = require_env("BUCKET")
INPUT_PREFIX          = require_env("INPUT_PREFIX")
DATA_ROOT             = require_env("DATA_ROOT")
POSITIVE_GEOJSON_BLOB = require_env("POSITIVE_GEOJSON")
IGNORE_GEOJSON_BLOB   = require_env("IGNORE_GEOJSON")
WORK_DIR              = require_env("WORK_DIR")
MAX_WORKERS           = int(require_env("MAX_WORKERS"))

_test_limit = os.environ.get("TEST_LIMIT")
TEST_LIMIT = int(_test_limit) if _test_limit else None

RGB_PREFIX    = f"{DATA_ROOT}PLANET-RGB/"
LABELS_PREFIX = f"{DATA_ROOT}labels/"

print("config loaded:")
print(f"  BUCKET:        {BUCKET}")
print(f"  INPUT_PREFIX:  {INPUT_PREFIX}")
print(f"  DATA_ROOT:     {DATA_ROOT}")
print(f"  RGB_PREFIX:    {RGB_PREFIX}")
print(f"  LABELS_PREFIX: {LABELS_PREFIX}")
print(f"  MAX_WORKERS:   {MAX_WORKERS}")
print(f"  TEST_LIMIT:    {TEST_LIMIT or 'None (all tiles)'}")

auth.authenticate_user()

os.makedirs(f"{WORK_DIR}/input",  exist_ok=True)
os.makedirs(f"{WORK_DIR}/output", exist_ok=True)

client = storage.Client()
bucket = client.bucket(BUCKET)

print("\nDownloading polygon datasets from GCS...")

positive_local = f"{WORK_DIR}/input/positive.geojson"
ignore_local   = f"{WORK_DIR}/input/ignore.geojson"

bucket.blob(POSITIVE_GEOJSON_BLOB).download_to_filename(positive_local)
gdf_positive = gpd.read_file(positive_local)
print(f"  Positive polygons: {len(gdf_positive)} features")

bucket.blob(IGNORE_GEOJSON_BLOB).download_to_filename(ignore_local)
gdf_ignore = gpd.read_file(ignore_local)
print(f"  Ignore polygons:   {len(gdf_ignore)} features")

print(f"\nFinding imagery tiles: {INPUT_PREFIX}")

all_tile_blobs = [
    blob.name
    for blob in bucket.list_blobs(prefix=INPUT_PREFIX)
    if blob.name.endswith(".tif")
][:TEST_LIMIT]

print(f"  Found {len(all_tile_blobs)} tiles")

if len(all_tile_blobs) == 0:
    print("No tiles found — check INPUT_PREFIX and BUCKET")
    sys.exit(0)

print("Checking for already-processed tiles...")

already_done_names = {
    blob.name.split("/")[-1]
    for blob in bucket.list_blobs(prefix=LABELS_PREFIX)
    if blob.name.endswith(".tif")
}

existing_uids = [
    int(name.replace(".tif", ""))
    for name in already_done_names
    if name.replace(".tif", "").isdigit()
]

next_uid = max(existing_uids) + 1 if existing_uids else 1

tiles_to_process = [b for b in all_tile_blobs if b.split("/")[-1] not in already_done_names]

print(f"  Already processed: {len(all_tile_blobs) - len(tiles_to_process)}")
print(f"  Remaining:         {len(tiles_to_process)}")
print(f"  Next UID:          {next_uid:06d}")

if len(tiles_to_process) == 0:
    print("All tiles already processed")
    sys.exit(0)

tile_uid_map = {
    blob_path: next_uid + i
    for i, blob_path in enumerate(tiles_to_process)
}


def worker_init(positive_path, ignore_path):
    global gdf_positive, gdf_ignore
    gdf_positive = gpd.read_file(positive_path)
    gdf_ignore   = gpd.read_file(ignore_path)


def process_single_tile(blob_path, uid, bucket_name, work_dir, rgb_prefix, labels_prefix):
    tile_filename = blob_path.split("/")[-1]
    base_name     = tile_filename.replace(".tif", "")
    parts         = blob_path.split("/")
    col, row      = parts[-3], parts[-2]
    uid_str       = f"{uid:06d}"

    local_input      = f"{work_dir}/input/{col}_{row}_{tile_filename}"
    local_rgb_out    = f"{work_dir}/output/{uid_str}_rgb.tif"
    local_label_out  = f"{work_dir}/output/{uid_str}_label.tif"

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

        base_profile = dict(
            driver="GTiff",
            width=tile_width,
            height=tile_height,
            crs=tile_crs,
            transform=tile_transform,
            compress="LZW",
        )
        if tile_nodata is not None:
            base_profile["nodata"] = tile_nodata

        rgb_profile = {**base_profile, "dtype": rgb_data.dtype, "count": 3}
        with rasterio.open(local_rgb_out, "w", **rgb_profile) as dst:
            dst.write(rgb_data)
            dst.set_band_description(1, "Red")
            dst.set_band_description(2, "Green")
            dst.set_band_description(3, "Blue")

        label_profile = {**base_profile, "dtype": np.uint8, "count": 1}
        with rasterio.open(local_label_out, "w", **label_profile) as dst:
            dst.write(new_mask[np.newaxis, :, :])
            dst.set_band_description(1, "Mask: 0=background 1=positive 255=ignore")

        bucket.blob(f"{rgb_prefix}{uid_str}.tif").upload_from_filename(local_rgb_out)
        bucket.blob(f"{labels_prefix}{uid_str}.tif").upload_from_filename(local_label_out)

        os.remove(local_rgb_out)
        os.remove(local_label_out)

        return f"{uid_str} — {col}/{row}/{tile_filename}"

    finally:
        if os.path.exists(local_input):
            os.remove(local_input)


print(f"\nProcessing {len(tiles_to_process)} tiles with {MAX_WORKERS} workers\n")

success_count = 0
skip_count    = 0
error_count   = 0

with concurrent.futures.ProcessPoolExecutor(
    max_workers=MAX_WORKERS,
    initializer=worker_init,
    initargs=(positive_local, ignore_local)
) as executor:

    tile_args = [
        (blob_path, tile_uid_map[blob_path], BUCKET, WORK_DIR, RGB_PREFIX, LABELS_PREFIX)
        for blob_path in tiles_to_process
    ]

    future_to_blob = {
        executor.submit(process_single_tile, *args): args[0]
        for args in tile_args
    }

    with tqdm(total=len(tiles_to_process), desc="processing tiles", unit="tile") as pbar:
        for future in concurrent.futures.as_completed(future_to_blob):
            blob_path = future_to_blob[future]
            try:
                result = future.result()
                if result is not None:
                    success_count += 1
                else:
                    skip_count += 1
            except Exception as exc:
                error_count += 1
                tqdm.write(f"ERROR: {blob_path.split('/')[-1]} — {exc}")
            pbar.update(1)

print(f"\nComplete — {success_count} written, {skip_count} skipped, {error_count} errors")
print(f"RGB:    gs://{BUCKET}/{RGB_PREFIX}")
print(f"Labels: gs://{BUCKET}/{LABELS_PREFIX}")
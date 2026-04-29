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
import pandas as pd

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
METADATA_SUBREGIONS   = require_env("METADATA_SUBREGIONS")
WORK_DIR              = require_env("WORK_DIR")
MAX_WORKERS           = int(require_env("MAX_WORKERS"))

_test_limit = os.environ.get("TEST_LIMIT")
TEST_LIMIT = int(_test_limit) if _test_limit else None

RGB_PREFIX     = f"{DATA_ROOT}PLANET-RGB/"
LABELS_PREFIX  = f"{DATA_ROOT}labels/"
METADATA_PREFIX = f"{DATA_ROOT}"

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
regions_local  = f"{WORK_DIR}/input/regions.geojson"

bucket.blob(POSITIVE_GEOJSON_BLOB).download_to_filename(positive_local)
gdf_positive = gpd.read_file(positive_local)
print(f"  Positive polygons: {len(gdf_positive)} features")

bucket.blob(IGNORE_GEOJSON_BLOB).download_to_filename(ignore_local)
gdf_ignore = gpd.read_file(ignore_local)
print(f"  Ignore polygons:   {len(gdf_ignore)} features")

# Download the regions GeoJSON (used for RegionName lookup)
bucket.blob(METADATA_SUBREGIONS).download_to_filename(regions_local)
gdf_regions = gpd.read_file(regions_local)
if "ECO_NAME" not in gdf_regions.columns:
    print("ERROR: 'ECO_NAME' column not found in regions GeoJSON.")
    print(f"  Available columns: {list(gdf_regions.columns)}")
    sys.exit(1)
print(f"  Regions: {len(gdf_regions)} features")

# Pre-compute region centroids in WGS84 for nearest-centroid matching.
# We reproject to a geographic CRS so centroid distance is in degrees,
# which is sufficient for coarse nearest-region lookup.
gdf_regions_wgs84 = gdf_regions.to_crs("EPSG:4326")
region_centroids  = gdf_regions_wgs84.geometry.centroid  # GeoSeries of Points


def find_nearest_region(tile_centroid_wgs84):
    """Return ECO_NAME of the region whose centroid is closest to the tile centroid."""
    distances = region_centroids.distance(tile_centroid_wgs84)
    nearest_idx = distances.idxmin()
    return gdf_regions_wgs84.loc[nearest_idx, "ECO_NAME"]


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


def worker_init(positive_path, ignore_path):
    global gdf_positive, gdf_ignore
    gdf_positive = gpd.read_file(positive_path)
    gdf_ignore   = gpd.read_file(ignore_path)


def process_single_tile(blob_path, bucket_name, work_dir):
    tile_filename = blob_path.split("/")[-1]
    base_name     = blob_path.replace("/", "_").replace(".tif", "")
    parts         = blob_path.split("/")
    col, row      = parts[-3], parts[-2]

    local_input     = f"{work_dir}/input/{base_name}.tif"
    local_rgb_out   = f"{work_dir}/output/rgb_{base_name}.tif"
    local_label_out = f"{work_dir}/output/label_{base_name}.tif"

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
            bgr = src.read(indexes=[1, 2, 3])
            rgb_data = bgr[[2, 1, 0], :, :]

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

            # --- Compute tile centroid in WGS84 for metadata ---
            from shapely.geometry import Point
            import pyproj
            from shapely.ops import transform as shapely_transform

            tile_centroid_native = tile_bbox.centroid

            # Reproject centroid to WGS84
            project = pyproj.Transformer.from_crs(
                tile_crs.to_epsg(),
                4326,
                always_xy=True
            ).transform
            tile_centroid_wgs84 = shapely_transform(project, tile_centroid_native)
            centroid_lon = tile_centroid_wgs84.x
            centroid_lat = tile_centroid_wgs84.y

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

        return (local_rgb_out, local_label_out, centroid_lat, centroid_lon)

    finally:
        if os.path.exists(local_input):
            os.remove(local_input)


print(f"\nProcessing {len(tiles_to_process)} tiles with {MAX_WORKERS} workers\n")

success_count  = 0
skip_count     = 0
error_count    = 0
metadata_rows  = []

with concurrent.futures.ProcessPoolExecutor(
    max_workers=MAX_WORKERS,
    initializer=worker_init,
    initargs=(positive_local, ignore_local)
) as executor:

    tile_args = [
        (blob_path, BUCKET, WORK_DIR)
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
                    local_rgb, local_label, centroid_lat, centroid_lon = result

                    uid_str = f"{next_uid:06d}"

                    bucket.blob(f"{RGB_PREFIX}{uid_str}.tif").upload_from_filename(local_rgb)
                    bucket.blob(f"{LABELS_PREFIX}{uid_str}.tif").upload_from_filename(local_label)
                    os.remove(local_rgb)
                    os.remove(local_label)

                    # Nearest-region lookup using pre-computed WGS84 region centroids
                    from shapely.geometry import Point
                    tile_point  = Point(centroid_lon, centroid_lat)
                    region_name = find_nearest_region(tile_point)

                    metadata_rows.append({
                        "Tile_ID":      uid_str,
                        "centroid_lat": round(centroid_lat, 6),
                        "centroid_lon": round(centroid_lon, 6),
                        "TrainClass":   "positive",
                        "RegionName":   region_name,
                        "UIDs":         9999,
                    })

                    next_uid += 1
                    success_count += 1
                else:
                    skip_count += 1
            except Exception as exc:
                error_count += 1
                tqdm.write(f"ERROR: {blob_path.split('/')[-1]} — {exc}")
            pbar.update(1)

# ── Write metadata CSV ────────────────────────────────────────────────────────

if metadata_rows:
    metadata_df = pd.DataFrame(
        metadata_rows,
        columns=["Tile_ID", "centroid_lat", "centroid_lon", "TrainClass", "RegionName", "UIDs"]
    )

    local_csv = f"{WORK_DIR}/output/metadata.csv"

    
    metadata_blob_path = f"{METADATA_PREFIX}metadata.csv"
    existing_blob = bucket.blob(metadata_blob_path)
    existing_df = None
    if existing_blob.exists():
        existing_local = f"{WORK_DIR}/input/metadata_existing.csv"
        existing_blob.download_to_filename(existing_local)
        existing_df = pd.read_csv(existing_local, dtype={"Tile_ID": str})

    existing_uids = []
    if existing_df is not None and "Tile_ID" in existing_df.columns:
        existing_uids = [int(v) for v in existing_df["Tile_ID"] if str(v).isdigit()]
    next_uid = max(existing_uids) + 1 if existing_uids else 1

    metadata_df.to_csv(local_csv, index=False)
    bucket.blob(metadata_blob_path).upload_from_filename(local_csv)
    os.remove(local_csv)
    print(f"Metadata CSV uploaded → gs://{BUCKET}/{metadata_blob_path}")
else:
    print("\nNo successful tiles — metadata CSV not written")

print(f"\nComplete — {success_count} written, {skip_count} skipped, {error_count} errors")
print(f"RGB:      gs://{BUCKET}/{RGB_PREFIX}")
print(f"Labels:   gs://{BUCKET}/{LABELS_PREFIX}")
print(f"Metadata: gs://{BUCKET}/{METADATA_PREFIX}")
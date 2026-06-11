"""
Positive Training Tile Creation Script

Outputs: - RGB tiles in gs://{BUCKET}/{RGB_PREFIX} as 3-band GeoTIFFs
         - Label tiles in gs://{BUCKET}/{LABELS_PREFIX} as single-band GeoTIFFs
         - metadata.csv in gs://{BUCKET}/{METADATA_PREFIX} with columns: Tile_ID, centroid_lat, centroid_lon, TrainClass, RegionName, UIDs, Version

More info on metadata columns and downstream use can be found in the data.md documentation file.

Imagery source: Planet basemap quad tiles. The blob path for each tile is constructed from
INPUT_PREFIX and MOSAIC_NAME env vars plus the col/row parsed from the tile boundary's Name
column (e.g. tile_91_1573_c1_r6 -> {INPUT_PREFIX}/91/1573/{MOSAIC_NAME}_91-1573_quad.tif).

Output dimensions are always 512x512 pixels matching the footprint extent.
Centroids in metadata are in WGS84 (4326).
"""

import os
import sys
import re
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds
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

BUCKET                = require_env("BUCKET")
DATA_ROOT             = require_env("DATA_ROOT")
POSITIVE_GEOJSON_BLOB = require_env("POSITIVE_GEOJSON")
IGNORE_GEOJSON_BLOB   = require_env("IGNORE_GEOJSON")
METADATA_SUBREGIONS   = require_env("METADATA_SUBREGIONS")
TILE_BOUNDARIES_BLOB  = require_env("TILE_BOUNDARIES_GEOJSON")  # Planet grid GeoJSON with train_selected, Name, geometry
INPUT_PREFIX          = require_env("INPUT_PREFIX").rstrip("/")  # e.g. abrupt_thaw/planet_basemaps/global_quarterly/2024/q3
MOSAIC_NAME           = require_env("MOSAIC_NAME")               # e.g. global_quarterly_2024q3_mosaic
WORK_DIR              = require_env("WORK_DIR")
MAX_WORKERS           = int(require_env("MAX_WORKERS"))
METADATA_VERSION      = require_env("METADATA_VERSION")
METADATA_FILENAME     = os.environ.get("METADATA_FILENAME", "metadata.csv")

_test_limit = os.environ.get("TEST_LIMIT")
TEST_LIMIT  = int(_test_limit) if _test_limit else None

TILE_SIZE        = 512
WORKING_CRS      = "EPSG:6933"

RGB_PREFIX       = f"{DATA_ROOT}PLANET-RGB/"
LABELS_PREFIX    = f"{DATA_ROOT}labels/"
METADATA_PREFIX  = f"{DATA_ROOT}"
METADATA_COLUMNS = ["Tile_ID", "centroid_lat", "centroid_lon", "TrainClass", "RegionName", "UIDs", "Version"]

auth.authenticate_user()

os.makedirs(f"{WORK_DIR}/input",  exist_ok=True)
os.makedirs(f"{WORK_DIR}/output", exist_ok=True)

client = storage.Client()
bucket = client.bucket(BUCKET)

positive_local        = f"{WORK_DIR}/input/positive.geojson"
ignore_local          = f"{WORK_DIR}/input/ignore.geojson"
regions_local         = f"{WORK_DIR}/input/regions.geojson"
tile_boundaries_local = f"{WORK_DIR}/input/tile_boundaries.geojson"

bucket.blob(POSITIVE_GEOJSON_BLOB).download_to_filename(positive_local)
bucket.blob(IGNORE_GEOJSON_BLOB).download_to_filename(ignore_local)
bucket.blob(METADATA_SUBREGIONS).download_to_filename(regions_local)
bucket.blob(TILE_BOUNDARIES_BLOB).download_to_filename(tile_boundaries_local)

gdf_positive = gpd.read_file(positive_local)
gdf_ignore   = gpd.read_file(ignore_local)
gdf_regions  = gpd.read_file(regions_local)

# --- Load tile boundaries --------------------------------------------------

gdf_grid = gpd.read_file(tile_boundaries_local)

for col in ("train_selected", "Name"):
    if col not in gdf_grid.columns:
        print(f"ERROR: '{col}' column not found in tile boundaries GeoJSON. "
              f"Available columns: {list(gdf_grid.columns)}")
        sys.exit(1)

gdf_selected = gdf_grid[gdf_grid["train_selected"] == 1].copy().reset_index(drop=True)

if gdf_selected.empty:
    print("ERROR: No tiles with train_selected == 1 found.")
    sys.exit(1)

print(f"Grid loaded: {len(gdf_grid)} total quads, {len(gdf_selected)} selected for training")

if TEST_LIMIT:
    gdf_selected = gdf_selected.head(TEST_LIMIT)
    print(f"TEST_LIMIT={TEST_LIMIT}: using first {TEST_LIMIT} selected tiles")


# --- Blob path construction ------------------------------------------------
# Name format: tile_{col}_{row}_c{n}_r{n}  e.g. tile_91_1573_c1_r6
# Produces:    {INPUT_PREFIX}/{col}/{row}/{MOSAIC_NAME}_{col}-{row}_quad.tif

_NAME_RE = re.compile(r"^tile_(\d+)_(\d+)_")

def name_to_blob(name: str) -> str | None:
    m = _NAME_RE.match(name)
    if not m:
        return None
    col, row = m.group(1), m.group(2)
    filename = f"{MOSAIC_NAME}_{col}-{row}_quad.tif"
    return f"{INPUT_PREFIX}/{col}/{row}/{filename}"


# --- Ecoregions ------------------------------------------------------------

if "ECO_NAME" not in gdf_regions.columns:
    print(f"ERROR: 'ECO_NAME' column not found in regions GeoJSON. Available: {list(gdf_regions.columns)}")
    sys.exit(1)

gdf_regions_work = gdf_regions.to_crs(WORKING_CRS)
_region_tree     = STRtree(gdf_regions_work.geometry.centroid.values)


def find_nearest_region(centroid_lon: float, centroid_lat: float) -> str:
    projector = pyproj.Transformer.from_crs(4326, WORKING_CRS, always_xy=True).transform
    pt_work   = shapely_transform(projector, Point(centroid_lon, centroid_lat))
    return gdf_regions_work.iloc[_region_tree.nearest(pt_work)]["ECO_NAME"]


# --- Build tasks -----------------------------------------------------------

tasks          = []
bad_name_count = 0

for _, row in gdf_selected.iterrows():
    blob_path = name_to_blob(row["Name"])
    if blob_path is None:
        bad_name_count += 1
        print(f"WARNING: could not parse col/row from Name '{row['Name']}' — skipping")
        continue
    tasks.append({
        "footprint_geom": row.geometry,
        "blob_path":      blob_path,
    })

if bad_name_count:
    print(f"WARNING: {bad_name_count} tiles skipped due to unparseable Name values")

print(f"Tasks built: {len(tasks)}")


# --- Resume support --------------------------------------------------------

metadata_blob_path = f"{METADATA_PREFIX}{METADATA_FILENAME}"
metadata_blob      = bucket.blob(metadata_blob_path)
existing_df        = None
done_centroids     = set()

if metadata_blob.exists():
    existing_local = f"{WORK_DIR}/input/metadata_existing.csv"
    metadata_blob.download_to_filename(existing_local)
    existing_df = pd.read_csv(existing_local, dtype={"Tile_ID": str})
    done_centroids = set(
        zip(
            existing_df["centroid_lat"].round(6),
            existing_df["centroid_lon"].round(6),
        )
    )
    print(f"Found existing metadata: {len(existing_df)} rows, {len(done_centroids)} centroids done")
else:
    print("No existing metadata found - starting fresh")

tasks_to_run = tasks
print(f"{len(done_centroids)} tiles already in metadata, {len(tasks_to_run)} tasks queued")

if not tasks_to_run:
    print("All tiles already processed")
    sys.exit(0)


# --- UID derivation --------------------------------------------------------

_GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"

def make_tile_uid(lat: float, lon: float, precision: int = 12) -> str:
    lat_range, lon_range = [-90.0, 90.0], [-180.0, 180.0]
    bits    = [16, 8, 4, 2, 1]
    bit_idx = 0
    even    = True
    ch      = 0
    result  = []
    while len(result) < precision:
        if even:
            mid = (lon_range[0] + lon_range[1]) / 2
            if lon >= mid:
                ch |= bits[bit_idx]
                lon_range[0] = mid
            else:
                lon_range[1] = mid
        else:
            mid = (lat_range[0] + lat_range[1]) / 2
            if lat >= mid:
                ch |= bits[bit_idx]
                lat_range[0] = mid
            else:
                lat_range[1] = mid
        even = not even
        if bit_idx < 4:
            bit_idx += 1
        else:
            result.append(_GEOHASH_BASE32[ch])
            ch      = 0
            bit_idx = 0
    return "".join(result)


# --- Worker ----------------------------------------------------------------

def worker_init(positive_path, ignore_path, bucket_name):
    global _gdf_positive, _gdf_ignore, _gcs_client, _gcs_bucket
    _gdf_positive = gpd.read_file(positive_path)
    _gdf_ignore   = gpd.read_file(ignore_path)
    _gcs_client   = storage.Client()
    _gcs_bucket   = _gcs_client.bucket(bucket_name)


def process_single_tile(task: dict, work_dir: str, footprint_crs_epsg: int):
    footprint_geom = task["footprint_geom"]
    blob_path      = task["blob_path"]

    base_name       = blob_path.split("/")[-1].replace(".tif", "")
    local_input     = f"{work_dir}/input/{base_name}.tif"
    local_rgb_out   = f"{work_dir}/output/rgb_{base_name}.tif"
    local_label_out = f"{work_dir}/output/label_{base_name}.tif"

    try:
        _gcs_bucket.blob(blob_path).download_to_filename(local_input)

        with rasterio.open(local_input) as src:
            # Reproject footprint to quad's native CRS
            transformer      = pyproj.Transformer.from_crs(
                footprint_crs_epsg, src.crs.to_epsg(), always_xy=True
            ).transform
            footprint_native = shapely_transform(transformer, footprint_geom)

            minx, miny, maxx, maxy = footprint_native.bounds

            # Window from exact footprint bounds
            win = from_bounds(minx, miny, maxx, maxy, src.transform)
            # Clamp to valid raster extent (guards against sub-pixel float drift)
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

            if win.width < 1 or win.height < 1:
                return None

            try:
                rgb_data = src.read(
                    out_shape=(src.count, TILE_SIZE, TILE_SIZE),
                    window=win,
                )
            except Exception:
                return None

            chip_tf     = rasterio.windows.transform(win, src.transform)
            tile_bbox   = box(*rasterio.transform.array_bounds(TILE_SIZE, TILE_SIZE, chip_tf))
            native_crs  = src.crs
            tile_nodata = src.nodata

        # --- Label rasterization ---
        def subset_to_chip(gdf):
            if gdf.crs != native_crs:
                gdf = gdf.to_crs(native_crs)
            gdf = gdf.copy()
            gdf["geometry"] = gdf["geometry"].buffer(0)
            return gdf[gdf.intersects(tile_bbox)]

        raster_kwargs = dict(
            out_shape=(TILE_SIZE, TILE_SIZE),
            transform=chip_tf,
            fill=0,
            dtype=np.uint8,
        )

        new_mask      = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
        pos_subset    = subset_to_chip(_gdf_positive)
        ignore_subset = subset_to_chip(_gdf_ignore)

        if len(ignore_subset) > 0:
            new_mask = rasterize([(geom, 255) for geom in ignore_subset.geometry], **raster_kwargs)
        if len(pos_subset) > 0:
            pos_raster = rasterize([(geom, 1) for geom in pos_subset.geometry], **raster_kwargs)
            new_mask[pos_raster == 1] = 1

        if new_mask.max() == 0:
            return None

        # Centroid in WGS84
        project_to_wgs84 = pyproj.Transformer.from_crs(
            native_crs.to_epsg(), 4326, always_xy=True
        ).transform
        centroid     = shapely_transform(project_to_wgs84, tile_bbox.centroid)
        centroid_lon = centroid.x
        centroid_lat = centroid.y

        # --- Write outputs ---
        base_profile = dict(
            driver="GTiff",
            width=TILE_SIZE,
            height=TILE_SIZE,
            crs=native_crs,
            transform=chip_tf,
            compress="LZW",
        )
        if tile_nodata is not None:
            base_profile["nodata"] = tile_nodata

        with rasterio.open(
            local_rgb_out, "w",
            **{**base_profile, "dtype": rgb_data.dtype, "count": src.count}
        ) as dst:
            dst.write(rgb_data)
            dst.colorinterp = src.colorinterp
            for i, desc in enumerate(src.descriptions, start=1):
                if desc:
                    dst.set_band_description(i, desc)

        with rasterio.open(
            local_label_out, "w",
            **{**base_profile, "dtype": np.uint8, "count": 1}
        ) as dst:
            dst.write(new_mask[np.newaxis, :, :])
            dst.set_band_description(1, "Mask: 0=background 1=positive 255=ignore")

        return (local_rgb_out, local_label_out, centroid_lat, centroid_lon)

    finally:
        if os.path.exists(local_input):
            os.remove(local_input)


# --- Run -------------------------------------------------------------------

footprint_crs_epsg = gdf_selected.crs.to_epsg()
print(f"Processing {len(tasks_to_run)} tiles with {MAX_WORKERS} workers\n")

success_count   = 0
skip_count      = 0
error_count     = 0
duplicate_count = 0
metadata_rows   = []

with concurrent.futures.ProcessPoolExecutor(
    max_workers=MAX_WORKERS,
    initializer=worker_init,
    initargs=(positive_local, ignore_local, BUCKET),
) as executor:

    future_to_task = {
        executor.submit(process_single_tile, task, WORK_DIR, footprint_crs_epsg): task
        for task in tasks_to_run
    }

    with tqdm(total=len(tasks_to_run), desc="processing tiles", unit="tile") as pbar:
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                if result is not None:
                    local_rgb, local_label, centroid_lat, centroid_lon = result
                    centroid_key = (round(centroid_lat, 6), round(centroid_lon, 6))

                    if centroid_key in done_centroids:
                        duplicate_count += 1
                        tqdm.write(f" already done  {task['blob_path'].split('/')[-1]}  (centroid match)")
                        os.remove(local_rgb)
                        os.remove(local_label)
                    else:
                        uid_str = make_tile_uid(centroid_key[0], centroid_key[1])

                        bucket.blob(f"{RGB_PREFIX}{uid_str}.tif").upload_from_filename(local_rgb)
                        bucket.blob(f"{LABELS_PREFIX}{uid_str}.tif").upload_from_filename(local_label)
                        os.remove(local_rgb)
                        os.remove(local_label)

                        region_name = find_nearest_region(centroid_lon, centroid_lat)

                        metadata_rows.append({
                            "Tile_ID":      uid_str,
                            "centroid_lat": centroid_key[0],
                            "centroid_lon": centroid_key[1],
                            "TrainClass":   "positive",
                            "RegionName":   region_name,
                            "UIDs":         9999,
                            "Version":      METADATA_VERSION,
                        })

                        done_centroids.add(centroid_key)
                        success_count += 1
                else:
                    skip_count += 1

            except Exception as exc:
                error_count += 1
                if error_count <= 5:
                    tqdm.write(f"ERROR: {task.get('blob_path', '?').split('/')[-1]} — {type(exc).__name__}: {exc}")

            pbar.update(1)


# --- Write metadata --------------------------------------------------------

if metadata_rows:
    new_df    = pd.DataFrame(metadata_rows, columns=METADATA_COLUMNS)
    local_csv = f"{WORK_DIR}/output/metadata.csv"

    if existing_df is not None:
        if "Version" not in existing_df.columns:
            existing_df["Version"] = pd.NA
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.reindex(columns=METADATA_COLUMNS)
    combined.to_csv(local_csv, index=False)

    bucket.blob(metadata_blob_path).upload_from_filename(local_csv)
    os.remove(local_csv)
    print(f"Metadata: {len(combined)} rows > gs://{BUCKET}/{metadata_blob_path}")
else:
    print("No successful tiles — metadata CSV not written")

print(f"\n  Written this run    : {success_count}")
print(f"  Skipped (no label)  : {skip_count}")
print(f"  Skipped (duplicate) : {duplicate_count}")
print(f"  Errors              : {error_count}")
print(f"RGB:      gs://{BUCKET}/{RGB_PREFIX}")
print(f"Labels:   gs://{BUCKET}/{LABELS_PREFIX}")

if metadata_rows or existing_df is not None:
    summary_df = combined if metadata_rows else existing_df.reindex(columns=METADATA_COLUMNS)

    print("\n-Metadata CSV head (5 rows)-")
    print(summary_df.head().to_string(index=False))

    print("\n-TrainClass counts-")
    counts = summary_df["TrainClass"].value_counts()
    for cls, n in counts.items():
        print(f"  {cls:<12} {n:>6} tiles")
    print(f"  {'TOTAL':<12} {len(summary_df):>6} tiles")

    print("\n-RegionName counts-")
    region_counts = summary_df["RegionName"].value_counts()
    for region, n in region_counts.items():
        print(f"  {region:<40} {n:>6} tiles")
else:
    print("\nNo metadata to summarise.")
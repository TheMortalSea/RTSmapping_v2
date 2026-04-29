# PLANET IS BGR not RGB — bands are 1=Blue, 2=Green, 3=Red
# This script creates one 512x512 RGB chip per sampled polygon,
# centered on the polygon and guaranteed to fully contain it.

import os
import sys
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.ops import transform as shapely_transform
from shapely.strtree import STRtree
from google.colab import auth
from google.cloud import storage
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import pyproj


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def require_env(name):
    val = os.environ.get(name)
    if val is None:
        print(f"ERROR: Required environment variable '{name}' is not set.")
        sys.exit(1)
    return val

BUCKET               = require_env("BUCKET")
POLYGON_GEOJSON_BLOB = require_env("POLYGON_GEOJSON_BLOB")
GRID_GEOJSON_BLOB    = require_env("GRID_GEOJSON_BLOB")
DATA_ROOT            = require_env("DATA_ROOT")
METADATA_SUBREGIONS  = require_env("METADATA_SUBREGIONS")
WORK_DIR             = require_env("WORK_DIR")
MAX_WORKERS          = int(require_env("MAX_WORKERS"))

_target      = os.environ.get("TARGET_TILES")
TARGET_TILES = int(_target) if _target else None

TILE_SIZE       = 512
WORKING_CRS     = "EPSG:6933"
RGB_PREFIX      = f"{DATA_ROOT}/PLANET-RGB/"
METADATA_PREFIX = f"{DATA_ROOT}/"

print("Configuration:")
print(f"  BUCKET:        {BUCKET}")
print(f"  RGB output:    gs://{BUCKET}/{RGB_PREFIX}")
print(f"  MAX_WORKERS:   {MAX_WORKERS}")
print(f"  TARGET_TILES:  {TARGET_TILES or 'all'}")

auth.authenticate_user()

os.makedirs(f"{WORK_DIR}/input",  exist_ok=True)
os.makedirs(f"{WORK_DIR}/output", exist_ok=True)

client = storage.Client()
bucket = client.bucket(BUCKET)


# ---------------------------------------------------------------------------
# Download datasets
# ---------------------------------------------------------------------------

print("\nDownloading datasets from GCS...")

polygon_local = f"{WORK_DIR}/input/polygons.geojson"
regions_local = f"{WORK_DIR}/input/regions.geojson"
grid_local    = f"{WORK_DIR}/input/grid.geojson"

bucket.blob(POLYGON_GEOJSON_BLOB).download_to_filename(polygon_local)
bucket.blob(METADATA_SUBREGIONS).download_to_filename(regions_local)
bucket.blob(GRID_GEOJSON_BLOB).download_to_filename(grid_local)

# Polygons — filter to negative, reproject, fix invalid geometries
gdf_polygons = gpd.read_file(polygon_local)
gdf_polygons = gdf_polygons[gdf_polygons["TrainClass"] == "Negative"].copy()
gdf_polygons = gdf_polygons.to_crs(WORKING_CRS)
gdf_polygons["geometry"] = gdf_polygons["geometry"].buffer(0)
print(f"  Negative polygons:  {len(gdf_polygons)}")

# Regions
gdf_regions      = gpd.read_file(regions_local)
gdf_regions_work = gdf_regions.to_crs(WORKING_CRS)

if "ECO_NAME" not in gdf_regions.columns:
    print(f"ERROR: 'ECO_NAME' not found. Available: {list(gdf_regions.columns)}")
    sys.exit(1)
print(f"  Ecoregions:         {len(gdf_regions)}")

# Grid — reproject to working CRS for spatial joins
gdf_grid = gpd.read_file(grid_local)
gdf_grid = gdf_grid.to_crs(WORKING_CRS)
print(f"  Grid cells:         {len(gdf_grid)}")


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

# Nearest fallback for any centroid that missed all regions
missing = gdf_polygons["ECO_NAME"].isna()
if missing.any():
    region_tree = STRtree(gdf_regions_work.geometry.centroid.values)
    for idx in gdf_polygons[missing].index:
        nearest_i = region_tree.nearest(gdf_polygons.loc[idx, "geometry"].centroid)
        gdf_polygons.at[idx, "ECO_NAME"] = gdf_regions_work.iloc[nearest_i]["ECO_NAME"]
    print(f"  {missing.sum()} polygons assigned via nearest centroid")


# ---------------------------------------------------------------------------
# Stratified sampling — TARGET_TILES controls exact output count
#
# We sample TARGET_TILES polygons total, stratified by ecoregion.
# Each polygon produces exactly one output chip, so sampled count == output count
# (minus any skips for oversized or no-data tiles).
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

# If TARGET_TILES was set, trim any overshoot from rounding
if TARGET_TILES and len(gdf_sampled) > TARGET_TILES:
    gdf_sampled = gdf_sampled.sample(n=TARGET_TILES, random_state=42).reset_index(drop=True)

print(f"  Sampled polygons:   {len(gdf_sampled)}")

sampled_local = f"{WORK_DIR}/input/polygons_sampled.geojson"
gdf_sampled.to_file(sampled_local, driver="GeoJSON")


# ---------------------------------------------------------------------------
# Build one task per polygon — find the grid cell whose centroid covers it
# ---------------------------------------------------------------------------

print("\nBuilding per-polygon tasks...")

def grid_row_to_blob(row) -> str:
    delivery = row["delivery_location"].rstrip("/")
    filename = f"{row['basemap_name']}_{row['grid_column']}-{row['grid_row']}_quad.tif"
    return f"{delivery}/{filename}"

grid_sindex = gdf_grid.sindex

tasks        = []
no_grid_count = 0

for i, poly_row in gdf_sampled.iterrows():
    centroid = poly_row.geometry.centroid

    # Find grid cells whose geometry contains the polygon centroid
    candidate_idxs = list(grid_sindex.intersection(centroid.bounds))
    covering = [
        j for j in candidate_idxs
        if gdf_grid.iloc[j].geometry.contains(centroid)
    ]

    # Fallback: nearest grid cell by distance
    if not covering and candidate_idxs:
        covering = [min(
            candidate_idxs,
            key=lambda j: gdf_grid.iloc[j].geometry.distance(centroid)
        )]

    if not covering:
        no_grid_count += 1
        continue

    grid_row  = gdf_grid.iloc[covering[0]]
    blob_path = grid_row_to_blob(grid_row)

    tasks.append({
        "polygon_geom": poly_row.geometry,
        "polygon_idx":  i,
        "blob_path":    blob_path,
        "eco_name":     poly_row.get("ECO_NAME", ""),
    })

print(f"  Tasks built:        {len(tasks)}")
if no_grid_count:
    print(f"  Polygons with no covering grid cell (skipped): {no_grid_count}")


# ---------------------------------------------------------------------------
# Resume: skip already-processed polygons by checking existing Tile_ID count
# and re-deriving which polygon indices have been handled via blob_path matching.
# polygon_idx is kept in task dicts (in memory only) and never written to CSV.
# ---------------------------------------------------------------------------

metadata_blob_path = f"{METADATA_PREFIX}metadata.csv"
existing_blob      = bucket.blob(metadata_blob_path)
processed_blob_paths = set()
existing_df          = None

if existing_blob.exists():
    existing_local = f"{WORK_DIR}/input/metadata_existing.csv"
    existing_blob.download_to_filename(existing_local)
    existing_df = pd.read_csv(existing_local, dtype={"Tile_ID": str})

existing_uids = []
if existing_df is not None and "Tile_ID" in existing_df.columns:
    existing_uids = [int(v) for v in existing_df["Tile_ID"] if str(v).isdigit()]
next_uid = max(existing_uids) + 1 if existing_uids else 1

# Derive already-processed polygon indices from the task list length vs existing rows.
# Since each task maps 1:1 to a Tile_ID (in order), we skip the first N tasks where
# N = number of rows already in the metadata CSV.
n_existing = len(existing_df) if existing_df is not None else 0
tasks_to_run = tasks[n_existing:]

print(f"  Already processed:   {n_existing} tiles")
print(f"  After resume filter: {len(tasks_to_run)} tasks remaining")
print(f"  Next UID:            {next_uid:06d}")

if not tasks_to_run:
    print("All polygons already processed.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Helper — centered 512×512 window that fully contains the polygon
# ---------------------------------------------------------------------------

def get_containing_window(src, poly_bounds_native, tile_size=512):
    """
    Return a Window (pixel space) of exactly tile_size×tile_size centred on
    the polygon bounding box. Returns None if:
      - the polygon bbox is larger than tile_size pixels in either dimension
      - the raster is smaller than tile_size pixels in either dimension
    The window is clamped to the raster extent when the centroid is near an edge.
    """
    minx, miny, maxx, maxy = poly_bounds_native

    # Polygon extent in pixel space — check it fits
    tl_row, tl_col = rowcol(src.transform, minx, maxy)
    br_row, br_col = rowcol(src.transform, maxx, miny)
    poly_px_w = abs(br_col - tl_col)
    poly_px_h = abs(br_row - tl_row)
    if poly_px_w > tile_size or poly_px_h > tile_size:
        return None  # polygon too large to fit in a single chip

    if src.width < tile_size or src.height < tile_size:
        return None  # source quad too small

    # Centre of polygon in pixel space
    cx, cy           = (minx + maxx) / 2, (miny + maxy) / 2
    center_row, center_col = rowcol(src.transform, cx, cy)

    half    = tile_size // 2
    col_off = int(center_col) - half
    row_off = int(center_row) - half

    # Clamp so window stays within raster bounds
    col_off = max(0, min(col_off, src.width  - tile_size))
    row_off = max(0, min(row_off, src.height - tile_size))

    return Window(col_off, row_off, tile_size, tile_size)


# ---------------------------------------------------------------------------
# Worker — per-process initialisation caches GCS client and CRS transformers
# ---------------------------------------------------------------------------

def worker_init(sampled_polygon_path: str):
    global _gdf_polygons, _polygon_sindex
    global _gcs_bucket
    global _project_to_wgs84

    _gdf_polygons   = gpd.read_file(sampled_polygon_path)   # already WORKING_CRS
    _polygon_sindex = _gdf_polygons.sindex

    _gcs_client  = storage.Client()
    _gcs_bucket  = _gcs_client.bucket(BUCKET)

    # WGS84 transformer — Planet tiles are always EPSG:3857
    _project_to_wgs84 = pyproj.Transformer.from_crs(3857, 4326, always_xy=True).transform


# ---------------------------------------------------------------------------
# Per-polygon worker
# ---------------------------------------------------------------------------

def process_single_tile(task: dict, work_dir: str):
    """
    Download the quad that covers this polygon, extract a centered 512×512
    chip, reorder bands from BGR → RGB, and return the local output path
    plus the chip centroid in WGS84.

    Returns (local_rgb_path, centroid_lat, centroid_lon) on success, or None.
    """
    poly_geom  = task["polygon_geom"]
    poly_idx   = task["polygon_idx"]
    blob_path  = task["blob_path"]

    base_name     = f"poly_{poly_idx:06d}"
    local_input   = f"{work_dir}/input/{base_name}.tif"
    local_rgb_out = f"{work_dir}/output/rgb_{base_name}.tif"

    try:
        _gcs_bucket.blob(blob_path).download_to_filename(local_input)

        with rasterio.open(local_input) as src:
            # Project polygon from WORKING_CRS → raster native CRS (EPSG:3857)
            project_to_native = pyproj.Transformer.from_crs(
                WORKING_CRS, src.crs.to_epsg(), always_xy=True
            ).transform
            poly_native = shapely_transform(project_to_native, poly_geom)

            win = get_containing_window(src, poly_native.bounds, TILE_SIZE)
            if win is None:
                return None     # polygon too large or quad too small

            # Read bands 1=Blue, 2=Green, 3=Red and reorder to R, G, B
            try:
                bgr      = src.read(indexes=[1, 2, 3], window=win)  # B, G, R
                rgb_data = bgr[[2, 1, 0], :, :]                     # → R, G, B
            except Exception:
                return None

            # Skip entirely no-data chips
            if src.nodata is not None and (rgb_data == src.nodata).all():
                return None

            chip_tf      = rasterio.windows.transform(win, src.transform)
            tile_bbox    = box(*rasterio.transform.array_bounds(
                win.height, win.width, chip_tf
            ))
            centroid_wgs84 = shapely_transform(_project_to_wgs84, tile_bbox.centroid)

            native_crs = src.crs   # captured before closing

        profile = dict(
            driver="GTiff", count=3, dtype=rgb_data.dtype,
            width=TILE_SIZE, height=TILE_SIZE,
            crs=native_crs, transform=chip_tf, compress="LZW",
        )
        with rasterio.open(local_rgb_out, "w", **profile) as dst:
            dst.write(rgb_data)
            dst.set_band_description(1, "Red")
            dst.set_band_description(2, "Green")
            dst.set_band_description(3, "Blue")

        return (local_rgb_out, centroid_wgs84.y, centroid_wgs84.x)

    finally:
        if os.path.exists(local_input):
            os.remove(local_input)


# ---------------------------------------------------------------------------
# STRtree for fast ecoregion lookup (main process only)
# ---------------------------------------------------------------------------

_region_tree = STRtree(gdf_regions_work.geometry.centroid.values)

def find_nearest_region(centroid_lon: float, centroid_lat: float) -> str:
    projector = pyproj.Transformer.from_crs(4326, WORKING_CRS, always_xy=True).transform
    pt_work   = shapely_transform(projector, Point(centroid_lon, centroid_lat))
    nearest_i = _region_tree.nearest(pt_work)
    return gdf_regions_work.iloc[nearest_i]["ECO_NAME"]


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

print(f"\nProcessing {len(tasks_to_run)} polygons with {MAX_WORKERS} workers...\n")

success_count = 0
skip_count    = 0
error_count   = 0
metadata_rows = []

with concurrent.futures.ProcessPoolExecutor(
    max_workers=MAX_WORKERS,
    initializer=worker_init,
    initargs=(sampled_local,),
) as executor:

    future_to_task = {
        executor.submit(process_single_tile, task, WORK_DIR): task
        for task in tasks_to_run
    }

    with tqdm(total=len(tasks_to_run), desc="tiles", unit="tile") as pbar:
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                if result is not None:
                    local_rgb, centroid_lat, centroid_lon = result
                    uid_str = f"{next_uid:06d}"

                    bucket.blob(f"{RGB_PREFIX}{uid_str}.tif").upload_from_filename(local_rgb)
                    os.remove(local_rgb)

                    region_name = find_nearest_region(centroid_lon, centroid_lat)

                    metadata_rows.append({
                        "Tile_ID":      uid_str,
                        "centroid_lat": round(centroid_lat, 6),
                        "centroid_lon": round(centroid_lon, 6),
                        "TrainClass":   "negative",
                        "RegionName":   region_name,
                        "UIDs":         9999,
                    })

                    next_uid      += 1
                    success_count += 1
                    tqdm.write(f"✓ {uid_str}  poly_{task['polygon_idx']:06d}  {task['blob_path'].split('/')[-1]}")
                else:
                    skip_count += 1
                    tqdm.write(f"↷ skipped  poly_{task['polygon_idx']:06d}  (oversized or no-data)")

            except Exception as exc:
                error_count += 1
                tqdm.write(f"✗ poly_{task['polygon_idx']:06d} — {exc}")

            pbar.update(1)


# ---------------------------------------------------------------------------
# Write metadata
# ---------------------------------------------------------------------------

if metadata_rows:
    new_df = pd.DataFrame(
        metadata_rows,
        columns=["Tile_ID", "centroid_lat", "centroid_lon",
                 "TrainClass", "RegionName", "UIDs"],
    )
    local_csv = f"{WORK_DIR}/output/metadata.csv"

    if existing_df is not None:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        print(f"\nAppended {len(new_df)} rows → {len(combined)} total")
    else:
        combined = new_df
        print(f"\nNew metadata CSV with {len(combined)} rows")

    combined.to_csv(local_csv, index=False)
    bucket.blob(metadata_blob_path).upload_from_filename(local_csv)
    os.remove(local_csv)
    print(f"Metadata → gs://{BUCKET}/{metadata_blob_path}")
else:
    print("\nNo successful tiles — metadata not written.")

print(f"\n{success_count} written | {skip_count} skipped | {error_count} errors")
print(f"RGB chips: gs://{BUCKET}/{RGB_PREFIX}")
print(f"Metadata:  gs://{BUCKET}/{METADATA_PREFIX}")
"""0.5x-scale (2x GSD) training tile creation — multiscale POC (Phase A).

Cuts 512x512 tiles at ~9.55 m/px (2x the native ~4.78 m/px) from the 2024
PlanetScope quarterly quads: each tile is a 1024x1024-px native window
bilinear-downsampled to 512 — the same decimated read the inference pipeline
uses for scale-0.5 (`inference/tiles.py::_read_window_with_retry`), per the
train-inference consistency contract (CLAUDE.md Rule 3).

Positive tiles: quad-aligned 1024-px blocks (2x2 groups of the Planet 512
grid) covering the v1.0 `train_selected` footprints. Quad-alignment
guarantees every block reads from exactly one quad (4096 = 4 x 1024).

Negative tiles: 1024-px windows centered on the ARTS v6 Negative polygon
centroids (the v1.0 negative source), clamped inside their quad; windows that
intersect any positive/ignore/unrefined-ARTS geometry are skipped so the
expanded context cannot smuggle RTS into background tiles.

Label rules at 0.5x (documented in data/data.md §"0.5x-scale staging"):
  1. ignore features that intersect >=1 refined positive polygon are
     AUTO-CONVERTED to positive (they were ignore for lack of within-tile
     context; at 4x FOV the context is present) — user decision 2026-07-02;
  2. remaining ignore features stay 255;
  3. ARTS positives with no overlap with any refined positive ("unrefined
     ARTS") rasterize as 255: known-but-undelineated RTS in the expanded
     context must not train as background;
  4. positives (refined + auto-converted) rasterize as 1, overwriting 255;
  5. sub-pixel guard: any positive feature covering < MIN_POSITIVE_PX pixels
     at 0.5x is re-burned as 255 (below diagnostic size, data.md §2.2).

Outputs (same layout/conventions as v1.0):
  {out_root}/PLANET-RGB/{uid}.tif   3-band uint8
  {out_root}/labels/{uid}.tif      single-band uint8 (positives only)
  {out_root}/metadata.csv          Tile_ID, centroid_lat, centroid_lon,
                                   TrainClass, RegionName, UIDs, Version

Run inside the rts-train:v2 container (GOOGLE_APPLICATION_CREDENTIALS set):
  python scripts/scale05_tile_creation.py \
      --out-root gs://rts-mapping-v2/training/v1.0_scale05 --workers 32
Vector inputs default to the verified v1.0 staging sources (provenance check
2026-07-02: 57/60 sampled v1.0 label tiles reproduce exactly from them; the
3 mismatches are later hotfix edits).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from shapely.geometry import Point, box
from shapely.strtree import STRtree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.quad_index import QUAD_SIZE_M, RESOLUTION_M, WORLD_MIN  # noqa: E402

logger = logging.getLogger("scale05_tile_creation")

TILE_SIZE = 512                      # output pixels
SCALE = 0.5
RES_05 = RESOLUTION_M / SCALE        # ~9.5546 m/px
BLOCK_M = TILE_SIZE * RES_05         # 1024 native px = QUAD_SIZE_M / 4
MIN_POSITIVE_PX = 10                 # sub-pixel guard (rule 5)
ARTS_BUFFER_M = 50.0                 # unrefined-ARTS ignore buffer (RTS growth since ARTS date)
MAX_NODATA_FRAC = 0.5                # skip tiles mostly NoData/black
METADATA_COLUMNS = ["Tile_ID", "centroid_lat", "centroid_lon",
                    "TrainClass", "RegionName", "UIDs", "Version"]

# Verified v1.0 staging inputs (provenance check, see module docstring).
DEFAULT_VECTORS = {
    "positive": [
        "gs://abrupt_thaw/RTS_MODEL_V2/DATA/training_labels/training_positive_labels_03_03_2026.geojson",
        "gs://abrupt_thaw/RTS_MODEL_V2/DATA/training_labels/batch2_positive_labels.geojson",
        "gs://abrupt_thaw/RTS_MODEL_V2/DATA/training_labels/batch3_positive_labels.geojson",
        "gs://abrupt_thaw/RTS_MODEL_V2/DATA/training_labels/rts_labels_tem_batch2_2.geojson",
    ],
    "ignore": [
        "gs://abrupt_thaw/RTS_MODEL_V2/DATA/training_labels/ignore_regions_3Mar2026.geojson",
        "gs://abrupt_thaw/RTS_MODEL_V2/DATA/training_labels/batch2_ignore.json",
        "gs://abrupt_thaw/RTS_MODEL_V2/DATA/training_labels/batch3_ignore_regions.geojson",
    ],
    "footprints": [
        "gs://abrupt_thaw/RTS_MODEL_V2/DATA/training_labels/batch1_Footprint_1939_.geojson",
        "gs://abrupt_thaw/RTS_MODEL_V2/DATA/training_labels/batch2_footprints.geojson",
        "gs://abrupt_thaw/RTS_MODEL_V2/DATA/training_labels/batch3_footprints.geojson",
    ],
    "arts": "gs://abrupt_thaw/RTS_MODEL_V2/DATA/ARTS_main_dataset_v.6.0.0.geojson",
    "regions": "gs://rts-mapping-v2/training/v1.0/circumpolar_subregions.geojson",
}
DEFAULT_INPUT_PREFIX = "gs://abrupt_thaw/planet_basemaps/global_quarterly/2024/q3"
DEFAULT_MOSAIC_NAME = "global_quarterly_2024q3_mosaic"


# ---------------------------------------------------------------------------
# Pure geometry / label helpers (unit-tested in tests/test_scale05_staging.py)
# ---------------------------------------------------------------------------


def block_index(x: float, y: float) -> tuple[int, int]:
    """Quad-aligned 1024-px block index containing EPSG:3857 point (x, y)."""
    return int((x - WORLD_MIN) // BLOCK_M), int((y - WORLD_MIN) // BLOCK_M)


def block_bounds(bx: int, by: int) -> tuple[float, float, float, float]:
    """Projected bounds of block (bx, by). Nests exactly 4x4 per mosaic quad."""
    minx = WORLD_MIN + bx * BLOCK_M
    miny = WORLD_MIN + by * BLOCK_M
    return minx, miny, minx + BLOCK_M, miny + BLOCK_M


def quad_for_block(bx: int, by: int) -> tuple[int, int]:
    """Mosaic quad (col, row) containing block (bx, by)."""
    return bx // 4, by // 4


def blocks_for_bounds(bounds: tuple[float, float, float, float]) -> set[tuple[int, int]]:
    """All block indices a footprint's bounds overlap (edge-safe)."""
    minx, miny, maxx, maxy = bounds
    eps = 1e-6  # keep shared edges from claiming the neighbour block
    bx0, by0 = block_index(minx + eps, miny + eps)
    bx1, by1 = block_index(maxx - eps, maxy - eps)
    return {(bx, by) for bx in range(bx0, bx1 + 1) for by in range(by0, by1 + 1)}


def clamp_window_to_quad(
    cx: float, cy: float
) -> tuple[tuple[float, float, float, float], tuple[int, int]]:
    """1024-px-native window centered on (cx, cy), shifted to fit its quad.

    Returns (bounds, (quad_col, quad_row)). The shift is at most half a window,
    so the source polygon centroid always stays inside the window.
    """
    qcol = int((cx - WORLD_MIN) // QUAD_SIZE_M)
    qrow = int((cy - WORLD_MIN) // QUAD_SIZE_M)
    qminx = WORLD_MIN + qcol * QUAD_SIZE_M
    qminy = WORLD_MIN + qrow * QUAD_SIZE_M
    minx = min(max(cx - BLOCK_M / 2, qminx), qminx + QUAD_SIZE_M - BLOCK_M)
    miny = min(max(cy - BLOCK_M / 2, qminy), qminy + QUAD_SIZE_M - BLOCK_M)
    return (minx, miny, minx + BLOCK_M, miny + BLOCK_M), (qcol, qrow)


class LabelVectors:
    """Pre-classified label geometries (EPSG:3857 shapely, plain lists).

    STRtrees are built lazily so instances can cross a process fork safely
    (read-only GEOS state must not cross the fork — see inference/tiles.py).
    """

    def __init__(self, positive: list, ignore_convert: list, ignore_keep: list,
                 arts_unrefined: list):
        self.positive = positive
        self.ignore_convert = ignore_convert
        self.ignore_keep = ignore_keep
        self.arts_unrefined = arts_unrefined
        self._trees: dict[str, STRtree] = {}

    def __getstate__(self):
        # STRtrees are per-process only; geometries pickle fine.
        state = self.__dict__.copy()
        state["_trees"] = {}
        return state

    def tree(self, name: str) -> STRtree:
        """Lazy per-process STRtree over one geometry group."""
        if name not in self._trees:
            geoms = getattr(self, name)
            self._trees[name] = STRtree(geoms if geoms else [box(0, 0, 0, 0)])
        return self._trees[name]

    def intersects(self, name: str, geom) -> bool:
        geoms = getattr(self, name)
        if not geoms:
            return False
        return len(self.tree(name).query(geom, predicate="intersects")) > 0

    @classmethod
    def classify(cls, positive: list, ignore: list, arts_positive: list) -> "LabelVectors":
        """Apply rules 1-3 globally: split ignore by touch-positive; find unrefined ARTS."""
        pos_tree = STRtree(positive)
        conv, keep = [], []
        for g in ignore:
            hits = pos_tree.query(g, predicate="intersects")
            (conv if len(hits) else keep).append(g)
        unrefined = [
            g.buffer(ARTS_BUFFER_M) for g in arts_positive
            if not len(pos_tree.query(g, predicate="intersects"))
        ]
        return cls(positive=positive, ignore_convert=conv, ignore_keep=keep,
                   arts_unrefined=unrefined)


def build_label(
    vectors: LabelVectors,
    bounds: tuple[float, float, float, float],
    tile_size: int = TILE_SIZE,
    ignore_index: int = 255,
    min_positive_px: int = MIN_POSITIVE_PX,
) -> np.ndarray | None:
    """Rasterize the 0.5x label for `bounds` per rules 1-5. None if no positive px."""
    minx, miny, maxx, maxy = bounds
    res = (maxx - minx) / tile_size
    tf = from_origin(minx, maxy, res, res)
    bbox = box(*bounds)
    kw = dict(out_shape=(tile_size, tile_size), transform=tf, fill=0, dtype=np.uint8)

    def hits(name: str) -> list:
        geoms = getattr(vectors, name)
        if not geoms:
            return []
        return [geoms[i] for i in vectors.tree(name).query(bbox, predicate="intersects")]

    label = np.zeros((tile_size, tile_size), dtype=np.uint8)
    ign = hits("ignore_keep") + hits("arts_unrefined")
    if ign:
        label = rasterize([(g.buffer(0), ignore_index) for g in ign], **kw)

    pos = hits("positive") + hits("ignore_convert")
    if pos:
        pos_r = rasterize([(g.buffer(0), 1) for g in pos], **kw)
        label[pos_r == 1] = 1
        # Rule 5: re-burn sub-pixel positive features as ignore.
        for g in pos:
            if g.area < 4 * min_positive_px * res * res:   # cheap prefilter
                r = rasterize([(g.buffer(0), 1)], **kw)
                n = int((r == 1).sum())
                if 0 < n < min_positive_px:
                    label[r == 1] = ignore_index

    if not (label == 1).any():
        return None
    return label


# UID derivation — identical geohash to positive/negative_tile_creation.py.
_GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def make_tile_uid(lat: float, lon: float, precision: int = 12) -> str:
    """Geohash of the tile centroid (same encoding as v1.0 staging scripts)."""
    lat_range, lon_range = [-90.0, 90.0], [-180.0, 180.0]
    bits = [16, 8, 4, 2, 1]
    bit_idx, even, ch = 0, True, 0
    result: list[str] = []
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
            ch, bit_idx = 0, 0
    return "".join(result)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def quad_path(input_prefix: str, mosaic_name: str, col: int, row: int) -> str:
    return f"{input_prefix}/{col}/{row}/{mosaic_name}_{col}-{row}_quad.tif"


def read_rgb_05(path: str, bounds: tuple[float, float, float, float]) -> np.ndarray | None:
    """Bilinear-decimated (3, 512, 512) uint8 read — mirrors inference/tiles.py.

    Returns None when the quad is missing or the window is mostly NoData.
    """
    try:
        with rasterio.open(path) as src:
            win = from_bounds(*bounds, transform=src.transform)
            rgb = src.read(indexes=(1, 2, 3), window=win, boundless=True,
                           fill_value=0, out_shape=(3, TILE_SIZE, TILE_SIZE),
                           resampling=Resampling.bilinear)
    except rasterio.errors.RasterioIOError as exc:
        logger.warning("Quad read failed %s: %s", path, exc)
        return None
    nodata_frac = float((rgb == 0).all(axis=0).mean())
    if nodata_frac > MAX_NODATA_FRAC:
        return None
    return rgb.astype(np.uint8)


def tile_bytes(arr: np.ndarray, bounds: tuple[float, float, float, float],
               band_desc: str | None = None) -> bytes:
    """Encode a (C, 512, 512) or (512, 512) array as EPSG:3857 GeoTIFF bytes."""
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    minx, miny, maxx, maxy = bounds
    tf = from_origin(minx, maxy, (maxx - minx) / arr.shape[2], (maxy - miny) / arr.shape[1])
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", width=arr.shape[2], height=arr.shape[1],
                      count=arr.shape[0], dtype=arr.dtype, crs="EPSG:3857",
                      transform=tf, compress="LZW") as dst:
            dst.write(arr)
            if band_desc:
                dst.set_band_description(1, band_desc)
        return mem.read()


_FS = None


def _fs():
    """Per-process gcsfs filesystem (lazy — must not cross the fork)."""
    global _FS
    if _FS is None:
        import gcsfs
        _FS = gcsfs.GCSFileSystem()
    return _FS


def write_bytes(uri: str, payload: bytes) -> None:
    if uri.startswith("gs://"):
        with _fs().open(uri, "wb") as f:
            f.write(payload)
    else:
        os.makedirs(os.path.dirname(uri), exist_ok=True)
        with open(uri, "wb") as f:
            f.write(payload)


def centroid_wgs84(bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    """(lat, lon) of the bounds center."""
    import pyproj
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    lon, lat = pyproj.Transformer.from_crs(3857, 4326, always_xy=True).transform(cx, cy)
    return lat, lon


# ---------------------------------------------------------------------------
# Vector loading + task building (parent process)
# ---------------------------------------------------------------------------


def _read_3857(uri: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(uri)
    if g.crs is None:
        g = g.set_crs(3857)
    return g.to_crs(3857)


def load_vectors(v: dict) -> tuple[LabelVectors, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load + classify all staging vectors.

    Returns (label_vectors, footprints_selected, arts_negatives, regions).
    """
    positive = [g for uri in v["positive"] for g in _read_3857(uri).geometry]
    ignore = [g for uri in v["ignore"] for g in _read_3857(uri).geometry]
    arts = _read_3857(v["arts"])
    arts_pos = list(arts[arts["TrainClass"] == "Positive"].geometry)
    arts_neg = arts[arts["TrainClass"] == "Negative"].copy()
    vectors = LabelVectors.classify(positive, ignore, arts_pos)
    logger.info(
        "Vectors: %d positive, %d ignore (%d auto-converted / %d kept), "
        "%d unrefined-ARTS ignore, %d ARTS negatives",
        len(positive), len(ignore), len(vectors.ignore_convert),
        len(vectors.ignore_keep), len(vectors.arts_unrefined), len(arts_neg))

    fps = pd.concat([_read_3857(uri) for uri in v["footprints"]], ignore_index=True)
    fps = fps[fps["train_selected"] == 1].reset_index(drop=True)
    logger.info("Footprints: %d selected 512-grid tiles", len(fps))

    regions = _read_3857(v["regions"])
    return vectors, fps, arts_neg, regions


def build_tasks(fps: gpd.GeoDataFrame, arts_neg: gpd.GeoDataFrame,
                input_prefix: str, mosaic_name: str) -> list[dict]:
    """Positive block tasks (deduped, quad-aligned) + negative window tasks.

    Sorted by quad path within each kind so consecutive tasks in a worker hit
    the same quad (GDAL's VSI curl cache stays warm).
    """
    blocks: set[tuple[int, int]] = set()
    for geom in fps.geometry:
        blocks |= blocks_for_bounds(geom.bounds)
    pos = []
    for bx, by in sorted(blocks):
        col, row = quad_for_block(bx, by)
        pos.append({"kind": "positive", "bounds": block_bounds(bx, by),
                    "quad": quad_path(input_prefix, mosaic_name, col, row)})
    neg = []
    for geom in arts_neg.geometry:
        c = geom.centroid
        bounds, (col, row) = clamp_window_to_quad(c.x, c.y)
        neg.append({"kind": "negative", "bounds": bounds,
                    "quad": quad_path(input_prefix, mosaic_name, col, row)})
    pos.sort(key=lambda t: t["quad"])
    neg.sort(key=lambda t: t["quad"])
    logger.info("Tasks: %d positive blocks, %d negative windows", len(pos), len(neg))
    return pos + neg


# ---------------------------------------------------------------------------
# Worker. Processes are SPAWNED, not forked: the parent holds live GCS/gRPC
# event-loop threads (vector reads, resume metadata read) and forked children
# segfault inside them (observed: cygrpc event_engine SIGSEGV). Workers get
# their state from a local pickle via the initializer instead.
# ---------------------------------------------------------------------------

_G: dict = {}


def _worker_init(state_pickle: str) -> None:
    import pickle
    logging.basicConfig(level=logging.WARNING)
    with open(state_pickle, "rb") as f:
        _G.update(pickle.load(f))


def process_task(task: dict) -> dict | None:
    """Full per-tile job: guards → read → label → write. Returns metadata row."""
    v: LabelVectors = _G["vectors"]
    bounds = task["bounds"]
    bbox = box(*bounds)

    if task["kind"] == "positive":
        label = build_label(v, bounds)
        if label is None:
            return {"skip": "no_label"}
    else:
        label = None
        if (v.intersects("positive", bbox) or v.intersects("ignore_convert", bbox)
                or v.intersects("ignore_keep", bbox) or v.intersects("arts_unrefined", bbox)):
            return {"skip": "dirty_context"}

    rgb = read_rgb_05(task["quad"], bounds)
    if rgb is None:
        return {"skip": "nodata_or_missing"}

    lat, lon = centroid_wgs84(bounds)
    key = (round(lat, 6), round(lon, 6))
    if key in _G["done"]:
        return {"skip": "duplicate"}
    uid = make_tile_uid(*key)

    out_root = _G["out_root"]
    write_bytes(f"{out_root}/PLANET-RGB/{uid}.tif", tile_bytes(rgb, bounds))
    if label is not None:
        write_bytes(f"{out_root}/labels/{uid}.tif",
                    tile_bytes(label, bounds,
                               band_desc="Mask: 0=background 1=positive 255=ignore"))

    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    regions: gpd.GeoDataFrame = _G["regions"]
    if "region_tree" not in _G:
        _G["region_tree"] = STRtree(list(regions.geometry))
    region = str(regions.iloc[int(_G["region_tree"].nearest(Point(cx, cy)))]["ECO_NAME"])

    return {"Tile_ID": uid, "centroid_lat": key[0], "centroid_lon": key[1],
            "TrainClass": task["kind"], "RegionName": region,
            "UIDs": 9999, "Version": _G["version"]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", required=True,
                    help="e.g. gs://rts-mapping-v2/training/v1.0_scale05")
    ap.add_argument("--input-prefix", default=DEFAULT_INPUT_PREFIX)
    ap.add_argument("--mosaic-name", default=DEFAULT_MOSAIC_NAME)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--test-limit", type=int, default=None,
                    help="process only the first N tasks of each kind")
    ap.add_argument("--positives-only", action="store_true")
    ap.add_argument("--metadata-version", default="scale05_v1")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    # Range-read friendliness for 21k+ windowed quad reads.
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", "200000000")
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")

    out_root = args.out_root.rstrip("/")
    vectors, fps, arts_neg, regions = load_vectors(DEFAULT_VECTORS)
    tasks = build_tasks(fps, arts_neg, args.input_prefix, args.mosaic_name)
    if args.positives_only:
        tasks = [t for t in tasks if t["kind"] == "positive"]
    if args.test_limit:
        pos = [t for t in tasks if t["kind"] == "positive"][: args.test_limit]
        neg = [t for t in tasks if t["kind"] == "negative"][: args.test_limit]
        tasks = pos + neg

    # Resume: skip tiles whose centroid is already in metadata.csv.
    meta_uri = f"{out_root}/metadata.csv"
    try:
        existing = pd.read_csv(meta_uri, dtype={"Tile_ID": str})
        done = set(zip(existing["centroid_lat"].round(6), existing["centroid_lon"].round(6)))
        logger.info("Resume: %d rows already in metadata", len(existing))
    except FileNotFoundError:
        existing, done = None, set()

    import multiprocessing
    import pickle
    import tempfile
    state = {"vectors": vectors, "regions": regions, "done": done,
             "out_root": out_root, "version": args.metadata_version}
    state_pickle = os.path.join(tempfile.mkdtemp(prefix="scale05_"), "state.pkl")
    with open(state_pickle, "wb") as f:
        pickle.dump(state, f)

    from collections import Counter
    counts: Counter = Counter()
    rows: list[dict] = []
    seen_ids: set[str] = set(existing["Tile_ID"]) if existing is not None else set()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init,
        initargs=(state_pickle,),
    ) as ex:
        for i, res in enumerate(ex.map(process_task, tasks, chunksize=8), 1):
            if res is None:
                counts["error"] += 1
            elif "skip" in res:
                counts[res["skip"]] += 1
            elif res["Tile_ID"] in seen_ids:
                # e.g. two negative centroids clamped to the same quad-corner window
                counts["duplicate"] += 1
            else:
                seen_ids.add(res["Tile_ID"])
                rows.append(res)
                counts["written"] += 1
            if i % 500 == 0 or i == len(tasks):
                logger.info("%d/%d tasks | %s", i, len(tasks), dict(counts))
                if rows:  # checkpoint metadata so long runs are resumable
                    _write_metadata(meta_uri, existing, rows)

    if rows:
        combined = _write_metadata(meta_uri, existing, rows)
        logger.info("Metadata: %d rows -> %s", len(combined), meta_uri)
        logger.info("TrainClass counts: %s", combined["TrainClass"].value_counts().to_dict())
    logger.info("Done: %s", dict(counts))


def _write_metadata(meta_uri: str, existing: pd.DataFrame | None,
                    rows: list[dict]) -> pd.DataFrame:
    new_df = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    combined = pd.concat([existing, new_df], ignore_index=True) if existing is not None else new_df
    combined = combined.reindex(columns=METADATA_COLUMNS)
    combined.to_csv(meta_uri, index=False)
    return combined


if __name__ == "__main__":
    main()

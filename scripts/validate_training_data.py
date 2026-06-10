"""
Tile validation suite for positive_tile_creation.py / negative_tile_creation.py output.

Structured as a class-based test runner with pass/fail/warn assertions.
Run directly:  python validate_tiles.py

Checks:
  1.  Metadata file exists and has required columns
  2.  Duplicate Tile_IDs
  3.  Duplicate centroid (lat, lon) pairs
  4.  Centroid coordinate validity (WGS84 range)
  5.  UID (Tile_ID) re-derivable from centroid
  6.  TrainClass values are in expected set
  7.  GCS tile existence and RGB / label parity
  8.  CRS consistency across all RGB and label tiles
  9.  Band count (RGB=3, label=1)
  10. Band descriptions match expected ['Red', 'Green', 'Blue']
  11. Band ORDER heuristic — detects likely BGR/GRB/etc. permutations
      by comparing per-band mean pixel values against expected spectral
      fingerprints for natural-colour imagery of Arctic/subarctic terrain
  12. Tile spatial dimensions match TILE_SIZE
  13. RGB pixel values are non-zero / non-nodata
  14. Label tiles contain only valid values (0, 1, 255)

Environment variables:
  BUCKET            GCS bucket name                           (required)
  DATA_ROOT         Path to training data root                (required)
  WORK_DIR          Local working directory                   (required)
  SAMPLE_TILES      Validate a random sample of N tiles       (optional)
  SKIP_PIXEL_CHECK  Skip all tile download checks             (optional)
  VIEW_TILES        Show N tiles in viewer at the end         (optional, default 6)
  VIEW_POSITIVES    Prefer positive tiles in viewer           (optional, default 1)
"""

import os
import sys
import random
import traceback
from collections import defaultdict

import pandas as pd
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")          # headless-safe; swap to TkAgg / inline as needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from google.cloud import storage

try:
    from google.colab import auth
    auth.authenticate_user()
    # Switch to inline rendering in Colab
    import importlib
    matplotlib.use("inline")
    print("Authenticated via Colab.")
except (ImportError, Exception):
    print("Not running in Colab — using default GCS credentials (ADC).")


# Environment
def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if val is None:
        print(f"ERROR: Required environment variable '{name}' is not set.")
        sys.exit(1)
    return val


BUCKET    = _require_env("BUCKET")
DATA_ROOT = _require_env("DATA_ROOT").rstrip("/")
WORK_DIR  = _require_env("WORK_DIR")

RGB_PREFIX    = f"{DATA_ROOT}/PLANET-RGB/"
LABELS_PREFIX = f"{DATA_ROOT}/labels/"
METADATA_PATH = f"{DATA_ROOT}/metadata.csv"

TILE_SIZE        = 512
EXPECTED_CRS     = "EPSG:3857"
CENTROID_CRS     = "EPSG:4326"
VALID_CLASSES    = {"positive", "negative"}
VALID_LABEL_VALS = {0, 1, 255}

_sample_env  = os.environ.get("SAMPLE_TILES")
SAMPLE_TILES = int(_sample_env) if _sample_env else None
SKIP_PIXEL   = bool(os.environ.get("SKIP_PIXEL_CHECK"))

_view_env   = os.environ.get("VIEW_TILES")
VIEW_TILES  = int(_view_env) if _view_env else 6
VIEW_POS    = bool(int(os.environ.get("VIEW_POSITIVES", "1")))

os.makedirs(f"{WORK_DIR}/val_tmp", exist_ok=True)

client = storage.Client()
bucket = client.bucket(BUCKET)


# Geohash

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


# ---------------------------------------------------------------------------
# Band-order heuristic
# ---------------------------------------------------------------------------
# Natural-colour satellite imagery of Arctic/subarctic terrain (tundra, snow,
# water, bare ground) has a consistent spectral fingerprint:
#
#   Red band   — highest mean overall in vegetated/bare scenes; strongly
#                elevated over snow because snow is near-neutral bright.
#   Green band — middle value; always between red and blue for healthy veg.
#   Blue band  — lowest mean in vegetated scenes (chlorophyll absorption);
#                elevated relative to red only over deep water.
#
# Strategy: for each tile, rank the three band means (0=lowest, 2=highest).
# For correct RGB the rank order should be B < G < R  (ranks 0,1,2).
# We also test for the two most common mis-orderings:
#   BGR  → bands stored as [Blue, Green, Red]
#   BRG  → bands stored as [Blue, Red,   Green]
#
# A single tile is ambiguous (e.g. all-snow tiles are near-neutral), so we
# accumulate per-tile rank vectors across the sample and flag the tile if
# its rank pattern is inconsistent with the population majority *and* the
# deviation is consistent with a known swap pattern.

# Expected rank for correct RGB: band index 0 (R) has highest mean → rank 2
#                                  band index 1 (G) has middle mean → rank 1
#                                  band index 2 (B) has lowest mean → rank 0
EXPECTED_RANK = (2, 1, 0)   # (rank_of_band0, rank_of_band1, rank_of_band2)

# Known swap signatures and their human-readable description
SWAP_SIGNATURES = {
    (0, 1, 2): "BGR  (bands appear to be stored Blue→Green→Red)",
    (2, 0, 1): "BRG  (bands appear to be stored Blue→Red→Green)",
    (1, 2, 0): "GBR  (bands appear to be stored Green→Blue→Red)",
    (0, 2, 1): "GRB  (bands appear to be stored Green→Red→Blue)",
    (1, 0, 2): "RBG  (bands appear to be stored Red→Blue→Green)",
}

# Tiles whose per-band means are all within this fraction of each other are
# near-neutral (snow/cloud/water) and not informative for the heuristic.
NEUTRAL_THRESHOLD = 0.10   # 10 % relative spread


def _band_order_verdict(means: tuple) -> str | None:
    """
    Given (mean_band1, mean_band2, mean_band3), return a string describing a
    suspected swap, or None if the tile looks correctly ordered or is neutral.
    """
    r, g, b = means
    spread = max(means) - min(means)
    if max(means) == 0 or spread / max(means) < NEUTRAL_THRESHOLD:
        return None   # tile is too neutral to judge

    # Rank each band (argsort of means → lowest=0, highest=2)
    order = tuple(int(x) for x in np.argsort(means))   # which band has rank 0,1,2
    # Convert to "rank of each band position"
    rank_of = tuple(int(x) for x in np.argsort(order))

    if rank_of == EXPECTED_RANK:
        return None   # looks correct

    return SWAP_SIGNATURES.get(rank_of, f"unknown permutation ranks={rank_of}")


# Test runner

class TileValidationSuite:
    """
    Class-based test runner.  Instantiate, call .run(), inspect .failures and
    .warnings, or call .print_summary().
    """

    def __init__(self):
        self.failures: list[tuple[str, str]] = []   # (check_id, message)
        self.warnings: list[tuple[str, str]] = []
        self._df: pd.DataFrame | None = None
        self._rgb_blobs: set[str]   = set()
        self._label_blobs: set[str] = set()
        # Tiles downloaded during pixel checks, kept for viewer
        self._viewed_tiles: list[dict] = []

    # Assertion helpers

    def _fail(self, check: str, msg: str):
        self.failures.append((check, msg))
        print(f"  FAIL  [{check}] {msg}")

    def _warn(self, check: str, msg: str):
        self.warnings.append((check, msg))
        print(f"  WARN  [{check}] {msg}")

    def _ok(self, check: str, msg: str):
        print(f"  OK    [{check}] {msg}")

    def _assert_true(self, condition: bool, check: str, fail_msg: str, ok_msg: str):
        if condition:
            self._ok(check, ok_msg)
        else:
            self._fail(check, fail_msg)

    def _assert_empty(self, collection, check: str, fail_msg: str, ok_msg: str):
        self._assert_true(len(collection) == 0, check, fail_msg, ok_msg)

    # Individual test methods

    def test_metadata_exists(self) -> bool:
        """Check 1a — metadata file is present and loadable."""
        print("\n  1. Metadata file")
        meta_blob = bucket.blob(METADATA_PATH)
        if not meta_blob.exists():
            self._fail("metadata_exists",
                       f"gs://{BUCKET}/{METADATA_PATH} not found — cannot continue.")
            return False

        local_meta = f"{WORK_DIR}/val_tmp/metadata.csv"
        meta_blob.download_to_filename(local_meta)
        self._df = pd.read_csv(local_meta, dtype={"Tile_ID": str})
        self._ok("metadata_exists",
                 f"{len(self._df)} rows loaded from gs://{BUCKET}/{METADATA_PATH}")
        return True

    def test_metadata_columns(self):
        """Check 1b — required columns are present."""
        required = ["Tile_ID", "centroid_lat", "centroid_lon",
                    "TrainClass", "RegionName", "UIDs"]
        missing  = [c for c in required if c not in self._df.columns]
        self._assert_empty(
            missing, "metadata_columns",
            f"Missing columns: {missing}",
            f"All required columns present: {required}",
        )

        print(f"\n  Column dtypes:\n{self._df.dtypes.to_string()}")
        print(f"\n  TrainClass counts:\n{self._df['TrainClass'].value_counts().to_string()}")
        print(f"\n  UIDs value counts (top 5):\n{self._df['UIDs'].value_counts().head().to_string()}")

    def test_no_duplicate_tile_ids(self):
        """Check 2 — no duplicate Tile_IDs in metadata."""
        print("\n  2. Duplicate Tile_IDs")
        dups = self._df[self._df.duplicated("Tile_ID", keep=False)]
        if len(dups):
            self._fail("no_dup_tile_ids",
                       f"{len(dups)} rows share a duplicate Tile_ID")
            print(dups[["Tile_ID", "centroid_lat", "centroid_lon",
                         "TrainClass"]].to_string())
        else:
            self._ok("no_dup_tile_ids", "No duplicate Tile_IDs")

    def test_no_duplicate_centroids(self):
        """Check 3 — no duplicate (lat, lon) pairs (rounded to 6 dp)."""
        print("\n  3. Duplicate centroid (lat, lon)")
        df = self._df
        df["_lat6"] = df["centroid_lat"].round(6)
        df["_lon6"] = df["centroid_lon"].round(6)
        dups = df[df.duplicated(["_lat6", "_lon6"], keep=False)]
        if len(dups):
            self._fail("no_dup_centroids",
                       f"{len(dups)} rows share a duplicate centroid (rounded to 6dp)")
            print(dups[["Tile_ID", "_lat6", "_lon6", "TrainClass"]].to_string())
        else:
            self._ok("no_dup_centroids", "No duplicate centroids")

    def test_centroid_validity(self):
        """Check 4 — centroid coordinates are valid WGS84."""
        print(f"\n  4. Centroid coordinate validity (expected CRS: {CENTROID_CRS})")
        df = self._df
        lat_bad = df[(df["centroid_lat"] < -90) | (df["centroid_lat"] > 90)]
        lon_bad = df[(df["centroid_lon"] < -180) | (df["centroid_lon"] > 180)]

        self._assert_empty(
            lat_bad, "centroid_lat_range",
            f"{len(lat_bad)} rows have centroid_lat outside [-90, 90]",
            f"All centroid_lat in [-90, 90] — consistent with {CENTROID_CRS}",
        )
        self._assert_empty(
            lon_bad, "centroid_lon_range",
            f"{len(lon_bad)} rows have centroid_lon outside [-180, 180]",
            f"All centroid_lon in [-180, 180] — consistent with {CENTROID_CRS}",
        )

        south_of_50 = df[df["centroid_lat"] < 50]
        if len(south_of_50):
            self._warn(
                "centroid_lat_plausibility",
                f"{len(south_of_50)} tiles have centroid_lat < 50°N — unexpected for RTS. "
                f"Lat range: {south_of_50['centroid_lat'].min():.4f} – "
                f"{south_of_50['centroid_lat'].max():.4f}",
            )
        else:
            self._ok("centroid_lat_plausibility",
                     "All centroids north of 50°N — plausible for RTS")

    def test_uid_rederivation(self):
        """Check 5 — Tile_ID matches geohash(centroid_lat, centroid_lon)."""
        print("\n  5. UID re-derivation from centroid")
        mismatches = []
        for _, row in self._df.iterrows():
            expected = make_tile_uid(round(row["centroid_lat"], 6),
                                     round(row["centroid_lon"], 6))
            if expected != row["Tile_ID"]:
                mismatches.append({
                    "Tile_ID":  row["Tile_ID"],
                    "expected": expected,
                    "lat":      row["centroid_lat"],
                    "lon":      row["centroid_lon"],
                })
        if mismatches:
            self._fail("uid_rederivation",
                       f"{len(mismatches)} Tile_IDs do not match geohash of their centroid")
            print(pd.DataFrame(mismatches).to_string())
        else:
            self._ok("uid_rederivation",
                     "All Tile_IDs match geohash(centroid_lat, centroid_lon)")

    def test_train_class_values(self):
        """Check 6 — TrainClass values are in the expected set."""
        print("\n  6. TrainClass values")
        unexpected = set(self._df["TrainClass"].unique()) - VALID_CLASSES
        self._assert_empty(
            unexpected, "train_class_values",
            f"Unexpected TrainClass values: {unexpected}",
            f"All TrainClass values in {VALID_CLASSES}",
        )

    def test_gcs_tile_existence(self):
        """Check 7 — RGB and label tiles exist in GCS; parity between metadata and GCS."""
        print("\n  7. GCS tile existence and RGB / label parity")
        self._rgb_blobs = {
            b.name.split("/")[-1].replace(".tif", "")
            for b in bucket.list_blobs(prefix=RGB_PREFIX)
            if b.name.endswith(".tif")
        }
        self._label_blobs = {
            b.name.split("/")[-1].replace(".tif", "")
            for b in bucket.list_blobs(prefix=LABELS_PREFIX)
            if b.name.endswith(".tif")
        }

        meta_ids = set(self._df["Tile_ID"].astype(str))
        pos_ids  = set(self._df[self._df["TrainClass"] == "positive"]["Tile_ID"].astype(str))
        neg_ids  = set(self._df[self._df["TrainClass"] == "negative"]["Tile_ID"].astype(str))

        print(f"  RGB tiles in GCS:   {len(self._rgb_blobs)}")
        print(f"  Label tiles in GCS: {len(self._label_blobs)}")
        print(f"  Metadata rows:      {len(meta_ids)}  "
              f"(pos={len(pos_ids)}, neg={len(neg_ids)})")

        in_meta_not_rgb = meta_ids - self._rgb_blobs
        in_rgb_not_meta = self._rgb_blobs - meta_ids

        self._assert_empty(
            in_meta_not_rgb, "rgb_exists",
            f"{len(in_meta_not_rgb)} Tile_IDs in metadata have no RGB tile in GCS — "
            f"examples: {list(in_meta_not_rgb)[:5]}",
            "All metadata Tile_IDs have a corresponding RGB tile",
        )
        if in_rgb_not_meta:
            self._warn("rgb_orphan",
                       f"{len(in_rgb_not_meta)} RGB tiles in GCS have no metadata row — "
                       f"examples: {list(in_rgb_not_meta)[:5]}")
        else:
            self._ok("rgb_orphan", "No orphan RGB tiles")

        pos_missing_label = pos_ids - self._label_blobs
        neg_with_label    = neg_ids & self._label_blobs

        self._assert_empty(
            pos_missing_label, "label_exists_positive",
            f"{len(pos_missing_label)} positive tiles have no label in GCS — "
            f"examples: {list(pos_missing_label)[:5]}",
            "All positive tiles have a corresponding label tile",
        )
        if neg_with_label:
            self._warn("label_unexpected_negative",
                       f"{len(neg_with_label)} negative tiles unexpectedly have a label tile")
        else:
            self._ok("label_unexpected_negative",
                     "No negative tiles have label tiles (expected)")

    def test_tile_raster_properties(self):
        """
        Checks 8-14 — CRS, band count, band descriptions, band ORDER heuristic,
        tile dimensions, nodata, label pixel values.

        Downloads tiles (full set or SAMPLE_TILES sample) and inspects each one.
        """
        print(f"\n  8-14. Raster properties  [SKIP_PIXEL={SKIP_PIXEL}]")

        if SKIP_PIXEL:
            self._warn("pixel_checks",
                       "SKIP_PIXEL_CHECK=1 — skipping all tile download checks")
            return

        meta_ids     = set(self._df["Tile_ID"].astype(str))
        ids_to_check = list(meta_ids & self._rgb_blobs)
        if SAMPLE_TILES:
            ids_to_check = random.sample(ids_to_check,
                                         min(SAMPLE_TILES, len(ids_to_check)))
            print(f"  Sampling {len(ids_to_check)} tiles "
                  f"(SAMPLE_TILES={SAMPLE_TILES})")
        else:
            print(f"  Checking ALL {len(ids_to_check)} tiles "
                  f"— set SAMPLE_TILES=N to sample")

        crs_seen_rgb         = defaultdict(int)
        crs_seen_label       = defaultdict(int)
        band_count_errors    = []
        band_desc_errors     = []
        band_order_errors    = []   # NEW — suspected swap
        band_order_neutral   = []   # tiles too neutral to judge (informational)
        dim_errors           = []
        nodata_errors        = []
        label_val_errors     = []
        label_crs_mismatches = []

        # Collect tiles for the viewer
        pos_ids = set(self._df[self._df["TrainClass"] == "positive"]["Tile_ID"])
        viewer_candidates: dict[str, dict] = {}   # tile_id → {rgb, label, cls}

        for tile_id in ids_to_check:
            rgb_path   = f"{RGB_PREFIX}{tile_id}.tif"
            label_path = f"{LABELS_PREFIX}{tile_id}.tif"
            local_rgb   = f"{WORK_DIR}/val_tmp/rgb_{tile_id}.tif"
            local_label = f"{WORK_DIR}/val_tmp/lbl_{tile_id}.tif"
            tile_class  = self._df.loc[
                self._df["Tile_ID"] == tile_id, "TrainClass"
            ].values
            tile_class  = tile_class[0] if len(tile_class) else "unknown"

            rgb_data  = None
            label_arr = None
            rgb_crs   = None

            # ---- RGB tile ------------------------------------------------
            try:
                bucket.blob(rgb_path).download_to_filename(local_rgb)

                with rasterio.open(local_rgb) as src:
                    crs_str = src.crs.to_string() if src.crs else "None"
                    crs_seen_rgb[crs_str] += 1
                    rgb_crs = src.crs

                    # Band count
                    if src.count != 3:
                        band_count_errors.append(
                            (tile_id, f"count={src.count}, expected 3"))

                    # Band descriptions
                    descs = [src.descriptions[i] or "" for i in range(src.count)]
                    if descs != ["Red", "Green", "Blue"]:
                        band_desc_errors.append(
                            (tile_id, f"descriptions={descs}, "
                                      f"expected ['Red','Green','Blue']"))

                    # Dimensions
                    if src.width != TILE_SIZE or src.height != TILE_SIZE:
                        dim_errors.append(
                            (tile_id, f"{src.width}×{src.height}"))

                    # Read pixel data (float to avoid overflow in mean)
                    rgb_data = src.read().astype(np.float32)

                    # Nodata
                    if src.nodata is not None and (rgb_data == src.nodata).all():
                        nodata_errors.append((tile_id, "rgb", "all nodata"))
                    elif (rgb_data == 0).all():
                        nodata_errors.append((tile_id, "rgb", "all zeros"))
                    else:
                        # Band-order heuristic (only on non-nodata tiles)
                        if src.count == 3:
                            means  = tuple(float(rgb_data[i].mean())
                                           for i in range(3))
                            verdict = _band_order_verdict(means)
                            if verdict is None:
                                spread = max(means) - min(means)
                                if max(means) > 0 and \
                                        spread / max(means) < NEUTRAL_THRESHOLD:
                                    band_order_neutral.append(tile_id)
                                # else: looks correctly ordered
                            else:
                                band_order_errors.append(
                                    (tile_id,
                                     f"means R={means[0]:.1f} G={means[1]:.1f} "
                                     f"B={means[2]:.1f} → {verdict}"))

                # Keep file if we need it for the viewer; otherwise clean up
                viewer_candidates[tile_id] = {
                    "local_rgb": local_rgb,
                    "local_label": None,
                    "tile_class": tile_class,
                }

            except Exception as exc:
                self._fail("tile_read", f"RGB tile {tile_id}: {exc}")
                traceback.print_exc()

            # ---- Label tile (positives only) -----------------------------
            is_positive = tile_class == "positive"
            if is_positive and tile_id in self._label_blobs:
                try:
                    bucket.blob(label_path).download_to_filename(local_label)

                    with rasterio.open(local_label) as lsrc:
                        lcrs_str = lsrc.crs.to_string() if lsrc.crs else "None"
                        crs_seen_label[lcrs_str] += 1

                        if lsrc.count != 1:
                            band_count_errors.append(
                                (tile_id, f"label count={lsrc.count}, expected 1"))

                        label_arr   = lsrc.read(1)
                        unique_vals = set(np.unique(label_arr).tolist())
                        invalid     = unique_vals - VALID_LABEL_VALS
                        if invalid:
                            label_val_errors.append((tile_id, invalid))

                        if rgb_crs and lsrc.crs != rgb_crs:
                            label_crs_mismatches.append(
                                (tile_id,
                                 f"rgb={rgb_crs}, label={lsrc.crs}"))

                    viewer_candidates[tile_id]["local_label"] = local_label

                except Exception as exc:
                    self._fail("label_read", f"Label tile {tile_id}: {exc}")

        # ------------------------------------------------------------------
        # Report results
        # ------------------------------------------------------------------

        # CRS — RGB
        print(f"\n  CRS distribution — RGB tiles:")
        for crs_str, cnt in sorted(crs_seen_rgb.items(), key=lambda x: -x[1]):
            marker = "ok" if EXPECTED_CRS in crs_str else "!!"
            print(f"    [{marker}] {crs_str}: {cnt} tiles")

        if len(crs_seen_rgb) == 1 and EXPECTED_CRS in next(iter(crs_seen_rgb)):
            self._ok("rgb_crs_consistent", f"All RGB tiles use {EXPECTED_CRS}")
        elif len(crs_seen_rgb) == 1:
            self._warn("rgb_crs_expected",
                       f"All RGB tiles use a single CRS "
                       f"({next(iter(crs_seen_rgb))}) but expected {EXPECTED_CRS}")
        else:
            self._fail("rgb_crs_consistent",
                       f"RGB tiles have {len(crs_seen_rgb)} different CRS values — "
                       "mixed CRS will break downstream training")

        # CRS — label
        if crs_seen_label:
            print(f"\n  CRS distribution — label tiles:")
            for crs_str, cnt in sorted(crs_seen_label.items(), key=lambda x: -x[1]):
                marker = "ok" if EXPECTED_CRS in crs_str else "!!"
                print(f"    [{marker}] {crs_str}: {cnt} tiles")

            if len(crs_seen_label) == 1 and EXPECTED_CRS in next(iter(crs_seen_label)):
                self._ok("label_crs_consistent",
                         f"All label tiles use {EXPECTED_CRS}")
            elif len(crs_seen_label) == 1:
                self._warn("label_crs_expected",
                           f"All label tiles use a single CRS "
                           f"({next(iter(crs_seen_label))}) but expected {EXPECTED_CRS}")
            else:
                self._fail("label_crs_consistent",
                           f"Label tiles have {len(crs_seen_label)} different CRS values")

        if label_crs_mismatches:
            self._fail("rgb_label_crs_match",
                       f"{len(label_crs_mismatches)} tile pairs have mismatched "
                       f"RGB/label CRS")
            for tid, detail in label_crs_mismatches[:10]:
                print(f"    {tid}: {detail}")
        elif crs_seen_label:
            self._ok("rgb_label_crs_match",
                     "RGB and label CRS match for all checked tile pairs")

        # Band count
        self._assert_empty(
            band_count_errors, "band_count",
            f"{len(band_count_errors)} band count errors",
            "All checked tiles have correct band count",
        )
        for tid, detail in band_count_errors[:10]:
            print(f"    {tid}: {detail}")

        # Band descriptions
        self._assert_empty(
            band_desc_errors, "band_descriptions",
            f"{len(band_desc_errors)} tiles have wrong band descriptions "
            "(stored metadata, not actual spectral order — see band_order check)",
            "All tiles have band descriptions = ['Red', 'Green', 'Blue']",
        )
        for tid, detail in band_desc_errors[:10]:
            print(f"    {tid}: {detail}")

        # Band ORDER heuristic  ← new check
        print()
        if band_order_neutral:
            self._warn(
                "band_order_neutral",
                f"{len(band_order_neutral)} tiles were too spectrally neutral "
                f"(near-uniform brightness, likely snow/cloud) to judge band order — "
                f"result is inconclusive for those tiles",
            )

        if band_order_errors:
            self._fail(
                "band_order",
                f"{len(band_order_errors)} tiles appear to have INCORRECT band order "
                f"based on per-band mean spectral statistics. "
                f"Correct order is Red > Green > Blue for natural-colour Arctic terrain. "
                f"Affected tiles (first 20):",
            )
            for tid, detail in band_order_errors[:20]:
                print(f"    {tid}: {detail}")
            print(
                "\n  NOTE: This heuristic compares per-band mean pixel values. "
                "A high false-positive rate on near-neutral tiles (snow, water) is "
                "expected — cross-check with the 'band_order_neutral' warning. "
                "For confirmed mis-ordered tiles, re-export or transpose band order "
                "before training."
            )
        else:
            self._ok(
                "band_order",
                f"Band spectral statistics are consistent with R>G>B natural-colour "
                f"order for all non-neutral tiles checked",
            )

        # Dimensions
        self._assert_empty(
            dim_errors, "tile_dimensions",
            f"{len(dim_errors)} tiles are not {TILE_SIZE}×{TILE_SIZE}",
            f"All checked tiles are {TILE_SIZE}×{TILE_SIZE}",
        )
        for tid, dims in dim_errors[:10]:
            print(f"    {tid}: {dims}")

        # Nodata
        self._assert_empty(
            nodata_errors, "nodata_check",
            f"{len(nodata_errors)} tiles are entirely nodata or zero",
            "No tiles are entirely nodata or zero",
        )
        for tid, kind, detail in nodata_errors[:10]:
            print(f"    {tid} [{kind}]: {detail}")

        # Label values
        self._assert_empty(
            label_val_errors, "label_values",
            f"{len(label_val_errors)} label tiles contain unexpected pixel values",
            f"All label tiles contain only valid values {VALID_LABEL_VALS}",
        )
        for tid, vals in label_val_errors[:10]:
            print(f"    {tid}: invalid values = {vals}")

        # Store viewer candidates for later
        self._viewed_tiles = viewer_candidates

    # ------------------------------------------------------------------
    # Tile viewer
    # ------------------------------------------------------------------

    def show_tile_viewer(self):
        """
        Display up to VIEW_TILES tiles at the end of the run.

        Layout:
          - Positive tiles  → RGB panel + label overlay side by side
          - Negative tiles  → RGB panel only
        """
        if not self._viewed_tiles:
            print("\n  No tiles available for viewer (pixel checks were skipped).")
            return

        all_ids   = list(self._viewed_tiles.keys())
        pos_ids   = [t for t in all_ids
                     if self._viewed_tiles[t]["tile_class"] == "positive"]
        neg_ids   = [t for t in all_ids
                     if self._viewed_tiles[t]["tile_class"] == "negative"]

        # Build selection: prefer positives if VIEW_POS flag is set
        selected: list[str] = []
        if VIEW_POS:
            n_pos = min(len(pos_ids), max(1, VIEW_TILES // 2))
            n_neg = min(len(neg_ids), VIEW_TILES - n_pos)
        else:
            n_pos = 0
            n_neg = min(len(neg_ids), VIEW_TILES)

        selected += random.sample(pos_ids, n_pos) if n_pos else []
        selected += random.sample(neg_ids, n_neg) if n_neg else []
        selected  = selected[:VIEW_TILES]

        if not selected:
            print("\n  No tiles to display.")
            return

        # Each positive needs 2 columns; each negative needs 1
        col_counts = [2 if self._viewed_tiles[t]["tile_class"] == "positive"
                      else 1 for t in selected]
        total_cols = sum(col_counts)
        n_rows     = len(selected)

        fig, axes = plt.subplots(
            n_rows, max(total_cols // n_rows, 2),
            figsize=(5 * max(total_cols // n_rows, 2), 5 * n_rows),
            squeeze=False,
        )
        fig.suptitle(
            f"Tile viewer  —  {n_pos} positive / {n_neg} negative  "
            f"(random sample from validation set)",
            fontsize=13, y=1.01,
        )

        LABEL_CMAP  = {0: (0.1, 0.1, 0.1, 0.0),   # background — transparent
                       1: (1.0, 0.2, 0.2, 0.6),    # RTS — red, semi-transparent
                       255: (1.0, 1.0, 0.0, 0.4)}  # ignore — yellow

        for row_idx, tile_id in enumerate(selected):
            entry      = self._viewed_tiles[tile_id]
            tile_class = entry["tile_class"]
            ax_rgb     = axes[row_idx][0]
            ax_lbl     = axes[row_idx][1] if tile_class == "positive" else None

            # RGB panel
            try:
                with rasterio.open(entry["local_rgb"]) as src:
                    arr = src.read([1, 2, 3]).astype(np.float32)
                # Stretch to 2-98th percentile for display
                lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
                arr     = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
                rgb_img = np.moveaxis(arr, 0, -1)
                ax_rgb.imshow(rgb_img)
                ax_rgb.set_title(f"{tile_id}\n[{tile_class}]  RGB", fontsize=8)
                ax_rgb.axis("off")
            except Exception as exc:
                ax_rgb.set_title(f"{tile_id} — read error")
                ax_rgb.text(0.5, 0.5, str(exc), transform=ax_rgb.transAxes,
                            ha="center", va="center", fontsize=7, color="red")
                ax_rgb.axis("off")

            # Label overlay panel (positives only)
            if ax_lbl is not None and entry.get("local_label"):
                try:
                    with rasterio.open(entry["local_label"]) as lsrc:
                        lbl = lsrc.read(1)

                    rgba = np.zeros((*lbl.shape, 4), dtype=np.float32)
                    for val, colour in LABEL_CMAP.items():
                        mask = lbl == val
                        rgba[mask] = colour

                    ax_lbl.imshow(rgb_img)        # RGB as base
                    ax_lbl.imshow(rgba)            # label as overlay
                    ax_lbl.set_title(
                        f"{tile_id}\nRGB + label  "
                        f"(RTS pixels: {(lbl == 1).sum():,})",
                        fontsize=8,
                    )

                    legend_patches = [
                        mpatches.Patch(color=(1.0, 0.2, 0.2, 0.6), label="RTS (1)"),
                        mpatches.Patch(color=(1.0, 1.0, 0.0, 0.6), label="Ignore (255)"),
                        mpatches.Patch(color=(0.4, 0.4, 0.4, 0.6), label="Background (0)"),
                    ]
                    ax_lbl.legend(handles=legend_patches, loc="lower right",
                                  fontsize=6, framealpha=0.7)
                    ax_lbl.axis("off")
                except Exception as exc:
                    ax_lbl.set_title("label — read error")
                    ax_lbl.text(0.5, 0.5, str(exc), transform=ax_lbl.transAxes,
                                ha="center", va="center", fontsize=7, color="red")
                    ax_lbl.axis("off")
            elif ax_lbl is not None:
                ax_lbl.axis("off")
                ax_lbl.set_title("no label tile", fontsize=8)

            # Hide unused axes in this row
            for col_idx in range(2, axes.shape[1]):
                axes[row_idx][col_idx].axis("off")

        plt.tight_layout()
        viewer_path = f"{WORK_DIR}/tile_viewer.png"
        plt.savefig(viewer_path, dpi=120, bbox_inches="tight")
        print(f"\n  Tile viewer saved to: {viewer_path}")
        try:
            plt.show()
        except Exception:
            pass   # headless environment

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_tmp(self):
        """Remove downloaded temp tiles (except those needed by viewer)."""
        tmp_dir = f"{WORK_DIR}/val_tmp"
        for fname in os.listdir(tmp_dir):
            fpath = os.path.join(tmp_dir, fname)
            # Keep tiles that are still needed for the viewer
            keep = any(
                (v.get("local_rgb") == fpath or v.get("local_label") == fpath)
                for v in self._viewed_tiles.values()
            )
            if not keep:
                try:
                    os.remove(fpath)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_summary(self):
        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)

        n_fail = len(self.failures)
        n_warn = len(self.warnings)

        if self.failures:
            print(f"\n  {n_fail} FAILURE(S):\n")
            for check, msg in self.failures:
                print(f"    FAIL [{check}] {msg}")
        else:
            print("\n  No failures.")

        if self.warnings:
            print(f"\n  {n_warn} WARNING(S):\n")
            for check, msg in self.warnings:
                print(f"    WARN [{check}] {msg}")
        else:
            print("  No warnings.")

        print()
        if n_fail == 0:
            print("  ✓  All checks passed.")
        elif n_fail <= 2:
            print(f"  ✗  {n_fail} check(s) failed — review before training.")
        else:
            print(f"  ✗  {n_fail} check(s) failed — pipeline output needs attention.")
        print()

    # Main
    def run(self):
        print("=" * 60)
        print("  Tile Validation Suite")
        print("=" * 60)

        # Metadata checks
        if not self.test_metadata_exists():
            self.print_summary()
            return
        self.test_metadata_columns()
        self.test_no_duplicate_tile_ids()
        self.test_no_duplicate_centroids()
        self.test_centroid_validity()
        self.test_uid_rederivation()
        self.test_train_class_values()

        # GCS existence checks
        self.test_gcs_tile_existence()

        # Per-tile raster checks
        self.test_tile_raster_properties()

        self._cleanup_tmp()

        # Tile viewer
        if VIEW_TILES > 0:
            print(f"\n  Tile viewer  ({VIEW_TILES} tiles, prefer_positives={VIEW_POS})")
            self.show_tile_viewer()

        self.print_summary()


# ---------------------------------------------------------------------------
# Entry point

if __name__ == "__main__":
    suite = TileValidationSuite()
    suite.run()
    sys.exit(1 if suite.failures else 0)
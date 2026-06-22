"""
Tile validation suite for positive_tile_creation.py / negative_tile_creation.py output.

Run directly:  python validate_tiles.py

Checks are grouped:

  Metadata checks (all tiles)
    - metadata file exists / required columns
    - duplicate Tile_IDs
    - duplicate centroid (lat, lon) pairs
    - centroid coordinate validity (WGS84 range + plausibility)
    - UID (Tile_ID) re-derivable from centroid
    - TrainClass values in expected set

  GCS existence checks (split positive / negative)
    - every metadata Tile_ID has an RGB tile in GCS
    - positive tiles have a matching label tile
    - negative tiles do NOT have a label tile (warn if they do)
    - orphan tiles in GCS with no metadata row

  Corruption scan (optional, full or sampled)
    - every RGB/label tile can be opened by rasterio
    - tile dimensions match TILE_SIZE
    - band count correct (RGB=3, label=1)
    - tile is not entirely nodata / all-zero
    - label tiles contain only valid values (0, 1, 255)
    - CRS consistency across all checked tiles

  Tile viewer
    - shows a sample of positive and negative tiles (RGB + label overlay)

Environment variables:
  BUCKET            GCS bucket name                           (required)
  DATA_ROOT         Path to training data root                (required)
  WORK_DIR          Local working directory                   (required)
  SAMPLE_TILES      Validate a random sample of N tiles per class (optional, default = all)
  SKIP_PIXEL_CHECK  Skip the corruption scan entirely         (optional)
  VIEW_TILES        Show N tiles in viewer at the end         (optional, default 6)
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from google.cloud import storage

try:
    from google.colab import auth
    auth.authenticate_user()
    matplotlib.use("inline")
    print("Authenticated via Colab.")
except (ImportError, Exception):
    print("Not running in Colab — using default GCS credentials (ADC).")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

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

_view_env  = os.environ.get("VIEW_TILES")
VIEW_TILES = int(_view_env) if _view_env else 6

os.makedirs(f"{WORK_DIR}/val_tmp", exist_ok=True)

client = storage.Client()
bucket = client.bucket(BUCKET)


# ---------------------------------------------------------------------------
# Geohash
# ---------------------------------------------------------------------------

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
# Test runner
# ---------------------------------------------------------------------------

class TileValidationSuite:
    """
    Class-based test runner.  Instantiate, call .run(), inspect .failures and
    .warnings, or call .print_summary().
    """

    def __init__(self):
        self.failures: list[tuple[str, str]] = []
        self.warnings: list[tuple[str, str]] = []
        self._df: pd.DataFrame | None = None
        self._rgb_blobs: set[str]   = set()
        self._label_blobs: set[str] = set()
        self._viewed_tiles: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Assertion helpers
    # ------------------------------------------------------------------

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

    # ==================================================================
    # METADATA CHECKS  (all tiles)
    # ==================================================================

    def test_metadata_exists(self) -> bool:
        """Metadata file is present and loadable."""
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
        """Required columns are present."""
        required = ["Tile_ID", "centroid_lat", "centroid_lon",
                    "TrainClass", "RegionName", "UIDs"]
        missing  = [c for c in required if c not in self._df.columns]
        self._assert_empty(
            missing, "metadata_columns",
            f"Missing columns: {missing}",
            f"All required columns present: {required}",
        )

        print(f"\n  TrainClass counts:\n{self._df['TrainClass'].value_counts().to_string()}")

    def test_no_duplicate_tile_ids(self):
        """No duplicate Tile_IDs in metadata."""
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
        """No duplicate (lat, lon) pairs (rounded to 6 dp)."""
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
        """Centroid coordinates are valid WGS84 and plausible for RTS work."""
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
        """Tile_ID matches geohash(centroid_lat, centroid_lon)."""
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
        """TrainClass values are in the expected set."""
        print("\n  6. TrainClass values")
        unexpected = set(self._df["TrainClass"].unique()) - VALID_CLASSES
        self._assert_empty(
            unexpected, "train_class_values",
            f"Unexpected TrainClass values: {unexpected}",
            f"All TrainClass values in {VALID_CLASSES}",
        )

    # ==================================================================
    # GCS EXISTENCE CHECKS  (split positive / negative)
    # ==================================================================

    def test_gcs_tile_existence(self):
        """RGB and label tiles exist in GCS; parity between metadata and GCS,
        evaluated separately for positive and negative tiles."""
        print("\n  7. GCS tile existence")
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

        # --- RGB existence, split by class ---
        for cls_name, ids in (("positive", pos_ids), ("negative", neg_ids)):
            missing = ids - self._rgb_blobs
            self._assert_empty(
                missing, f"rgb_exists_{cls_name}",
                f"{len(missing)} {cls_name} Tile_IDs have no RGB tile in GCS — "
                f"examples: {list(missing)[:5]}",
                f"All {cls_name} Tile_IDs ({len(ids)}) have a corresponding RGB tile",
            )

        # --- orphan RGB tiles (no metadata row) ---
        in_rgb_not_meta = self._rgb_blobs - meta_ids
        if in_rgb_not_meta:
            self._warn("rgb_orphan",
                       f"{len(in_rgb_not_meta)} RGB tiles in GCS have no metadata row — "
                       f"examples: {list(in_rgb_not_meta)[:5]}")
        else:
            self._ok("rgb_orphan", "No orphan RGB tiles")

        # --- label existence: positives must have one, negatives must not ---
        pos_missing_label = pos_ids - self._label_blobs
        self._assert_empty(
            pos_missing_label, "label_exists_positive",
            f"{len(pos_missing_label)} positive tiles have no label in GCS — "
            f"examples: {list(pos_missing_label)[:5]}",
            f"All positive tiles ({len(pos_ids)}) have a corresponding label tile",
        )

        neg_with_label = neg_ids & self._label_blobs
        if neg_with_label:
            self._warn("label_unexpected_negative",
                       f"{len(neg_with_label)} negative tiles unexpectedly have a label tile — "
                       f"examples: {list(neg_with_label)[:5]}")
        else:
            self._ok("label_unexpected_negative",
                     "No negative tiles have label tiles (expected)")

        # --- orphan label tiles ---
        in_label_not_pos = self._label_blobs - pos_ids
        if in_label_not_pos:
            self._warn("label_orphan",
                       f"{len(in_label_not_pos)} label tiles in GCS don't correspond to a "
                       f"positive metadata row — examples: {list(in_label_not_pos)[:5]}")
        else:
            self._ok("label_orphan", "No orphan label tiles")

    # ==================================================================
    # CORRUPTION SCAN  (full or sampled, split positive / negative)
    # ==================================================================

    def _scan_one_tile(self, tile_id, path, expect_bands, kind, crs_seen, errors):
        """
        Open one raster and run the basic integrity checks.
        kind: 'rgb' or 'label'
        Returns the opened array (or None on failure).
        """
        local_path = f"{WORK_DIR}/val_tmp/{kind}_{tile_id}.tif"
        try:
            bucket.blob(path).download_to_filename(local_path)
            with rasterio.open(local_path) as src:
                crs_str = src.crs.to_string() if src.crs else "None"
                crs_seen[crs_str] += 1

                if src.count != expect_bands:
                    errors["band_count"].append(
                        (tile_id, f"{kind} count={src.count}, expected {expect_bands}"))

                if src.width != TILE_SIZE or src.height != TILE_SIZE:
                    errors["dimensions"].append(
                        (tile_id, f"{kind} {src.width}×{src.height}"))

                arr = src.read()

                if src.nodata is not None and (arr == src.nodata).all():
                    errors["nodata"].append((tile_id, kind, "all nodata"))
                elif (arr == 0).all():
                    errors["nodata"].append((tile_id, kind, "all zeros"))

                if kind == "label":
                    unique_vals = set(np.unique(arr).tolist())
                    invalid = unique_vals - VALID_LABEL_VALS
                    if invalid:
                        errors["label_values"].append((tile_id, invalid))

            return local_path
        except Exception as exc:
            errors["read_errors"].append((tile_id, kind, str(exc)))
            return None

    def test_corruption_scan(self):
        """
        Open every (or a sample of) RGB/label tile with rasterio and check:
          - file opens without error
          - dimensions == TILE_SIZE
          - band count correct
          - not entirely nodata/zero
          - label values valid
          - CRS consistent across tiles

        Run separately for positive and negative tiles.
        """
        print(f"\n  8. Corruption scan  [SKIP_PIXEL_CHECK={SKIP_PIXEL}]")

        if SKIP_PIXEL:
            self._warn("corruption_scan",
                       "SKIP_PIXEL_CHECK=1 — skipping corruption scan")
            return

        meta_ids = set(self._df["Tile_ID"].astype(str))
        pos_ids  = sorted(meta_ids & self._rgb_blobs &
                          set(self._df[self._df["TrainClass"] == "positive"]["Tile_ID"].astype(str)))
        neg_ids  = sorted(meta_ids & self._rgb_blobs &
                          set(self._df[self._df["TrainClass"] == "negative"]["Tile_ID"].astype(str)))

        for cls_name, ids in (("positive", pos_ids), ("negative", neg_ids)):
            ids_to_check = ids
            if SAMPLE_TILES:
                ids_to_check = random.sample(ids, min(SAMPLE_TILES, len(ids)))
                print(f"\n  -- {cls_name}: sampling {len(ids_to_check)} of {len(ids)} tiles "
                      f"(SAMPLE_TILES={SAMPLE_TILES})")
            else:
                print(f"\n  -- {cls_name}: checking ALL {len(ids_to_check)} tiles "
                      f"— set SAMPLE_TILES=N to sample")

            crs_seen_rgb   = defaultdict(int)
            crs_seen_label = defaultdict(int)
            errors = defaultdict(list)

            for tile_id in ids_to_check:
                local_rgb = self._scan_one_tile(
                    tile_id, f"{RGB_PREFIX}{tile_id}.tif",
                    expect_bands=3, kind="rgb",
                    crs_seen=crs_seen_rgb, errors=errors,
                )

                local_label = None
                if cls_name == "positive" and tile_id in self._label_blobs:
                    local_label = self._scan_one_tile(
                        tile_id, f"{LABELS_PREFIX}{tile_id}.tif",
                        expect_bands=1, kind="label",
                        crs_seen=crs_seen_label, errors=errors,
                    )

                # keep for viewer
                self._viewed_tiles[tile_id] = {
                    "local_rgb": local_rgb,
                    "local_label": local_label,
                    "tile_class": cls_name,
                }

            # ---- report this class's results ----
            prefix = f"{cls_name}"

            self._assert_empty(
                errors["read_errors"], f"{prefix}_readable",
                f"{len(errors['read_errors'])} {cls_name} tiles failed to open "
                f"(possibly corrupted)",
                f"All checked {cls_name} tiles opened successfully",
            )
            for tid, kind, detail in errors["read_errors"][:10]:
                print(f"    {tid} [{kind}]: {detail}")

            self._assert_empty(
                errors["band_count"], f"{prefix}_band_count",
                f"{len(errors['band_count'])} {cls_name} tiles have wrong band count",
                f"All checked {cls_name} tiles have correct band count",
            )
            for tid, detail in errors["band_count"][:10]:
                print(f"    {tid}: {detail}")

            self._assert_empty(
                errors["dimensions"], f"{prefix}_dimensions",
                f"{len(errors['dimensions'])} {cls_name} tiles are not {TILE_SIZE}×{TILE_SIZE}",
                f"All checked {cls_name} tiles are {TILE_SIZE}×{TILE_SIZE}",
            )
            for tid, detail in errors["dimensions"][:10]:
                print(f"    {tid}: {detail}")

            self._assert_empty(
                errors["nodata"], f"{prefix}_nodata",
                f"{len(errors['nodata'])} {cls_name} tiles are entirely nodata or zero",
                f"No {cls_name} tiles are entirely nodata or zero",
            )
            for tid, kind, detail in errors["nodata"][:10]:
                print(f"    {tid} [{kind}]: {detail}")

            if cls_name == "positive":
                self._assert_empty(
                    errors["label_values"], f"{prefix}_label_values",
                    f"{len(errors['label_values'])} label tiles contain unexpected pixel values",
                    f"All checked label tiles contain only valid values {VALID_LABEL_VALS}",
                )
                for tid, vals in errors["label_values"][:10]:
                    print(f"    {tid}: invalid values = {vals}")

            # CRS consistency (RGB)
            print(f"\n  CRS distribution — {cls_name} RGB tiles:")
            for crs_str, cnt in sorted(crs_seen_rgb.items(), key=lambda x: -x[1]):
                marker = "ok" if EXPECTED_CRS in crs_str else "!!"
                print(f"    [{marker}] {crs_str}: {cnt} tiles")

            if len(crs_seen_rgb) == 1 and EXPECTED_CRS in next(iter(crs_seen_rgb)):
                self._ok(f"{prefix}_rgb_crs_consistent",
                         f"All {cls_name} RGB tiles use {EXPECTED_CRS}")
            elif len(crs_seen_rgb) == 1:
                self._warn(f"{prefix}_rgb_crs_expected",
                           f"All {cls_name} RGB tiles use a single CRS "
                           f"({next(iter(crs_seen_rgb))}) but expected {EXPECTED_CRS}")
            elif len(crs_seen_rgb) > 1:
                self._fail(f"{prefix}_rgb_crs_consistent",
                           f"{cls_name} RGB tiles have {len(crs_seen_rgb)} different "
                           f"CRS values — mixed CRS will break downstream training")

            # CRS consistency (label)
            if crs_seen_label:
                print(f"\n  CRS distribution — {cls_name} label tiles:")
                for crs_str, cnt in sorted(crs_seen_label.items(), key=lambda x: -x[1]):
                    marker = "ok" if EXPECTED_CRS in crs_str else "!!"
                    print(f"    [{marker}] {crs_str}: {cnt} tiles")

                if len(crs_seen_label) == 1 and EXPECTED_CRS in next(iter(crs_seen_label)):
                    self._ok(f"{prefix}_label_crs_consistent",
                             f"All {cls_name} label tiles use {EXPECTED_CRS}")
                elif len(crs_seen_label) == 1:
                    self._warn(f"{prefix}_label_crs_expected",
                               f"All {cls_name} label tiles use a single CRS "
                               f"({next(iter(crs_seen_label))}) but expected {EXPECTED_CRS}")
                else:
                    self._fail(f"{prefix}_label_crs_consistent",
                               f"{cls_name} label tiles have {len(crs_seen_label)} "
                               f"different CRS values")

    # ==================================================================
    # TILE VIEWER
    # ==================================================================

    def show_tile_viewer(self):
        """
        Display up to VIEW_TILES tiles, split between positive and negative.

          - Positive tiles → RGB panel + label overlay side by side
          - Negative tiles → RGB panel only
        """
        usable = {
            tid: v for tid, v in self._viewed_tiles.items()
            if v.get("local_rgb") is not None
        }
        if not usable:
            print("\n  No tiles available for viewer (corruption scan was skipped "
                  "or all tiles failed to open).")
            return

        pos_ids = [t for t, v in usable.items() if v["tile_class"] == "positive"]
        neg_ids = [t for t, v in usable.items() if v["tile_class"] == "negative"]

        n_pos = min(len(pos_ids), max(1, VIEW_TILES // 2)) if pos_ids else 0
        n_neg = min(len(neg_ids), VIEW_TILES - n_pos)

        selected = (random.sample(pos_ids, n_pos) if n_pos else []) + \
                   (random.sample(neg_ids, n_neg) if n_neg else [])
        selected = selected[:VIEW_TILES]

        if not selected:
            print("\n  No tiles to display.")
            return

        n_rows  = len(selected)
        n_cols  = 2  # RGB + label overlay (label panel hidden for negatives)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5 * n_cols, 5 * n_rows),
            squeeze=False,
        )
        fig.suptitle(
            f"Tile viewer — {n_pos} positive / {n_neg} negative "
            f"(random sample from validation set)",
            fontsize=13, y=1.01,
        )

        LABEL_CMAP = {0: (0.1, 0.1, 0.1, 0.0),
                      1: (1.0, 0.2, 0.2, 0.6),
                      255: (1.0, 1.0, 0.0, 0.4)}

        for row_idx, tile_id in enumerate(selected):
            entry      = usable[tile_id]
            tile_class = entry["tile_class"]
            ax_rgb     = axes[row_idx][0]
            ax_lbl     = axes[row_idx][1]

            rgb_img = None
            try:
                with rasterio.open(entry["local_rgb"]) as src:
                    arr = src.read([1, 2, 3]).astype(np.float32)
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

            if tile_class == "positive" and entry.get("local_label") and rgb_img is not None:
                try:
                    with rasterio.open(entry["local_label"]) as lsrc:
                        lbl = lsrc.read(1)

                    rgba = np.zeros((*lbl.shape, 4), dtype=np.float32)
                    for val, colour in LABEL_CMAP.items():
                        rgba[lbl == val] = colour

                    ax_lbl.imshow(rgb_img)
                    ax_lbl.imshow(rgba)
                    ax_lbl.set_title(
                        f"{tile_id}\nRGB + label  (RTS pixels: {(lbl == 1).sum():,})",
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
            else:
                ax_lbl.axis("off")
                ax_lbl.set_title("no label tile", fontsize=8)

        plt.tight_layout()
        viewer_path = f"{WORK_DIR}/tile_viewer.png"
        plt.savefig(viewer_path, dpi=120, bbox_inches="tight")
        print(f"\n  Tile viewer saved to: {viewer_path}")
        try:
            plt.show()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_tmp(self):
        """Remove downloaded temp tiles (except those needed by viewer)."""
        tmp_dir = f"{WORK_DIR}/val_tmp"
        keep_paths = set()
        for v in self._viewed_tiles.values():
            if v.get("local_rgb"):
                keep_paths.add(v["local_rgb"])
            if v.get("local_label"):
                keep_paths.add(v["local_label"])

        for fname in os.listdir(tmp_dir):
            fpath = os.path.join(tmp_dir, fname)
            if fpath not in keep_paths and fname != "metadata.csv":
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
            print("  All checks passed.")
        else:
            print(f"  {n_fail} check(s) failed — review before training.")
        print()

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def run(self):
        print("=" * 60)
        print("  Tile Validation Suite")
        print("=" * 60)

        if not self.test_metadata_exists():
            self.print_summary()
            return

        self.test_metadata_columns()
        self.test_no_duplicate_tile_ids()
        self.test_no_duplicate_centroids()
        self.test_centroid_validity()
        self.test_uid_rederivation()
        self.test_train_class_values()

        self.test_gcs_tile_existence()

        self.test_corruption_scan()

        self._cleanup_tmp()

        if VIEW_TILES > 0:
            print(f"\n  Tile viewer ({VIEW_TILES} tiles)")
            self.show_tile_viewer()

        self.print_summary()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    suite = TileValidationSuite()
    suite.run()
    sys.exit(1 if suite.failures else 0)
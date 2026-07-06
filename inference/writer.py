"""Probability-tile COG writing + inference_log.json manifest (inference.md §8.3, §9).

Outputs follow §9.1: Float32 COG (NoData -1.0) or the scaled_uint8 encoding
(prob×250, NoData 255 — ~71× smaller), EPSG:3857, deflate. GCS writes
go through a local temp file then a single upload (GCS object creation is
atomic, so a crashed upload never leaves a half-written object). The manifest
records completed/skipped tiles so a restarted job skips finished work.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds as transform_from_bounds

logger = logging.getLogger(__name__)

NODATA_PROB = -1.0  # §9.1 sentinel
NODATA_MASK = 255   # §9.2

# scaled_uint8 probability encoding (§9.1 alt): prob×250 → uint8 [0,250],
# NoData = 255. ~8 KB/tile vs ~570 KB Float32 (71×), re-threshold precision
# 1/250 = 0.004. Values 251–254 are unused. Decode: value/250, 255→NoData.
SCALE_U8 = 250
NODATA_SCALED_U8 = 255

_COG_PROFILE = dict(
    driver="GTiff", compress="deflate", tiled=True,
    blockxsize=256, blockysize=256, crs="EPSG:3857",
)


def _write_raster(path: str, array: np.ndarray, bounds: tuple, dtype: str,
                  nodata: float | int) -> None:
    """Write a single-band georeferenced raster to local path or gs:// URI."""
    h, w = array.shape
    transform = transform_from_bounds(*bounds, w, h)
    profile = dict(_COG_PROFILE, height=h, width=w, count=1, dtype=dtype,
                   nodata=nodata, transform=transform)
    if str(path).startswith("gs://"):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with rasterio.open(tmp_path, "w", **profile) as dst:
                dst.write(array.astype(dtype), 1)
            from google.cloud import storage
            bucket_name, blob_name = str(path)[5:].split("/", 1)
            storage.Client().bucket(bucket_name).blob(blob_name).upload_from_filename(tmp_path)
        finally:
            os.unlink(tmp_path)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp_path = f"{path}.tmp"
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(array.astype(dtype), 1)
        os.replace(tmp_path, path)  # atomic on the same filesystem


def _encode_scaled_uint8(probs: np.ndarray) -> np.ndarray:
    """Encode a float prob array (NoData -1.0) → uint8 (prob×250, NoData 255)."""
    nodata = probs == NODATA_PROB
    out = np.clip(np.round(probs * SCALE_U8), 0, SCALE_U8).astype(np.uint8)
    out[nodata] = NODATA_SCALED_U8
    return out


def write_probability_tile(path: str, probs: np.ndarray, bounds: tuple,
                           dtype: str = "float32") -> None:
    """Write a §9.1 probability tile.

    dtype="float32" (default) → NoData -1.0; dtype="scaled_uint8" → prob×250
    uint8, NoData 255 (~71× smaller, 0.004 precision). Read back with
    :func:`read_probability_tile`, which auto-detects the on-disk encoding.
    """
    if dtype == "scaled_uint8":
        _write_raster(path, _encode_scaled_uint8(probs), bounds, "uint8",
                      NODATA_SCALED_U8)
    elif dtype == "float32":
        _write_raster(path, probs, bounds, "float32", NODATA_PROB)
    else:
        raise ValueError(f"unknown output dtype {dtype!r} "
                         "(expected 'float32' or 'scaled_uint8')")


def read_probability_tile(path: str) -> np.ndarray:
    """Read a probability COG → float32 with the NODATA_PROB (-1.0) sentinel.

    Auto-detects the on-disk encoding: uint8 rasters are decoded (value/250,
    255→NoData); float32 rasters are returned as-is. Raises the underlying
    ``rasterio`` error if the file is missing (callers treat that as a skipped
    tile). Keeps the encode↔decode contract in one place (Rule 3 / SSoT).
    """
    with rasterio.open(path) as src:
        arr = src.read(1)
        if src.dtypes[0] == "uint8":
            out = (arr.astype(np.float32) / SCALE_U8)
            out[arr == NODATA_SCALED_U8] = NODATA_PROB
            return out
        return arr.astype(np.float32)


def write_binary_mask(path: str, mask: np.ndarray, bounds: tuple) -> None:
    """Write a §9.2 binary mask (uint8 0/1, NoData 255)."""
    _write_raster(path, mask, bounds, "uint8", NODATA_MASK)


class Manifest:
    """inference_log.json — progress manifest + run metadata (§8.3, §9.4)."""

    def __init__(self, path: str, run_metadata: dict, checkpoint_every: int = 100):
        self.path = path
        self.checkpoint_every = checkpoint_every
        self.completed: dict[str, str] = {}   # tile_id -> "done" | skip reason
        self.metadata = dict(run_metadata)
        self._since_save = 0
        self._t0 = time.time()
        existing = self._load_existing()
        if existing:
            self.completed = existing.get("tiles", {})
            logger.info("Manifest resume: %d tiles already recorded in %s",
                        len(self.completed), path)

    def _load_existing(self) -> dict | None:
        try:
            if str(self.path).startswith("gs://"):
                import gcsfs
                fs = gcsfs.GCSFileSystem(token="google_default")
                if not fs.exists(self.path[5:]):
                    return None
                with fs.open(self.path[5:], "r") as f:
                    return json.load(f)
            p = Path(self.path)
            return json.loads(p.read_text()) if p.exists() else None
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load existing manifest %s: %s", self.path, exc)
            return None

    def is_done(self, tile_id: str) -> bool:
        return tile_id in self.completed

    def mark(self, tile_id: str, status: str = "done") -> None:
        self.completed[tile_id] = status
        self._since_save += 1
        if self._since_save >= self.checkpoint_every:
            self.save()

    def counts(self) -> dict[str, int]:
        n_done = sum(1 for v in self.completed.values() if v == "done")
        return {
            "n_tiles_processed": n_done,
            "n_tiles_skipped_nodata": sum(1 for v in self.completed.values()
                                          if v == "all_nodata"),
        }

    def save(self) -> None:
        payload = {
            **self.metadata,
            **self.counts(),
            "processing_time_hours": round((time.time() - self._t0) / 3600, 4),
            "tiles": self.completed,
        }
        text = json.dumps(payload, indent=1)
        if str(self.path).startswith("gs://"):
            import gcsfs
            fs = gcsfs.GCSFileSystem(token="google_default")
            with fs.open(self.path[5:], "w") as f:
                f.write(text)
        else:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            tmp = f"{self.path}.tmp"
            Path(tmp).write_text(text)
            os.replace(tmp, self.path)
        self._since_save = 0

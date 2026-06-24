"""Windowed 512x512 tile reads from basemap quads + training normalization.

Tiles are projected-grid windows (inference.md §4.1) that may straddle quad
boundaries; read_tile mosaics every intersecting quad into one array. NoData
follows inference.md §5.3: the quad alpha band marks NoData (alpha == 0), as do
pixels not covered by any indexed quad.

Normalization reuses data/normalization.py stats (CLAUDE.md Rule 3): NoData
pixels get the per-channel training mean *before* z-scoring (matching training,
training.md §4.4), and the caller masks them out of the prediction afterwards.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from torch.utils.data import Dataset

from data.normalization import apply_norm, build_norm_arrays, fill_nodata_with_mean

logger = logging.getLogger(__name__)

TILE_SIZE_PX = 512  # CLAUDE.md technical constraint; matches training tiles

# Band indices (1-based) in the bulk S2 composite COGs — export order B4,B3,B2,B8
# (scripts/export_s2_composites.py DEFAULT_BANDS). Red=B4, NIR=B8 → NDVI.
S2_RED_BAND = 1   # B4
S2_NIR_BAND = 4   # B8

# Transient-GCS-read retry (same rationale as data/dataset.py).
_READ_ATTEMPTS = 4
_RETRY_BASE_DELAY_S = 1.0


def _read_window_with_retry(
    path: str,
    bounds: tuple[float, float, float, float],
    out_size: int | None = None,
) -> np.ndarray:
    """Read all bands of `path` within `bounds`, boundless with 0-fill, retried.

    With `out_size`, the window is decimated/resampled to (out_size, out_size):
    RGB bands bilinear, alpha band (if present) nearest — so the NoData mask
    stays crisp instead of blending validity at coverage edges.
    """
    last_exc: Exception | None = None
    for attempt in range(_READ_ATTEMPTS):
        try:
            with rasterio.open(path) as src:
                window = from_bounds(*bounds, transform=src.transform)
                if out_size is None:
                    return src.read(window=window, boundless=True, fill_value=0)
                from rasterio.enums import Resampling
                rgb = src.read(indexes=list(range(1, min(src.count, 3) + 1)),
                               window=window, boundless=True, fill_value=0,
                               out_shape=(min(src.count, 3), out_size, out_size),
                               resampling=Resampling.bilinear)
                if src.count >= 4:
                    alpha = src.read(indexes=[4], window=window, boundless=True,
                                     fill_value=0, out_shape=(1, out_size, out_size),
                                     resampling=Resampling.nearest)
                    return np.concatenate([rgb, alpha], axis=0)
                return rgb
        except rasterio.errors.RasterioIOError as exc:
            last_exc = exc
            delay = _RETRY_BASE_DELAY_S * 2 ** attempt
            logger.warning("Read failed (%s) attempt %d/%d: %s; retrying in %.0fs",
                           path, attempt + 1, _READ_ATTEMPTS, exc, delay)
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def read_tile(
    bbox: tuple[float, float, float, float],
    quad_index: pd.DataFrame,
    tile_size_px: int = TILE_SIZE_PX,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one inference tile by mosaicking the intersecting quads.

    Args:
        bbox: (minx, miny, maxx, maxy) in EPSG:3857.
        quad_index: DataFrame from inference.quad_index (bounds + gcs_path).
        tile_size_px: output tile edge in pixels.
        scale: inference scale per inference.md §6.2 — at scale s the bbox
            covers tile_size_px/s native pixels and the read is decimated to
            tile_size_px (e.g. s=0.5 → 2× GSD, 4× ground area, same 512²).

    Returns:
        (rgb, nodata_mask): rgb float32 (3, H, W) raw values [0, 255];
        nodata_mask bool (H, W), True where no valid imagery exists.
    """
    minx, miny, maxx, maxy = bbox
    hits = quad_index[(quad_index["minx"] < maxx) & (quad_index["maxx"] > minx)
                      & (quad_index["miny"] < maxy) & (quad_index["maxy"] > miny)]

    rgb = np.zeros((3, tile_size_px, tile_size_px), dtype=np.float32)
    valid = np.zeros((tile_size_px, tile_size_px), dtype=bool)
    res_x = (maxx - minx) / tile_size_px
    res_y = (maxy - miny) / tile_size_px

    for _, quad in hits.iterrows():
        data = _read_window_with_retry(quad["gcs_path"], bbox,
                                       out_size=None if scale == 1.0 else tile_size_px)
        if data.shape[1] != tile_size_px or data.shape[2] != tile_size_px:
            raise ValueError(
                f"Quad {quad['quad_id']} window is {data.shape[1:]} for a "
                f"{tile_size_px}px tile — resolution mismatch with the tile grid")
        alpha = data[3] > 0 if data.shape[0] >= 4 else np.ones(data.shape[1:], bool)
        # Restrict to the quad's own extent: the boundless read 0-fills outside,
        # which is indistinguishable from alpha=0 — both stay invalid.
        col0 = int(round((quad["minx"] - minx) / res_x))
        row0 = int(round((maxy - quad["maxy"]) / res_y))
        cover = np.zeros_like(alpha)
        r0, r1 = max(row0, 0), min(row0 + int(round((quad["maxy"] - quad["miny"]) / res_y)), tile_size_px)
        c0, c1 = max(col0, 0), min(col0 + int(round((quad["maxx"] - quad["minx"]) / res_x)), tile_size_px)
        cover[r0:r1, c0:c1] = True
        ok = alpha & cover & ~valid
        rgb[:, ok] = data[:3, ok].astype(np.float32)
        valid |= ok

    return rgb, ~valid


def read_ndvi_tile(
    bbox: tuple[float, float, float, float],
    s2_index: pd.DataFrame,
    tile_size_px: int = TILE_SIZE_PX,
) -> np.ndarray:
    """Window NDVI from the bulk S2 composites onto one inference tile.

    Mirrors ``read_tile`` for the EXTRA=NDVI channel: mosaics every intersecting
    S2 composite cell, computing ``NDVI = (B8 - B4) / (B8 + B4)`` — the same
    formula training derives server-side from the same ``s2_sr_composite`` recipe
    (data/extra_channels.s2_image; NDVI is scale-invariant so the /10000
    reflectance cancels — CLAUDE Rule 3). The 10 m composite is resampled
    (bilinear) onto the tile's projected grid.

    No-coverage pixels (outside every cell, or cloud/edge gaps the export left as
    0) yield NaN — the NoData contract honoured downstream by ``apply_norm``
    (non-finite → 0), matching training's EXTRA handling.

    Args:
        bbox: (minx, miny, maxx, maxy) in EPSG:3857.
        s2_index: DataFrame from inference.s2_index (bounds + gcs_path).
        tile_size_px: output tile edge in pixels.

    Returns:
        ndvi float32 (H, W); NaN where no S2 coverage / invalid.
    """
    from rasterio.enums import Resampling

    minx, miny, maxx, maxy = bbox
    hits = s2_index[(s2_index["minx"] < maxx) & (s2_index["maxx"] > minx)
                    & (s2_index["miny"] < maxy) & (s2_index["maxy"] > miny)]

    ndvi = np.full((tile_size_px, tile_size_px), np.nan, dtype=np.float32)
    for _, cell in hits.iterrows():
        # Read red (B4) + NIR (B8), resampled to the tile grid; 0-fill outside.
        last_exc: Exception | None = None
        for attempt in range(_READ_ATTEMPTS):
            try:
                with rasterio.open(cell["gcs_path"]) as src:
                    window = from_bounds(*bbox, transform=src.transform)
                    bands = src.read(
                        indexes=[S2_RED_BAND, S2_NIR_BAND], window=window,
                        boundless=True, fill_value=0,
                        out_shape=(2, tile_size_px, tile_size_px),
                        resampling=Resampling.bilinear).astype(np.float32)
                break
            except rasterio.errors.RasterioIOError as exc:
                last_exc = exc
                delay = _RETRY_BASE_DELAY_S * 2 ** attempt
                logger.warning("S2 read failed (%s) attempt %d/%d: %s; retrying in %.0fs",
                               cell["gcs_path"], attempt + 1, _READ_ATTEMPTS, exc, delay)
                time.sleep(delay)
        else:
            raise last_exc  # type: ignore[misc]

        red, nir = bands[0], bands[1]
        denom = nir + red
        with np.errstate(invalid="ignore", divide="ignore"):
            cell_ndvi = np.where(denom > 0, (nir - red) / denom, np.nan).astype(np.float32)
        # First-valid-wins mosaic: fill only pixels still uncovered.
        take = np.isnan(ndvi) & np.isfinite(cell_ndvi)
        ndvi[take] = cell_ndvi[take]

    return ndvi


class InferenceTileDataset(Dataset):
    """Tile-list dataset for batched inference (inference.md §8.1).

    Yields dicts with normalized image tensors and the NoData mask; tiles that
    are entirely NoData are flagged (`all_nodata`) so the inference loop can
    skip + manifest-log them (§5.3) without crashing the batch.
    """

    def __init__(
        self,
        tile_list: pd.DataFrame,
        quad_index: pd.DataFrame,
        stats: dict,
        tile_size_px: int = TILE_SIZE_PX,
        scale: float = 1.0,
        s2_index: pd.DataFrame | None = None,
        extra_bands: list[dict] | None = None,
    ) -> None:
        """tile_list needs columns: tile_id, minx, miny, maxx, maxy.

        ``stats`` is the deployment ``normalization_stats.json`` dict; normalization
        runs through the shared ``apply_norm`` (CLAUDE Rule 3) so RGB(+EXTRA) z-score
        / clip / NoData-neutralization match training exactly.

        EXTRA=NDVI (the locked v2 channel) is sourced on the fly from the bulk S2
        composites: pass ``s2_index`` (inference.s2_index) + ``extra_bands`` (the
        deployment ``model_config.channels.extra`` list). RGB-only when both are None.
        """
        required = {"tile_id", "minx", "miny", "maxx", "maxy"}
        missing = required - set(tile_list.columns)
        if missing:
            raise ValueError(f"tile list missing columns {sorted(missing)}")
        self.with_extra = bool(extra_bands)
        if self.with_extra:
            names = [c["name"] for c in extra_bands]
            if names != ["ndvi"]:
                raise NotImplementedError(
                    f"inference EXTRA reader supports ndvi only, got {names}")
            if s2_index is None:
                raise ValueError("extra_bands=[ndvi] requires an s2_index to window NDVI")
            if scale != 1.0:
                raise NotImplementedError("NDVI EXTRA reader supports scale=1.0 only")
        self.tiles = tile_list.reset_index(drop=True)
        self.quad_index = quad_index
        self.s2_index = s2_index
        self.norm_params = build_norm_arrays(stats, with_extra=self.with_extra)
        self.rgb_mean = self.norm_params["mean"][:3]
        self.tile_size_px = tile_size_px
        self.scale = scale

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, i: int) -> dict:
        row = self.tiles.iloc[i]
        bbox = (row["minx"], row["miny"], row["maxx"], row["maxy"])
        rgb, nodata = read_tile(bbox, self.quad_index, self.tile_size_px,
                                scale=self.scale)
        all_nodata = bool(nodata.all())
        if all_nodata:
            # Discarded by the inference loop (§5.3); emit a correctly-shaped zero
            # tensor so the batch collate (np.stack) doesn't trip on a 3-vs-4
            # channel mismatch when EXTRA=NDVI is stacked on the kept tiles.
            n_ch = 4 if self.with_extra else 3
            image = np.zeros((n_ch, self.tile_size_px, self.tile_size_px), dtype=np.float32)
        else:
            # Mean-substitute RGB NoData before z-scoring via the shared helper so
            # training and inference neutralise NoData identically (Rule 3,
            # training.md §4.4); those pixels are masked to -1.0 afterwards (§5.3).
            rgb = fill_nodata_with_mean(rgb, np.broadcast_to(nodata, rgb.shape),
                                        self.rgb_mean, channel_axis=0)
            if self.with_extra:
                # NDVI from the S2 composites; no-coverage stays NaN → apply_norm
                # neutralizes to 0 (the channel mean), exactly as in training.
                ndvi = read_ndvi_tile(bbox, self.s2_index, self.tile_size_px)
                stack = np.concatenate([rgb, ndvi[None]], axis=0)
            else:
                stack = rgb
            image = apply_norm(stack, self.norm_params)
        return {
            "tile_id": row["tile_id"],
            "image": image,
            "nodata_mask": nodata,
            "all_nodata": all_nodata,
            "bounds": np.array(bbox, dtype=np.float64),
        }

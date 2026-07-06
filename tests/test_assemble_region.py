"""Region assembly (scripts/assemble_region.py): the blocked windowed merge must
reproduce a single-shot merge exactly, so non-overlapping blocks mosaic seamlessly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inference.quad_index import RESOLUTION_M
from inference.tiles import TILE_SIZE_PX
from inference.writer import NODATA_PROB, write_probability_tile
from scripts.assemble_region import (
    build_tile_paths, canvas_bounds, iter_blocks, merge_window,
)
from scripts.merge_predictions import merge_tiles

SIGMA = 128.0
STRIDE_PX = 300  # overlapping tiles (like the deploy 344 stride)


def _synthetic_tiles(tmp: Path, n: int = 3) -> tuple[pd.DataFrame, dict[str, str]]:
    """An n×n grid of overlapping constant-value prob tiles written as COGs."""
    res = RESOLUTION_M
    rows = []
    for i in range(n):
        for j in range(n):
            minx = j * STRIDE_PX * res
            maxx = minx + TILE_SIZE_PX * res
            maxy = -i * STRIDE_PX * res
            miny = maxy - TILE_SIZE_PX * res
            tid = f"t{i}_{j}"
            prob = np.full((TILE_SIZE_PX, TILE_SIZE_PX), 0.1 * (i * n + j + 1),
                           dtype=np.float32)
            write_probability_tile(str(tmp / f"{tid}.tif"), prob,
                                   (minx, miny, maxx, maxy), dtype="float32")
            rows.append(dict(tile_id=tid, minx=minx, miny=miny,
                             maxx=maxx, maxy=maxy))
    return pd.DataFrame(rows), build_tile_paths(str(tmp))


@pytest.mark.parametrize("block_px", [200, 333, 512])
def test_blocked_merge_matches_single_shot(tmp_path, block_px):
    tiles, tile_paths = _synthetic_tiles(tmp_path)
    full, bounds = merge_tiles(tiles, str(tmp_path), sigma_px=SIGMA)
    minx, miny, maxx, maxy = bounds
    (b_minx, b_miny, b_maxx, b_maxy), width, height = canvas_bounds(tiles, RESOLUTION_M)
    assert (width, height) == (full.shape[1], full.shape[0])

    recon = np.full_like(full, NODATA_PROB)
    tx0, tx1 = tiles["minx"].to_numpy(), tiles["maxx"].to_numpy()
    ty0, ty1 = tiles["miny"].to_numpy(), tiles["maxy"].to_numpy()
    for r0, r1, c0, c1 in iter_blocks(width, height, block_px):
        w_minx = minx + c0 * RESOLUTION_M
        w_maxx = minx + c1 * RESOLUTION_M
        w_maxy = maxy - r0 * RESOLUTION_M
        w_miny = maxy - r1 * RESOLUTION_M
        sel = (tx1 > w_minx) & (tx0 < w_maxx) & (ty1 > w_miny) & (ty0 < w_maxy)
        if not sel.any():
            continue
        block = merge_window(tiles.iloc[sel], tile_paths,
                             (w_minx, w_miny, w_maxx, w_maxy), SIGMA)
        assert block.shape == (r1 - r0, c1 - c0)
        recon[r0:r1, c0:c1] = block

    # Every pixel: blocked reconstruction == single-shot merge (incl. NoData).
    both_data = (full != NODATA_PROB) & (recon != NODATA_PROB)
    assert np.array_equal(full == NODATA_PROB, recon == NODATA_PROB)
    np.testing.assert_allclose(recon[both_data], full[both_data], atol=1e-5)


def test_iter_blocks_tiles_the_canvas_without_gaps():
    covered = np.zeros((37, 41), dtype=int)
    for r0, r1, c0, c1 in iter_blocks(41, 37, 16):
        covered[r0:r1, c0:c1] += 1
    assert covered.min() == 1 and covered.max() == 1  # partition: no gap/overlap

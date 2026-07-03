"""Unit tests for scripts/scale05_tile_creation.py — pure helpers only (no GCS).

Covers the multiscale-POC staging rules (module docstring rules 1-5):
block grid alignment, negative-window clamping, ignore auto-convert
classification, unrefined-ARTS ignore, rasterization order, and the
sub-pixel positive guard.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from inference.quad_index import QUAD_SIZE_M, WORLD_MIN, quad_bounds
from scripts.scale05_tile_creation import (
    BLOCK_M,
    LabelVectors,
    MIN_POSITIVE_PX,
    RES_05,
    block_bounds,
    block_index,
    blocks_for_bounds,
    build_label,
    clamp_window_to_quad,
    quad_for_block,
)


# ---------------------------------------------------------------------------
# Block grid geometry
# ---------------------------------------------------------------------------


def test_block_size_is_quarter_quad():
    assert BLOCK_M * 4 == pytest.approx(QUAD_SIZE_M)
    assert BLOCK_M == pytest.approx(512 * RES_05)


def test_block_nests_inside_single_quad():
    # Every block must lie entirely within the quad quad_for_block reports —
    # this is what lets the reader open exactly one quad per tile.
    for bx, by in [(0, 0), (3, 3), (4, 4), (413, 6229), (8191, 8191)]:
        qminx, qminy, qmaxx, qmaxy = quad_bounds(*quad_for_block(bx, by))
        minx, miny, maxx, maxy = block_bounds(bx, by)
        assert qminx - 1e-6 <= minx and maxx <= qmaxx + 1e-6
        assert qminy - 1e-6 <= miny and maxy <= qmaxy + 1e-6


def test_block_index_roundtrip():
    bx, by = 1234, 5678
    minx, miny, maxx, maxy = block_bounds(bx, by)
    assert block_index((minx + maxx) / 2, (miny + maxy) / 2) == (bx, by)


def test_blocks_for_bounds_of_512_tile_is_single_block():
    # A native 512-grid tile is exactly half a block in each axis; its bounds
    # (grid-aligned) must map to exactly one block.
    bx, by = 100, 200
    minx, miny, _, _ = block_bounds(bx, by)
    tile = (minx, miny, minx + BLOCK_M / 2, miny + BLOCK_M / 2)  # SW quarter
    assert blocks_for_bounds(tile) == {(bx, by)}
    # NE quarter of the same block
    tile = (minx + BLOCK_M / 2, miny + BLOCK_M / 2, minx + BLOCK_M, miny + BLOCK_M)
    assert blocks_for_bounds(tile) == {(bx, by)}


def test_clamp_window_stays_in_quad_and_covers_centroid():
    qcol, qrow = 300, 1500
    qminx, qminy, qmaxx, qmaxy = quad_bounds(qcol, qrow)
    # Centroid near the quad's NE corner forces a clamp.
    cx, cy = qmaxx - 10.0, qmaxy - 10.0
    (minx, miny, maxx, maxy), (c, r) = clamp_window_to_quad(cx, cy)
    assert (c, r) == (qcol, qrow)
    assert qminx - 1e-6 <= minx and maxx <= qmaxx + 1e-6
    assert qminy - 1e-6 <= miny and maxy <= qmaxy + 1e-6
    assert maxx - minx == pytest.approx(BLOCK_M)
    assert minx <= cx <= maxx and miny <= cy <= maxy


# ---------------------------------------------------------------------------
# LabelVectors.classify — rules 1-3
# ---------------------------------------------------------------------------


def _sq(x, y, side):
    return box(x, y, x + side, y + side)


def test_classify_splits_ignore_by_positive_touch():
    positive = [_sq(0, 0, 100)]
    touching = _sq(90, 0, 100)      # overlaps the positive → auto-convert
    isolated = _sq(1000, 1000, 100)  # far away → keep as ignore
    v = LabelVectors.classify(positive, [touching, isolated], arts_positive=[])
    assert v.ignore_convert == [touching]
    assert v.ignore_keep == [isolated]


def test_classify_finds_unrefined_arts():
    positive = [_sq(0, 0, 100)]
    arts_refined = _sq(10, 10, 50)       # overlaps a refined positive → dropped
    arts_unref = _sq(5000, 5000, 100)    # not delineated → ignore, buffered
    v = LabelVectors.classify(positive, [], [arts_refined, arts_unref])
    assert len(v.arts_unrefined) == 1
    assert v.arts_unrefined[0].contains(arts_unref)  # 50 m buffer applied


# ---------------------------------------------------------------------------
# build_label — rules 2-5 (rasterization order and guards)
# ---------------------------------------------------------------------------

BOUNDS = block_bounds(1000, 1000)
X0, Y0 = BOUNDS[0], BOUNDS[1]


def _label(v):
    return build_label(v, BOUNDS)


def test_positive_overwrites_ignore():
    big = _sq(X0, Y0, BLOCK_M)                      # whole tile as kept-ignore
    pos = _sq(X0 + 1000, Y0 + 1000, 1000)
    v = LabelVectors(positive=[pos], ignore_convert=[], ignore_keep=[big],
                     arts_unrefined=[])
    lab = _label(v)
    assert lab is not None
    assert (lab == 1).sum() > 0
    # inside the positive: 1; elsewhere: 255
    assert lab[0, 0] == 255
    n1, n255 = int((lab == 1).sum()), int((lab == 255).sum())
    assert n1 + n255 == 512 * 512


def test_auto_converted_ignore_burns_as_positive():
    conv = _sq(X0 + 500, Y0 + 500, 1000)
    v = LabelVectors(positive=[], ignore_convert=[conv], ignore_keep=[],
                     arts_unrefined=[])
    lab = _label(v)
    assert lab is not None
    exp_px = (1000 / RES_05) ** 2
    assert (lab == 1).sum() == pytest.approx(exp_px, rel=0.1)
    assert (lab == 255).sum() == 0


def test_unrefined_arts_is_ignore_not_background():
    pos = _sq(X0 + 100, Y0 + 100, 500)
    arts = _sq(X0 + 3000, Y0 + 3000, 500)
    v = LabelVectors(positive=[pos], ignore_convert=[], ignore_keep=[],
                     arts_unrefined=[arts])
    lab = _label(v)
    assert lab is not None
    assert (lab == 255).sum() > 0 and (lab == 1).sum() > 0


def test_no_positive_returns_none():
    v = LabelVectors(positive=[], ignore_convert=[],
                     ignore_keep=[_sq(X0, Y0, 2000)], arts_unrefined=[])
    assert _label(v) is None


def test_subpixel_positive_becomes_ignore():
    # ~2 px at 9.55 m/px (side 14 m ≈ 1.5 px x 1.5 px) — below MIN_POSITIVE_PX.
    tiny = _sq(X0 + 2000, Y0 + 2000, 14)
    big = _sq(X0 + 100, Y0 + 100, 1000)
    v = LabelVectors(positive=[tiny, big], ignore_convert=[], ignore_keep=[],
                     arts_unrefined=[])
    lab = _label(v)
    assert lab is not None
    # The tiny feature's pixels are 255, the big one survives as 1.
    tiny_px = int((lab == 255).sum())
    assert 0 < tiny_px < MIN_POSITIVE_PX
    assert (lab == 1).sum() > 100


def test_label_georeferencing_matches_bounds():
    # A positive filling the SW quarter must land in the lower-left of the
    # array (row-major from the NW corner).
    pos = Polygon([(X0, Y0), (X0 + BLOCK_M / 2, Y0),
                   (X0 + BLOCK_M / 2, Y0 + BLOCK_M / 2), (X0, Y0 + BLOCK_M / 2)])
    v = LabelVectors(positive=[pos], ignore_convert=[], ignore_keep=[],
                     arts_unrefined=[])
    lab = _label(v)
    assert lab is not None
    assert lab[384, 128] == 1      # SW quarter → bottom-left
    assert lab[128, 384] == 0      # NE quarter → top-right

"""Unit tests for scripts/shard_tiles.make_shards (plan Phase 1).

GPU-free, I/O-free: exercises the pure split logic. The key invariant for the
claim queue is that every tile lands in exactly one shard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from shard_tiles import make_shards  # noqa: E402
from inference.tiles import _spatial_sort  # noqa: E402

TILE = 1642.0  # ~stride-344 spacing in EPSG:3857 meters (value irrelevant to the test)


def _grid(n: int) -> pd.DataFrame:
    """n tiles on a simple raster walk; columns match the inference tile list."""
    rows = []
    side = int(n ** 0.5) + 1
    for k in range(n):
        gx, gy = k % side, k // side
        x0, y0 = gx * TILE, gy * TILE
        rows.append({"tile_id": f"t{k:05d}", "minx": x0, "miny": y0,
                     "maxx": x0 + TILE, "maxy": y0 + TILE})
    return pd.DataFrame(rows)


def test_every_tile_in_exactly_one_shard():
    tiles = _grid(250)
    shards = make_shards(tiles, shard_size=40)
    seen = pd.concat([df["tile_id"] for _, df in shards])
    assert len(seen) == 250                      # no tile dropped or duplicated
    assert set(seen) == set(tiles["tile_id"])    # exact coverage
    assert seen.is_unique


def test_shard_count_and_sizes():
    tiles = _grid(250)
    shards = make_shards(tiles, shard_size=40)
    assert len(shards) == 7                       # ceil(250/40)
    sizes = [len(df) for _, df in shards]
    assert sizes[:-1] == [40] * 6                 # full shards
    assert sizes[-1] == 10                         # remainder
    assert sum(sizes) == 250


def test_shard_ids_are_sequential_and_padded():
    shards = make_shards(_grid(95), shard_size=40)
    assert [sid for sid, _ in shards] == ["shard_000000", "shard_000001", "shard_000002"]


def test_shards_concatenate_to_spatial_order():
    """Contiguous shards in order == the spatial sort (cache-locality contract)."""
    tiles = _grid(130)
    shards = make_shards(tiles, shard_size=33)
    concat = pd.concat([df for _, df in shards], ignore_index=True)
    expected = _spatial_sort(tiles).reset_index(drop=True)
    pd.testing.assert_frame_equal(concat, expected)


def test_single_shard_when_size_exceeds_count():
    shards = make_shards(_grid(10), shard_size=10000)
    assert len(shards) == 1
    assert len(shards[0][1]) == 10


@pytest.mark.parametrize("bad", [0, -5])
def test_nonpositive_shard_size_rejected(bad):
    with pytest.raises(ValueError, match="shard_size"):
        make_shards(_grid(5), shard_size=bad)

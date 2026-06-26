"""Unit tests for scripts/inference_progress pure math (plan Phase 0/4 monitor).

GPU-free, network-free: exercises compute_progress / compute_s2_progress with
injected clocks. The GCS listing + rendering around these is thin glue.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from inference_progress import compute_progress, compute_s2_progress  # noqa: E402

NOW = 1_000_000.0
INDEX = {"n_shards": 4, "n_tiles": 400,
         "shards": [{"shard_id": f"shard_{i:06d}", "n_tiles": 100} for i in range(4)]}


# --- compute_progress -----------------------------------------------------
def test_progress_counts_and_pct():
    done = [("shard_000000", NOW - 100), ("shard_000001", NOW - 50)]
    claims = [("shard_000002", "hostA:1:gpu0", NOW - 10)]
    snap = compute_progress(INDEX, done, claims, NOW, window_s=900)
    assert snap["shards"] == {"total": 4, "done": 2, "active": 1, "remaining": 2}
    assert snap["tiles"]["done"] == 200
    assert snap["tiles"]["pct"] == 50.0
    assert snap["active_hosts"] == {"hostA": 1}
    assert snap["stale_workers"] == []


def test_progress_recent_rate_and_eta():
    done = [("shard_000000", NOW - 100), ("shard_000001", NOW - 50)]  # 200 tiles in window
    snap = compute_progress(INDEX, [], [], NOW)  # empty first
    assert snap["rate_tiles_s"] == 0.0 and snap["eta_hours"] is None
    snap = compute_progress(INDEX, done, [], NOW, window_s=900)
    assert snap["rate_recent_tiles_s"] == round(200 / 900, 2)
    # remaining 200 tiles at the recent rate
    assert snap["eta_hours"] == round(200 / (200 / 900) / 3600, 2)


def test_progress_falls_back_to_average_rate():
    """No completions in the window → use the since-start average rate."""
    done = [("shard_000000", NOW - 2000), ("shard_000001", NOW - 1800)]
    snap = compute_progress(INDEX, done, [], NOW, window_s=900)
    assert snap["rate_recent_tiles_s"] == 0.0
    assert snap["rate_tiles_s"] == round(200 / 2000, 2)  # avg over the 2000s span


def test_progress_flags_stale_worker():
    claims = [("shard_000003", "hostB:9:gpu3", NOW - 1200)]  # heartbeat 1200s > 900 window
    snap = compute_progress(INDEX, [], claims, NOW, window_s=900)
    assert len(snap["stale_workers"]) == 1
    assert snap["stale_workers"][0]["shard"] == "shard_000003"


def test_progress_aggregates_per_host():
    claims = [("s0", "vmA:1:gpu0", NOW), ("s1", "vmA:2:gpu1", NOW),
              ("s2", "vmB:1:gpu0", NOW)]
    snap = compute_progress(INDEX, [], claims, NOW)
    assert snap["active_hosts"] == {"vmA": 2, "vmB": 1}


# --- compute_s2_progress --------------------------------------------------
def test_s2_counts_and_eta():
    # 3 cells done in the last hour; total 1799
    ts = [NOW - 600, NOW - 1200, NOW - 1800]
    snap = compute_s2_progress(3, 1799, ts, NOW, launched="1300/1799", window_s=3600)
    assert snap["cells"]["done"] == 3
    assert snap["cells"]["remaining"] == 1796
    assert snap["launched"] == "1300/1799"
    assert snap["rate_recent_cells_hr"] == 3.0          # 3 cells / 1h window
    assert snap["eta_hours"] == round(1796 / 3.0, 1)


def test_s2_empty_has_no_eta():
    snap = compute_s2_progress(0, 1799, [], NOW)
    assert snap["rate_cells_hr"] == 0.0
    assert snap["eta_hours"] is None and snap["eta_days"] is None

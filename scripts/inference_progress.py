"""Progress monitor for the dual-fleet inference run and the S2 export (plan Phase 0/4).

Single pane of glass, run on the A100 master. Two sources:

* ``--base gs://.../inference/2025q3_south`` — the inference run: reads the cheap
  queue markers (``shards/index.json`` for totals, ``done/`` + ``claims/`` object
  listings) so a refresh is a couple of list calls, not a scan of millions of
  COGs. Reports shards done/active/total, tiles done (exact, from per-shard tile
  counts in the index), windowed tiles/s, ETA, and per-VM/worker activity with a
  stale-claim / silent-idle flag (pre-mortem #4).
* ``--s2 gs://.../S2_RGB/2025_south`` — the GEE export: counts distinct completed
  cells (``W####_N####``) vs ``--s2-total`` and the launched count from the
  driver log, with a cells/hour rate + ETA.

Modes:
    --once           one snapshot to stdout (default)
    --watch N        refresh every N seconds (terminal dashboard)
    --json PATH      also write the machine-readable snapshot to PATH each tick
                     (the Claude-Code watcher polls this for milestones/stalls)

The progress math (`compute_progress`, `compute_s2_progress`) is pure and unit
tested; this module only adds the GCS listing + rendering around it.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

_CELL_RE = re.compile(r"[WE]\d{4}_N\d{4}")


# --------------------------------------------------------------------------
# Pure progress math (unit tested)
# --------------------------------------------------------------------------
def compute_progress(index: dict, done: list[tuple[str, float]],
                     claims: list[tuple[str, str, Optional[float]]], now: float,
                     window_s: float = 900.0) -> dict:
    """Summarise inference-run progress from cheap queue listings.

    Args:
        index: parsed ``shards/index.json`` ({n_tiles, n_shards, shards:[{shard_id,n_tiles}]}).
        done: ``(shard_id, completed_at_epoch)`` per done marker (mtime is fine).
        claims: ``(shard_id, worker_id, heartbeat_at_epoch)`` per active claim.
        now: current epoch seconds.
        window_s: trailing window for the "recent" throughput + stale-claim age.

    Returns a JSON-able snapshot dict.
    """
    shard_tiles = {s["shard_id"]: s["n_tiles"] for s in index["shards"]}
    total_shards = int(index["n_shards"])
    total_tiles = int(index["n_tiles"])

    done_ids = [sid for sid, _ in done]
    tiles_done = sum(shard_tiles.get(sid, 0) for sid in done_ids)
    n_done = len(done_ids)

    recent = [(sid, t) for sid, t in done if now - t <= window_s]
    tiles_recent = sum(shard_tiles.get(sid, 0) for sid, _ in recent)
    rate_recent = tiles_recent / window_s if recent else 0.0
    if done:
        span = max(now - min(t for _, t in done), 1e-9)
        rate_avg = tiles_done / span
    else:
        rate_avg = 0.0
    rate = rate_recent or rate_avg  # tiles/s; prefer recent, fall back to avg

    remaining = total_tiles - tiles_done
    eta_h = remaining / rate / 3600 if rate > 0 else None

    workers, per_host = [], {}
    for sid, wid, hb in claims:
        host = (wid or "?").split(":")[0]
        hb_age = (now - hb) if hb is not None else None
        stale = hb_age is not None and hb_age > window_s
        workers.append({"shard": sid, "worker_id": wid, "host": host,
                        "heartbeat_age_s": None if hb_age is None else round(hb_age, 1),
                        "stale": stale})
        per_host[host] = per_host.get(host, 0) + 1

    return {
        "mode": "inference",
        "timestamp": now,
        "shards": {"total": total_shards, "done": n_done, "active": len(claims),
                   "remaining": total_shards - n_done},
        "tiles": {"total": total_tiles, "done": tiles_done,
                  "pct": round(100 * tiles_done / total_tiles, 3) if total_tiles else 0.0},
        "rate_tiles_s": round(rate, 2),
        "rate_recent_tiles_s": round(rate_recent, 2),
        "eta_hours": None if eta_h is None else round(eta_h, 2),
        "active_hosts": per_host,
        "stale_workers": [w for w in workers if w["stale"]],
        "workers": workers,
    }


def compute_s2_progress(done_count: int, total: int, timestamps: list[float],
                        now: float, launched: Optional[str] = None,
                        window_s: float = 3600.0) -> dict:
    """Summarise GEE S2-export progress (cells completed vs total)."""
    recent = [t for t in timestamps if now - t <= window_s]
    rate_recent = len(recent) / (window_s / 3600.0) if recent else 0.0  # cells/hr
    if timestamps:
        span_h = max((now - min(timestamps)) / 3600.0, 1e-9)
        rate_avg = done_count / span_h
    else:
        rate_avg = 0.0
    rate = rate_recent or rate_avg  # cells/hr
    remaining = total - done_count
    eta_h = remaining / rate if rate > 0 else None
    return {
        "mode": "s2_export",
        "timestamp": now,
        "cells": {"total": total, "done": done_count, "remaining": remaining,
                  "pct": round(100 * done_count / total, 2) if total else 0.0},
        "launched": launched,
        "rate_cells_hr": round(rate, 2),
        "rate_recent_cells_hr": round(rate_recent, 2),
        "eta_hours": None if eta_h is None else round(eta_h, 1),
        "eta_days": None if eta_h is None else round(eta_h / 24, 2),
    }


# --------------------------------------------------------------------------
# GCS readers
# --------------------------------------------------------------------------
def _split_gs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"expected gs:// URI, got {uri}")
    bucket, _, prefix = uri[5:].partition("/")
    return bucket, prefix.rstrip("/")


def _bucket(bucket_name: str):
    import os

    from google.cloud import storage
    # Project for billing of list/read; from --project (set into env by main) or
    # the ambient GOOGLE_CLOUD_PROJECT / GCE metadata server (None = auto-detect).
    return storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT")).bucket(bucket_name)


def inference_snapshot(base_uri: str, now: Optional[float] = None,
                       window_s: float = 900.0) -> dict:
    """Read the queue markers under ``base_uri`` and compute the run snapshot."""
    now = now if now is not None else time.time()
    bucket_name, prefix = _split_gs(base_uri)
    bucket = _bucket(bucket_name)

    index = json.loads(bucket.blob(f"{prefix}/shards/index.json").download_as_text())

    done_prefix = f"{prefix}/done/"
    done = [(b.name[len(done_prefix):], b.updated.timestamp())
            for b in bucket.list_blobs(prefix=done_prefix) if b.name != done_prefix]

    claim_prefix = f"{prefix}/claims/"
    claims = []
    for b in bucket.list_blobs(prefix=claim_prefix):
        if b.name == claim_prefix:
            continue
        sid = b.name[len(claim_prefix):]
        try:
            c = json.loads(b.download_as_text())
            claims.append((sid, c.get("worker_id"), c.get("heartbeat_at", c.get("claimed_at"))))
        except Exception:  # noqa: BLE001 - claim being written/deleted; skip this tick
            continue
    return compute_progress(index, done, claims, now, window_s)


def s2_snapshot(s2_uri: str, total: int, log_path: Optional[str] = None,
                now: Optional[float] = None, window_s: float = 3600.0) -> dict:
    """List the S2 output prefix, count distinct cells, compute export snapshot."""
    now = now if now is not None else time.time()
    bucket_name, prefix = _split_gs(s2_uri)
    bucket = _bucket(bucket_name)
    cell_times: dict[str, float] = {}
    for b in bucket.list_blobs(prefix=f"{prefix}/"):
        m = _CELL_RE.search(b.name)
        if not m:
            continue
        t = b.updated.timestamp()
        cell = m.group(0)
        # keep the earliest tile time per cell (cell "done" ~ first tile written)
        cell_times[cell] = min(cell_times.get(cell, t), t)
    launched = None
    if log_path and Path(log_path).exists():
        hits = re.findall(r"launched (\d+/\d+)", Path(log_path).read_text())
        launched = hits[-1] if hits else None
    return compute_s2_progress(len(cell_times), total, list(cell_times.values()),
                               now, launched, window_s)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def render(snap: dict) -> str:
    """Human-readable one-screen dashboard for a snapshot."""
    lines = []
    if snap["mode"] == "inference":
        s, t = snap["shards"], snap["tiles"]
        lines.append(f"INFERENCE  shards {s['done']}/{s['total']} done "
                     f"({s['active']} active) | tiles {t['done']:,}/{t['total']:,} "
                     f"({t['pct']}%)")
        eta = "—" if snap["eta_hours"] is None else f"{snap['eta_hours']}h"
        lines.append(f"  rate {snap['rate_tiles_s']} tiles/s "
                     f"(recent {snap['rate_recent_tiles_s']}) | ETA {eta}")
        if snap["active_hosts"]:
            hosts = "  ".join(f"{h}:{n}" for h, n in sorted(snap["active_hosts"].items()))
            lines.append(f"  active VMs: {hosts}")
        if snap["stale_workers"]:
            lines.append(f"  ⚠ STALE claims: {[w['shard'] for w in snap['stale_workers']]}")
    else:
        c = snap["cells"]
        eta = "—" if snap["eta_days"] is None else f"{snap['eta_days']}d ({snap['eta_hours']}h)"
        lines.append(f"S2 EXPORT  cells {c['done']}/{c['total']} ({c['pct']}%) | "
                     f"launched {snap['launched']}")
        lines.append(f"  rate {snap['rate_cells_hr']} cells/hr "
                     f"(recent {snap['rate_recent_cells_hr']}) | ETA {eta}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--base", help="gs:// base prefix of the inference run")
    src.add_argument("--s2", help="gs:// prefix of S2 export output (S2_RGB/<year>_<region>)")
    p.add_argument("--s2-total", type=int, default=1799, help="total S2 cells (export mode)")
    p.add_argument("--s2-log", default=None, help="export driver log (for launched count)")
    p.add_argument("--window-s", type=float, default=None,
                   help="trailing window for recent-rate/staleness (default 900 run, 3600 S2)")
    p.add_argument("--watch", type=float, default=None, metavar="N",
                   help="refresh every N seconds (terminal dashboard)")
    p.add_argument("--json", default=None, help="write the snapshot dict to this path each tick")
    p.add_argument("--project", default=None,
                   help="GCP project for GCS billing (default: GOOGLE_CLOUD_PROJECT / metadata)")
    args = p.parse_args()
    setup_logging()
    if args.project:
        import os
        os.environ["GOOGLE_CLOUD_PROJECT"] = args.project

    def snapshot() -> dict:
        if args.base:
            return inference_snapshot(args.base, window_s=args.window_s or 900.0)
        return s2_snapshot(args.s2, args.s2_total, args.s2_log, window_s=args.window_s or 3600.0)

    while True:
        snap = snapshot()
        if args.json:
            Path(args.json).write_text(json.dumps(snap, indent=1))
        if args.watch:
            print("\033[2J\033[H", end="")  # clear screen
        print(render(snap), flush=True)
        if not args.watch:
            break
        time.sleep(args.watch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Object-recall seed-noise floor from per-seed scorecards (plan: Phase 0C). Report-only.

Phase-1's recall accept-gate must beat *training-init* noise, which the per-region
bootstrap CI (object-sampling noise) does NOT capture. This harvests the aggregate
object metrics from the per-seed scorecards (one ``object_scorecard_*.json`` per existing
v2 seed checkpoint) and reports the cross-seed mean/std/spread.

CAVEAT (carried into the output): n=3 makes the std itself a high-variance estimate
(could be ~2× off). Use it as a *rough* floor / conservative multiple, and prefer a
sign-consistency read across seeds over a tight numeric threshold (ledger norm).

Run:
    python scripts/seed_recall_noise.py \
        --scorecard object_scorecard_seed42.json \
        --scorecard object_scorecard_seed1.json \
        --scorecard object_scorecard_seed2.json \
        --out /mnt/outputs/v1.0/diagnostics/seed_recall_noise.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

_METRICS = ("obj_recall", "obj_f1", "obj_precision")


def seed_noise(scorecards: list[dict]) -> dict:
    """Cross-seed mean/std/min/max for aggregate object metrics.

    ``scorecards``: parsed ``object_scorecard_*.json`` dicts (one per seed). Reads
    ``aggregate.{obj_recall,obj_f1,obj_precision}``.
    """
    n = len(scorecards)
    out: dict = {"n_seeds": n}
    for m in _METRICS:
        vals = [sc["aggregate"][m] for sc in scorecards]
        vals = [float(v) for v in vals if v is not None]
        if not vals:
            out[m] = {"values": [], "mean": None, "std": None}
            continue
        a = np.asarray(vals, dtype=np.float64)
        out[m] = {
            "values": [round(v, 4) for v in vals],
            "mean": round(float(a.mean()), 4),
            "std": round(float(a.std(ddof=1)), 4) if a.size > 1 else 0.0,
            "min": round(float(a.min()), 4),
            "max": round(float(a.max()), 4),
            "spread": round(float(a.max() - a.min()), 4),
        }
    out["_caveat"] = (
        f"std from n={n} is a high-variance estimate; treat as a rough recall floor / "
        "conservative multiple, prefer sign-consistency across seeds over a tight cut."
    )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scorecard", action="append", required=True, metavar="PATH",
                   help="per-seed object_scorecard_*.json; repeatable (>=2)")
    p.add_argument("--out", required=True, help="output JSON path")
    args = p.parse_args()

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=str(out_path.parent / "seed_recall_noise.log"))

    if len(args.scorecard) < 2:
        raise ValueError("need >=2 per-seed scorecards to estimate seed noise")
    scorecards = [json.loads(Path(s).read_text()) for s in args.scorecard]
    result = seed_noise(scorecards)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Wrote %s", out_path)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

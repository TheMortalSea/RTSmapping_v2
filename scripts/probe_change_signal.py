"""D2 — change-signal-in-invisible-footprints probe (plan: Phase 0D2). Report-only.

The free, pre-GPU applicability check for the multi-temporal change arm. The change
arm's whole premise is that the perception-invisible objects are *active* slumps whose
signal lives in year-over-year change. The competing hypothesis is that they are
*stabilized / revegetated* slumps — spectrally subtle single-date AND low year-over-year
change — which the change channel would NOT recover.

For every perception-invisible GT object (max ensemble prob in its footprint
< ``invisible_thr``), this compares the change magnitude inside its footprint against
the ambient (background) change in the same tile, and labels it **change-bright** (a
real change signature the channel could exploit) or **change-blank** (little change —
the channel can't help). The headline is the **change-blank fraction**: if the invisible
set is majority change-blank, the change arm is falsified here, before any retrain.

Inputs are two aligned caches (no GPU):
  - ``--probs-cache``  : ``*_probs.npz`` (tids, probs, labels) — the same val cache the
    scorecard uses, to identify the invisible objects.
  - ``--change-cache`` : ``*_change.npz`` (tids, change[H,W] float32) — a per-tile change
    magnitude raster (e.g. |ΔNDVI| or bare-soil/brightness delta) over the TRAIN year-pair
    (2024−2023), built on the L4 VM from the prior- and deploy-year imagery for the val
    tiles. Construction is a separate data step (imagery-layout dependent); this script
    only consumes it.

FACTS ONLY. Writes JSON only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def classify_invisible_change(
    probs: np.ndarray,
    labels: np.ndarray,
    change: np.ndarray,
    tids: list[str],
    *,
    ignore_index: int = 255,
    invisible_thr: float = 0.30,
    bright_bg_percentile: float = 90.0,
) -> tuple[list[dict], dict]:
    """Label each perception-invisible GT object change-bright vs change-blank.

    An invisible object (max prob in footprint < ``invisible_thr``) is **change-bright**
    when its mean footprint change is strictly above the ``bright_bg_percentile`` of the
    tile's *background* change (valid, non-GT pixels) — i.e. it stands out above ambient —
    otherwise **change-blank**. The percentile reference is scale-free and robust to
    per-tile/per-year change-magnitude offsets; the strict comparison keeps a zero-change
    object in a zero-change tile correctly classified as blank.

    Returns (per_object rows, summary). The summary's ``change_blank_fraction`` is the
    go/no-go headline for the change arm.
    """
    rows: list[dict] = []
    for prob, label, chg, tid in zip(probs, labels, change, tids):
        valid = label != ignore_index
        gt = (label == 1) & valid
        gt_labels, n_gt = ndimage.label(gt.astype(np.uint8))
        if n_gt == 0:
            continue
        bg = valid & ~gt
        bg_ref = float(np.percentile(chg[bg], bright_bg_percentile)) if bg.any() else float("inf")
        for g in range(1, n_gt + 1):
            footprint = gt_labels == g
            max_p = float(prob[footprint].max())
            if max_p >= invisible_thr:
                continue  # only the perception-invisible population
            obj_change = float(chg[footprint].mean())
            is_bright = obj_change > bg_ref
            rows.append({
                "tile_id": tid,
                "area_px": int(footprint.sum()),
                "max_prob": round(max_p, 4),
                "footprint_change_mean": round(obj_change, 6),
                "background_change_ref": round(bg_ref, 6) if np.isfinite(bg_ref) else None,
                "change_class": "change_bright" if is_bright else "change_blank",
            })

    n = len(rows)
    n_bright = sum(r["change_class"] == "change_bright" for r in rows)
    n_blank = n - n_bright
    summary = {
        "n_invisible_objects": n,
        "invisible_thr": invisible_thr,
        "bright_bg_percentile": bright_bg_percentile,
        "n_change_bright": n_bright,
        "n_change_blank": n_blank,
        "change_blank_fraction": round(n_blank / n, 4) if n else None,
        "change_bright_fraction": round(n_bright / n, 4) if n else None,
    }
    if rows:
        areas_blank = np.array([r["area_px"] for r in rows if r["change_class"] == "change_blank"])
        if areas_blank.size:
            summary["change_blank_area_px"] = {
                "p50": int(np.percentile(areas_blank, 50)),
                "p90": int(np.percentile(areas_blank, 90)),
            }
    return rows, summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--probs-cache", required=True, help="*_probs.npz (tids, probs, labels)")
    p.add_argument("--change-cache", required=True,
                   help="*_change.npz (tids, change[H,W]) — per-tile change magnitude")
    p.add_argument("--out", required=True, help="output dir")
    p.add_argument("--tag", default="heldout_val")
    p.add_argument("--invisible-thr", type=float, default=0.30)
    p.add_argument("--bright-bg-percentile", type=float, default=90.0)
    p.add_argument("--ignore-index", type=int, default=255)
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=str(out_dir / f"probe_change_signal_{args.tag}.log"))

    zp = np.load(args.probs_cache, allow_pickle=True)
    zc = np.load(args.change_cache, allow_pickle=True)
    ptids = [str(t) for t in zp["tids"]]
    ctids = {str(t): i for i, t in enumerate(zc["tids"])}
    missing = [t for t in ptids if t not in ctids]
    if missing:
        raise ValueError(f"{len(missing)} probs tiles have no change raster (first: {missing[:3]})")
    # Align change rasters to the probs tile order.
    change = np.stack([zc["change"][ctids[t]] for t in ptids], axis=0)

    rows, summary = classify_invisible_change(
        zp["probs"], zp["labels"], change, ptids,
        ignore_index=args.ignore_index, invisible_thr=args.invisible_thr,
        bright_bg_percentile=args.bright_bg_percentile,
    )
    logger.info("Invisible=%d | change_blank_fraction=%s",
                summary["n_invisible_objects"], summary["change_blank_fraction"])

    report = {
        "_tag": args.tag,
        "_probs_cache": args.probs_cache,
        "_change_cache": args.change_cache,
        "_note": ("change-blank-majority falsifies the change arm pre-GPU; "
                  "report-only, no recommendation."),
        "summary": summary,
        "objects": rows,
    }
    out_path = out_dir / f"probe_change_signal_{args.tag}.json"
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s", out_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

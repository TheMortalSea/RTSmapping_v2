"""Phase-0 outline-quality diagnosis on cached predictions — report-only, no GPU.

Answers one question: *is the loose outline of detected RTS objects a fixable
geometry problem, or are the outlines already as good as the labels allow?* This
is the go/no-go gate for the outline-shape refinement track (a separate track from
object recall — it only looks at objects the model already detects).

Runs on any cached ``*_probs.npz`` (tids, probs, labels), at the deployed operating
point (threshold + min_blob from deployment.yaml). Three facts-only analyses:

  1. **Matched-pair IoU distribution** of detected objects — overall and stratified
     by GT object size. (How tight are the outlines we get?)
  2. **IoU-gap decomposition — partial-detection vs loose-outline.** For each matched
     pair: coverage = |pred∩gt|/|gt| (partial = coverage<0.7 → model caught only part;
     loose = coverage>=0.7 → object caught, edge sloppy). And a spatial lens: what
     fraction of matched-detection error mass (FP+FN over matched objects) lies within
     ``band`` px of the GT boundary (edge slop) vs deeper (structural). Only loose /
     near-boundary error is what Door-1 refinement can touch.
  3. **Label-agreement ceiling (approximated).** No replicate annotations exist, so we
     approximate from the locked ignore-band physics: model two annotators each within
     +-1 px (2-px envelope). Per matched GT object the self-disagreement ceiling is
     ``|erode(obj,1)| / |dilate(obj,1)|`` (== IoU(erode, dilate)). We cannot outline
     more tightly than the labels agree with themselves — set the target below this.

Then it prints the **pre-registered decision**: proceed only if matched-pair IoU
median is >= 0.05 below the ceiling median (overall AND mid/large bins) AND the
loose-outline error share (near-boundary & coverage>=0.7) is >= 50% of matched-pair
error mass. Otherwise STOP — the gap is partial-detection (a recall problem), not
geometry.

FACTS ONLY + a mechanical decision flag; no recommendations. Writes JSON only,
never touches configs. Reuses the exact ``training.metrics`` object machinery so
the matched set is identical to the scorecard.

Run:
    python scripts/diagnose_outline_quality.py \
        --cache /mnt/outputs/v1.0/object_operating_point/effb5_ensemble/val_probs.npz \
        --metadata /mnt/outputs/v1.0/data_local/metadata.csv \
        --out /mnt/outputs/v1.0/diagnostics --tag heldout_val
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.splits import load_metadata  # noqa: E402
from training.metrics import _filter_small_blobs  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# Pre-registered decision constants (frozen before reading any Phase-0 output).
COVERAGE_LOOSE = 0.7      # coverage >= this => "loose-outline" (object essentially caught)
CEILING_GAP_MIN = 0.05    # matched IoU median must be >= this below ceiling median
LOOSE_SHARE_MIN = 0.50    # loose-outline error mass must be >= this of total error mass
CEILING_ERODE_ITERS = 1   # +-1 px annotator uncertainty (2-px envelope = locked band w=2)

# GT-object size bins (area in pixels; 1 px ~= 22.75 m^2 at 4.77 m/px).
SIZE_BIN_EDGES = [0, 2000, 10000, 50000, np.inf]
SIZE_BIN_LABELS = ["<2000", "2000-10000", "10000-50000", ">=50000"]
MID_LARGE_BINS = ["10000-50000", ">=50000"]  # bins the headroom rule also checks


def _size_bin(area_px: int) -> str:
    for i in range(len(SIZE_BIN_LABELS)):
        if SIZE_BIN_EDGES[i] <= area_px < SIZE_BIN_EDGES[i + 1]:
            return SIZE_BIN_LABELS[i]
    return SIZE_BIN_LABELS[-1]


def _ceiling_iou(blob: np.ndarray, iters: int = CEILING_ERODE_ITERS) -> float:
    """Label self-agreement ceiling for one GT object: IoU(erode, dilate).

    Two annotators each uncertain within +-``iters`` px draw the same object at its
    eroded vs dilated extent; since erode ⊆ dilate, IoU = |erode| / |dilate|.
    """
    struct = ndimage.generate_binary_structure(2, 1)
    ero = ndimage.binary_erosion(blob, structure=struct, iterations=iters, border_value=0)
    dil = ndimage.binary_dilation(blob, structure=struct, iterations=iters)
    d = int(dil.sum())
    return float(int(ero.sum()) / d) if d else 0.0


def _match_pairs(
    pred_labels: np.ndarray, n_pred: int,
    gt_labels: np.ndarray, n_gt: int,
    conf: np.ndarray, iou_thr: float,
) -> list[tuple[int, int, float, int, int, int]]:
    """Greedy 1-to-1 matching by descending confidence (identical rule to
    ``training.metrics._object_match_detail``), returning the matched pairs.

    Each pair: (pred_label_1based, gt_label_1based, iou, inter, pred_area, gt_area).
    """
    if n_pred == 0 or n_gt == 0:
        return []
    pred_masks = [pred_labels == (p + 1) for p in range(n_pred)]
    gt_masks = [gt_labels == (g + 1) for g in range(n_gt)]
    pred_areas = [int(m.sum()) for m in pred_masks]
    gt_areas = [int(m.sum()) for m in gt_masks]

    iou = np.zeros((n_pred, n_gt), dtype=np.float64)
    inter = np.zeros((n_pred, n_gt), dtype=np.int64)
    for p in range(n_pred):
        for g in range(n_gt):
            i = int(np.logical_and(pred_masks[p], gt_masks[g]).sum())
            if i:
                inter[p, g] = i
                iou[p, g] = i / (pred_areas[p] + gt_areas[g] - i)

    order = np.argsort(conf)[::-1]
    matched_gt: set[int] = set()
    pairs: list[tuple[int, int, float, int, int, int]] = []
    for p in order:
        row = iou[p].copy()
        for g in matched_gt:
            row[g] = 0.0
        g = int(np.argmax(row))
        if row[g] >= iou_thr:
            matched_gt.add(g)
            pairs.append((p + 1, g + 1, float(row[g]), int(inter[p, g]),
                          pred_areas[p], gt_areas[g]))
    return pairs


def analyze_tile(
    prob: np.ndarray, label: np.ndarray, *,
    ignore_index: int, thr: float, min_blob: int, iou_thr: float, band: int,
) -> dict:
    """Per-tile matched-pair geometry: pair records + matched-detection error mass.

    Returns per-pair rows (iou, coverage, gt_area, size_bin, ceiling) and the tile's
    error-mass decomposition over matched objects (near-boundary vs interior, split by
    the coverage-loose condition), computed per-pair against each pair's own GT blob.
    """
    valid = label != ignore_index
    gt = (label == 1) & valid
    pred = (prob >= thr) & valid
    pred_filt = _filter_small_blobs(pred.astype(np.uint8), min_blob)
    pred_labels, n_pred = ndimage.label(pred_filt)
    gt_labels, n_gt = ndimage.label(gt.astype(np.uint8))
    conf = (np.array(ndimage.mean(prob, pred_labels, index=np.arange(1, n_pred + 1)),
                     dtype=np.float64) if n_pred > 0 else np.zeros(0))
    pairs = _match_pairs(pred_labels, n_pred, gt_labels, n_gt, conf, iou_thr)

    rows: list[dict] = []
    err = {"total": 0, "near": 0, "loose_near": 0, "loose_total": 0}
    for p_id, g_id, iou, inter, p_area, g_area in pairs:
        coverage = inter / g_area if g_area else 0.0
        gm = gt_labels == g_id
        pm = pred_labels == p_id
        rows.append({
            "iou": round(iou, 4),
            "coverage": round(coverage, 4),
            "pred_precision": round(inter / p_area, 4) if p_area else 0.0,
            "gt_area_px": g_area,
            "size_bin": _size_bin(g_area),
            "ceiling_iou": round(_ceiling_iou(gm), 4),
        })
        # Per-pair error mass, classified by distance to THIS GT blob's boundary.
        fp = pm & ~gm
        fn = gm & ~pm
        dist_out = ndimage.distance_transform_edt(~gm)   # for FP (outside gt)
        dist_in = ndimage.distance_transform_edt(gm)     # for FN (inside gt)
        near = int((fp & (dist_out <= band)).sum()) + int((fn & (dist_in <= band)).sum())
        total = int(fp.sum()) + int(fn.sum())
        err["total"] += total
        err["near"] += near
        if coverage >= COVERAGE_LOOSE:
            err["loose_total"] += total
            err["loose_near"] += near
    return {"pairs": rows, "error": err}


def _pct(a: np.ndarray, q: float) -> float:
    return round(float(np.percentile(a, q)), 4)


def build_diagnosis(
    probs: np.ndarray, labels: np.ndarray, tids: list[str],
    tid_region: dict[str, str], *,
    ignore_index: int, thr: float, min_blob: int, iou_thr: float, band: int,
) -> dict:
    """Assemble the Phase-0 diagnosis (steps 1-3 + the pre-registered decision)."""
    all_rows: list[dict] = []
    err = {"total": 0, "near": 0, "loose_near": 0, "loose_total": 0}
    for prob, label in zip(probs, labels):
        t = analyze_tile(prob, label, ignore_index=ignore_index, thr=thr,
                         min_blob=min_blob, iou_thr=iou_thr, band=band)
        all_rows.extend(t["pairs"])
        for k in err:
            err[k] += t["error"][k]

    n = len(all_rows)
    ious = np.array([r["iou"] for r in all_rows]) if n else np.zeros(0)
    ceils = np.array([r["ceiling_iou"] for r in all_rows]) if n else np.zeros(0)
    cover = np.array([r["coverage"] for r in all_rows]) if n else np.zeros(0)

    # --- Step 1: matched-pair IoU distribution (overall + histogram) ---
    hist_edges = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]
    hist_counts = ([int(x) for x in np.histogram(ious, bins=np.arange(0.0, 1.01, 0.1))[0]]
                   if n else [])
    distribution = {
        "n_matched": n,
        "iou_median": _pct(ious, 50) if n else None,
        "iou_mean": round(float(ious.mean()), 4) if n else None,
        "iou_p10": _pct(ious, 10) if n else None,
        "iou_p90": _pct(ious, 90) if n else None,
        "histogram_edges": hist_edges,
        "histogram_counts": hist_counts,
    }

    # --- Step 3 (paired to Step 1): ceiling per matched object + gap, by size bin ---
    per_bin: dict[str, dict] = {}
    for lbl in SIZE_BIN_LABELS:
        idx = [i for i, r in enumerate(all_rows) if r["size_bin"] == lbl]
        if not idx:
            per_bin[lbl] = {"n_matched": 0}
            continue
        bi = ious[idx]; bc = ceils[idx]
        per_bin[lbl] = {
            "n_matched": len(idx),
            "iou_median": round(float(np.median(bi)), 4),
            "ceiling_median": round(float(np.median(bc)), 4),
            "gap": round(float(np.median(bc) - np.median(bi)), 4),
        }
    ceiling = {
        "method": f"IoU(erode,dilate) at +-{CEILING_ERODE_ITERS}px per matched GT object",
        "ceiling_median": _pct(ceils, 50) if n else None,
        "ceiling_p10": _pct(ceils, 10) if n else None,
        "gap_overall": round(float(np.median(ceils) - np.median(ious)), 4) if n else None,
        "by_size_bin": per_bin,
    }

    # --- Step 2: partial-detection vs loose-outline decomposition ---
    n_partial = int((cover < COVERAGE_LOOSE).sum()) if n else 0
    n_loose = int((cover >= COVERAGE_LOOSE).sum()) if n else 0
    total_err = err["total"]
    decomposition = {
        "coverage_lens": {
            "coverage_loose_threshold": COVERAGE_LOOSE,
            "n_partial_detection": n_partial,   # coverage < 0.7 (only part caught)
            "n_loose_outline": n_loose,         # coverage >= 0.7 (edge sloppy)
            "partial_fraction_of_matched": round(n_partial / n, 4) if n else None,
        },
        "spatial_lens": {
            "band_px": band,
            "matched_error_mass_px": total_err,
            "near_boundary_mass_px": err["near"],
            "near_boundary_share": round(err["near"] / total_err, 4) if total_err else None,
            "loose_and_near_mass_px": err["loose_near"],
            "loose_outline_error_share": round(err["loose_near"] / total_err, 4) if total_err else None,
        },
    }

    # --- Pre-registered decision (mechanical; frozen thresholds) ---
    gap_overall = ceiling["gap_overall"]
    headroom_overall = gap_overall is not None and gap_overall >= CEILING_GAP_MIN
    headroom_bins = all(
        per_bin.get(b, {}).get("n_matched", 0) > 0
        and per_bin[b]["gap"] >= CEILING_GAP_MIN
        for b in MID_LARGE_BINS
    )
    loose_share = decomposition["spatial_lens"]["loose_outline_error_share"]
    material = loose_share is not None and loose_share >= LOOSE_SHARE_MIN
    proceed = bool(headroom_overall and headroom_bins and material)
    decision = {
        "rule": (
            f"proceed iff (IoU-median <= ceiling-median - {CEILING_GAP_MIN} overall AND "
            f"in bins {MID_LARGE_BINS}) AND (loose-outline error share >= {LOOSE_SHARE_MIN})"
        ),
        "headroom_overall": bool(headroom_overall),
        "headroom_mid_large_bins": bool(headroom_bins),
        "material_loose_outline": bool(material),
        "PROCEED": proceed,
        "verdict": "PROCEED to Phase 1" if proceed
                   else "STOP — outlines are as good as the labels allow / gap is recall, not geometry",
    }

    return {
        "config": {
            "threshold": thr, "min_blob_size": min_blob, "object_iou_threshold": iou_thr,
            "boundary_band_px": band, "coverage_loose_threshold": COVERAGE_LOOSE,
            "size_bin_edges_px": [x if np.isfinite(x) else None for x in SIZE_BIN_EDGES],
        },
        "step1_matched_pair_iou": distribution,
        "step2_gap_decomposition": decomposition,
        "step3_label_agreement_ceiling": ceiling,
        "decision": decision,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache", required=True, help="*_probs.npz (tids, probs, labels)")
    p.add_argument("--metadata", required=True, help="metadata.csv for RegionName map")
    p.add_argument("--out", required=True, help="output dir")
    p.add_argument("--tag", default="heldout_val", help="split label (e.g. heldout_val)")
    p.add_argument("--deployment-yaml", default="configs/deployment.yaml")
    p.add_argument("--min-blob", type=int, default=None,
                   help="override min_blob (default: deployment.yaml min_blob_size_px)")
    p.add_argument("--iou-thr", type=float, default=0.3, help="object match IoU")
    p.add_argument("--band", type=int, default=2,
                   help="near-boundary band in px (default 2 = locked ignore-band width)")
    p.add_argument("--ignore-index", type=int, default=255)
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=str(out_dir / f"diagnose_outline_quality_{args.tag}.log"))

    dep = yaml.safe_load(Path(args.deployment_yaml).read_text())
    thr = float(dep["threshold"])
    min_blob = int(args.min_blob) if args.min_blob is not None else int(dep.get("min_blob_size_px", 10))

    logger.info("Loading cached predictions: %s", args.cache)
    z = np.load(args.cache, allow_pickle=True)
    tids = [str(t) for t in z["tids"]]
    probs, labels = z["probs"], z["labels"]
    logger.info("%d tiles | tag=%s thr=%.3f min_blob=%d band=%d",
                len(tids), args.tag, thr, min_blob, args.band)

    meta = load_metadata(args.metadata)
    tid_region = dict(zip(meta["Tile_ID"], meta["RegionName"]))

    diag = build_diagnosis(
        probs, labels, tids, tid_region,
        ignore_index=args.ignore_index, thr=thr, min_blob=min_blob,
        iou_thr=args.iou_thr, band=args.band,
    )
    diag["_tag"] = args.tag
    diag["_source_cache"] = args.cache

    out_path = out_dir / f"diagnose_outline_quality_{args.tag}.json"
    out_path.write_text(json.dumps(diag, indent=2))
    logger.info("Wrote %s", out_path)

    print(json.dumps({
        "tag": args.tag,
        "n_matched": diag["step1_matched_pair_iou"]["n_matched"],
        "iou_median": diag["step1_matched_pair_iou"]["iou_median"],
        "ceiling_median": diag["step3_label_agreement_ceiling"]["ceiling_median"],
        "gap_overall": diag["step3_label_agreement_ceiling"]["gap_overall"],
        "loose_outline_error_share": diag["step2_gap_decomposition"]["spatial_lens"]["loose_outline_error_share"],
        "near_boundary_share": diag["step2_gap_decomposition"]["spatial_lens"]["near_boundary_share"],
        "partial_fraction": diag["step2_gap_decomposition"]["coverage_lens"]["partial_fraction_of_matched"],
        "decision": diag["decision"]["verdict"],
        "PROCEED": diag["decision"]["PROCEED"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

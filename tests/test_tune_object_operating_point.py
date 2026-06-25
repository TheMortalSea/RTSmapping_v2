"""Tests for the Tier-1 object operating-point tuner (scripts/tune_object_operating_point.py).

GPU-free; synthetic prob/label maps with known objects. The load-bearing test is
parity: at the defaults (thr 0.5, min_blob 10, no morph) the tuner's object counts
must equal training.metrics.ValidationAccumulator for the same input.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.tune_object_operating_point import (
    _logit, decompose, evaluate_grid, object_counts,
)
from training.metrics import ValidationAccumulator

IGNORE = 255


def _parity_scene() -> tuple[np.ndarray, np.ndarray]:
    """64×64: GT1 with a matching pred (TP), a spurious pred (FP), GT2 unpredicted (FN)."""
    label = np.zeros((64, 64), dtype=np.int64)
    label[10:20, 10:20] = 1          # GT1 (100 px)
    label[40:50, 40:50] = 1          # GT2 (100 px, no prediction → FN)
    label[0:5, 0:5] = IGNORE         # ignored corner
    prob = np.full((64, 64), 0.05, dtype=np.float32)
    prob[10:20, 10:20] = 0.9         # matches GT1 → TP
    prob[30:34, 55:59] = 0.9         # 16 px spurious, no GT overlap → FP
    return prob, label


def test_parity_with_validation_accumulator():
    prob, label = _parity_scene()

    cfg = {"data": {"label_ignore_index": IGNORE}, "metrics": {"pr_auc_ratios": [200]}}
    acc = ValidationAccumulator(cfg, ratios=[200])
    logits = torch.from_numpy(_logit(prob).astype(np.float32))[None, None]  # (1,1,H,W)
    labels = torch.from_numpy(label)[None]                                  # (1,H,W)
    acc.update(logits, labels, ["t0"])

    tp, fp, fn, *_ = object_counts(prob, label, IGNORE, thr=0.5, min_blob=10,
                                   morph_r=0, iou_thr=0.3)
    assert (tp, fp, fn) == (acc.obj_tp, acc.obj_fp, acc.obj_fn)
    assert (tp, fp, fn) == (1, 1, 1)


def test_min_blob_filters_small_predictions():
    prob, label = _parity_scene()
    # The spurious blob is 16 px; min_blob=20 removes it → its FP disappears.
    tp, fp, fn, *_ = object_counts(prob, label, IGNORE, thr=0.5, min_blob=20,
                                   morph_r=0, iou_thr=0.3)
    assert (tp, fp, fn) == (1, 0, 1)


def test_morph_closing_merges_fragments():
    label = np.zeros((64, 64), dtype=np.int64)
    label[20:40, 20:40] = 1                      # one 20×20 GT
    prob = np.full((64, 64), 0.05, dtype=np.float32)
    prob[20:40, 20:30] = 0.9                      # left half
    prob[20:40, 31:40] = 0.9                      # right half (1-col gap at 30)

    # No morph: two blobs; one matches GT (TP), the other is a fragment FP.
    _, fp0, _, *_ = object_counts(prob, label, IGNORE, 0.5, 10, morph_r=0, iou_thr=0.3)
    # Closing the 1-px gap merges them into one blob → fragment FP gone.
    tp1, fp1, fn1, *_ = object_counts(prob, label, IGNORE, 0.5, 10, morph_r=1, iou_thr=0.3)
    assert fp0 >= 1
    assert (tp1, fp1, fn1) == (1, 0, 0)


def test_decompose_categories():
    prob, label = _parity_scene()
    d = decompose(prob[None], label[None], IGNORE, thr=0.5, min_blob=10, morph_r=0, iou_thr=0.3)
    assert d["fp_no_overlap"] == 1          # the spurious blob overlaps no GT
    assert d["fn_missed"] == 1              # GT2 has no prediction
    assert d["fp_low_iou"] == 0 and d["fp_fragment"] == 0


def test_evaluate_grid_equivalent_to_object_counts():
    # The optimized grid (label/morph once per thr×morph, reused across min_blob)
    # must equal summing object_counts over tiles for every cell.
    p1, l1 = _parity_scene()
    l2 = np.zeros((64, 64), dtype=np.int64); l2[20:40, 20:40] = 1
    p2 = np.full((64, 64), 0.05, dtype=np.float32); p2[20:40, 20:30] = 0.9; p2[20:40, 31:40] = 0.9
    l3 = np.zeros((64, 64), dtype=np.int64)                       # all-negative + spurious
    p3 = np.full((64, 64), 0.05, dtype=np.float32); p3[5:9, 5:9] = 0.7; p3[50:60, 50:60] = 0.95
    probs, labels = np.stack([p1, p2, p3]), np.stack([l1, l2, l3])

    grid = evaluate_grid(probs, labels, IGNORE, 0.3,
                         thresholds=[0.3, 0.5, 0.8], min_blobs=[1, 10, 40], morphs=[0, 1, 2])
    for r in grid:
        otp = ofp = ofn = ptp = pfp = pfn = 0
        for prob, label in zip(probs, labels):
            a, b, c, d, e, f = object_counts(prob, label, IGNORE, r["threshold"],
                                             r["min_blob_size"], r["morph_close_radius"], 0.3)
            otp += a; ofp += b; ofn += c; ptp += d; pfp += e; pfn += f
        assert (r["obj_tp"], r["obj_fp"], r["obj_fn"]) == (otp, ofp, ofn), r
        assert r["pixel_precision"] == round(ptp / (ptp + pfp) if ptp + pfp else 0.0, 4)


def test_evaluate_grid_shape_and_threshold_monotonicity():
    prob, label = _parity_scene()
    probs, labels = prob[None], label[None]
    grid = evaluate_grid(probs, labels, IGNORE, iou_thr=0.3,
                         thresholds=[0.5, 0.95], min_blobs=[10], morphs=[0])
    assert len(grid) == 2
    by_thr = {r["threshold"]: r for r in grid}
    # At 0.95 the 0.9-prob blobs vanish → no TP (recall drops to 0).
    assert by_thr[0.95]["obj_tp"] == 0
    assert by_thr[0.5]["obj_tp"] == 1

"""Minimum Mapping Unit (sub-MMU → ignore) at scoring/loss time — metric semantics.

The data-v1.1 fix marks positive connected-components below the Minimum Mapping Unit
as ignore (255) so the loss and every object metric treat them identically: never a
false negative (uncounted GT) and never a false positive (model firing there is
masked). These tests assert that composition through the EXISTING 255-ignore
machinery — not the primitive itself (covered by tests/test_label_cleaning.py).

All synthetic, CPU-only. (Imports training.metrics → torch, a test dependency.)
"""

from __future__ import annotations

import numpy as np
import torch

from data.label_cleaning import apply_min_mapping_unit
from losses.segmentation_losses import FocalLoss
from scripts.analyze_residual_errors import object_counts
from scripts.object_scorecard import build_scorecard


def _apply_mmu(labels: np.ndarray, mmu: int, ignore_index: int = 255) -> np.ndarray:
    return np.stack([apply_min_mapping_unit(lab, mmu, ignore_index=ignore_index) for lab in labels])


def test_apply_min_mapping_unit_off_is_identity():
    """mmu_px <= 1 returns the input unchanged (the reproducibility-preserving default)."""
    lab = np.zeros((16, 16), np.int64)
    lab[2:5, 2:5] = 1          # 9-px blob
    assert apply_min_mapping_unit(lab, 0) is lab
    assert apply_min_mapping_unit(lab, 1) is lab
    np.testing.assert_array_equal(apply_min_mapping_unit(lab, 0), lab)


def test_sub_mmu_gt_plus_correct_pred_zero_fp_zero_fn():
    """A sub-MMU GT covered by a correct pred → no false negative AND no false positive."""
    label = np.zeros((32, 32), np.int64)
    label[2:5, 2:5] = 1                         # 9-px sub-MMU sliver
    prob = np.zeros((32, 32), np.float32)
    prob[2:5, 2:5] = 0.9                        # model correctly fires exactly over it

    # WITHOUT the floor: the sliver is an object, pred is <min_blob → dropped → an FN.
    otp0, ofp0, ofn0, *_ = object_counts(prob, label, 255, thr=0.5, min_blob=50, morph_r=0, iou_thr=0.3)
    assert (otp0, ofp0, ofn0) == (0, 0, 1)

    # WITH the floor: the sliver → 255 → uncounted (no FN) and pred masked out (no FP).
    lab_mmu = apply_min_mapping_unit(label, 50)
    otp, ofp, ofn, *_ = object_counts(prob, lab_mmu, 255, thr=0.5, min_blob=50, morph_r=0, iou_thr=0.3)
    assert (otp, ofp, ofn) == (0, 0, 0)


def test_real_object_survives_mmu():
    """A ≥3000px GT with a matching pred still scores as 1 TP at mmu=600."""
    label = np.zeros((80, 80), np.int64)
    label[10:70, 10:70] = 1                     # 3600-px real object
    prob = np.zeros((80, 80), np.float32)
    prob[10:70, 10:70] = 0.9
    lab_mmu = apply_min_mapping_unit(label, 600)
    assert int((lab_mmu == 1).sum()) == 3600    # untouched
    otp, ofp, ofn, *_ = object_counts(prob, lab_mmu, 255, thr=0.5, min_blob=100, morph_r=0, iou_thr=0.3)
    assert (otp, ofp, ofn) == (1, 0, 0)


def test_straddle_pred_still_fp_on_background():
    """A pred straddling a sub-MMU sliver + genuine background keeps exactly 1 FP.

    The 255 carve-out removes only the ambiguous sliver pixels; the background portion
    of the pred (earned on true background) still forms a blob and counts as an FP.
    """
    label = np.zeros((32, 40), np.int64)
    label[0:2, 0:2] = 1                         # 4-px sub-MMU sliver at the corner
    prob = np.zeros((32, 40), np.float32)
    prob[0:2, 0:16] = 0.9                       # one contiguous pred: sliver + 28px background

    lab_mmu = apply_min_mapping_unit(label, 50)
    otp, ofp, ofn, *_ = object_counts(prob, lab_mmu, 255, thr=0.5, min_blob=4, morph_r=0, iou_thr=0.3)
    assert (otp, ofp, ofn) == (0, 1, 0)         # sliver gone (no FN); background blob is the FP


def _mk(H, W, gt_boxes, pred_boxes):
    lab = np.zeros((H, W), np.int64)
    pr = np.zeros((H, W), np.float32)
    for (r0, r1, c0, c1) in gt_boxes:
        lab[r0:r1, c0:c1] = 1
    for (r0, r1, c0, c1, v) in pred_boxes:
        pr[r0:r1, c0:c1] = v
    return lab, pr


def test_scorecard_self_check_holds_with_mmu():
    """Applying the floor to labels before build_scorecard keeps the parity self-check.

    Mirrors object_scorecard.main: clean labels once, then all three scoring paths
    (score_by_region / typology / detail) receive the identical array.
    """
    H = W = 40
    tids, probs, labels, regions = [], [], [], {}

    def add(tid, region, gt, pred):
        lab, pr = _mk(H, W, gt, pred)
        tids.append(tid); regions[tid] = region
        labels.append(lab); probs.append(pr)

    add("A1", "RegA", [(4, 30, 4, 30)], [(4, 30, 4, 30, 0.9)])       # big clean TP
    add("A2", "RegA", [(0, 3, 0, 3)], [(0, 3, 0, 3, 0.9)])           # sub-MMU sliver + pred
    add("A3", "RegA", [(2, 8, 2, 8)], [(2, 8, 2, 8, 0.1)])           # small invisible

    probs_a = np.stack(probs)
    labels_a = _apply_mmu(np.stack(labels), 50)
    sc = build_scorecard(
        probs_a, labels_a, tids, dict(regions),
        ignore_index=255, thr=0.65, min_blob=10, iou_thr=0.3,
        low_thr=0.3, overlap_frac=0.1, n_boot=100, seed=42,
    )
    assert sc["self_check"]["detail_vs_score_by_region_counts_match"] is True


def test_pixel_metrics_and_loss_ignore_sub_mmu():
    """Pixel counts of the real object are unmoved; the loss excludes the sub-MMU pixels."""
    label = np.zeros((40, 40), np.int64)
    label[5:35, 5:35] = 1                        # 900-px real object
    label[0:2, 0:2] = 1                          # 4-px sub-MMU sliver
    prob = np.zeros((40, 40), np.float32)
    prob[5:35, 5:35] = 0.9                        # correct over the real object only

    _, _, _, ptp0, pfp0, pfn0 = object_counts(prob, label, 255, 0.5, 10, 0, 0.3)
    lab_mmu = apply_min_mapping_unit(label, 50)
    _, _, _, ptp1, pfp1, pfn1 = object_counts(prob, lab_mmu, 255, 0.5, 10, 0, 0.3)
    assert ptp1 == ptp0 and pfp1 == pfp0        # real object pixels unchanged
    assert pfn1 == pfn0 - 4                       # only the 4 sliver px leave the FN count

    # Loss: pixels under the sub-MMU sliver are now 255 → masked out of both numerator
    # and denominator, so the loss is invariant to whatever the model predicts there.
    fl = FocalLoss()
    lab_t = torch.from_numpy(lab_mmu[None])
    logits_a = torch.full((1, 1, 40, 40), -3.0)          # confident background
    logits_b = logits_a.clone(); logits_b[0, 0, 0:2, 0:2] = 100.0   # wildly different under sliver
    assert torch.allclose(fl(logits_a, lab_t), fl(logits_b, lab_t))

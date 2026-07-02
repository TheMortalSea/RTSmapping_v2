"""Unit tests for the object scorecard + applicability probes (Phase 0).

Covers the report-only diagnostic instruments:
  - scripts.analyze_residual_errors: object_detail_counts, bootstrap_region_object_ci,
    _geometry_summary
  - scripts.object_scorecard: build_scorecard (self-check, splits/merges, typology)
  - scripts.score_insample_train: sample_region_stratified
  - scripts.probe_change_signal: classify_invisible_change
  - scripts.seed_recall_noise: seed_noise

All synthetic, no GPU/GCS. (Modules import training.metrics → torch; torch is a
test dependency per tests/tests.md.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_residual_errors import (
    _geometry_summary,
    bootstrap_region_object_ci,
    object_counts,
    object_detail_counts,
)
from scripts.make_invisible_contact_sheet import find_invisible_objects
from scripts.object_scorecard import build_scorecard
from scripts.probe_change_signal import classify_invisible_change
from scripts.score_insample_train import sample_region_stratified
from scripts.seed_recall_noise import seed_noise


# ---------------------------------------------------------------------------
# object_detail_counts — parity with object_counts + split/merge
# ---------------------------------------------------------------------------


def test_detail_counts_match_object_counts_and_flag_split():
    """obj tp/fp/fn must equal object_counts; a 1-GT/2-pred tile yields n_splits>=1."""
    label = np.zeros((20, 20), np.int64)
    label[2:6, 2:14] = 1                       # one wide GT object
    prob = np.zeros((20, 20), np.float32)
    prob[2:6, 2:6] = 0.9                        # fragment A over the GT
    prob[2:6, 9:13] = 0.9                       # fragment B over the GT (split)
    prob[15:18, 15:18] = 0.9                    # separate FP blob
    oc = object_counts(prob, label, 255, thr=0.5, min_blob=4, morph_r=0, iou_thr=0.3)
    otp, ofp, ofn, ns, nm, ious = object_detail_counts(
        prob, label, 255, thr=0.5, min_blob=4, iou_thr=0.3,
    )
    assert (otp, ofp, ofn) == (oc[0], oc[1], oc[2])
    assert ns >= 1 and nm == 0


def test_detail_counts_merge():
    """One prediction spanning two GT objects -> n_merges>=1."""
    label = np.zeros((12, 24), np.int64)
    label[2:8, 2:8] = 1
    label[2:8, 14:20] = 1
    prob = np.zeros((12, 24), np.float32)
    prob[2:8, 2:20] = 0.9                       # single blob covering both GTs
    _, _, _, ns, nm, _ = object_detail_counts(prob, label, 255, 0.5, 4, 0.3)
    assert nm >= 1 and ns == 0


# ---------------------------------------------------------------------------
# bootstrap_region_object_ci
# ---------------------------------------------------------------------------


def test_bootstrap_point_and_ci_ordering():
    tc = [(1, 0, 0), (1, 0, 1), (0, 1, 1), (1, 0, 0), (0, 0, 1), (1, 1, 0)]
    b = bootstrap_region_object_ci(tc, n_boot=400, seed=42)
    # tp=4, fp=2, fn=3 -> precision 4/6, recall 4/7.
    assert b["precision"]["point"] == pytest.approx(4 / 6, abs=1e-3)
    assert b["recall"]["point"] == pytest.approx(4 / 7, abs=1e-3)
    for k in ("precision", "recall", "f1"):
        assert 0.0 <= b[k]["lo"] <= b[k]["point"] <= b[k]["hi"] <= 1.0
    assert b["n_tiles"] == 6


def test_bootstrap_empty_region_is_none():
    b = bootstrap_region_object_ci([])
    assert b["n_tiles"] == 0
    assert b["recall"]["point"] is None


def test_bootstrap_deterministic():
    tc = [(1, 0, 1), (0, 1, 0), (1, 0, 0)]
    assert bootstrap_region_object_ci(tc, n_boot=100, seed=7) == \
        bootstrap_region_object_ci(tc, n_boot=100, seed=7)


# ---------------------------------------------------------------------------
# _geometry_summary
# ---------------------------------------------------------------------------


def test_geometry_summary_basic():
    gs = _geometry_summary([0.4, 0.5, 0.6, 0.9])
    assert gs["n_matched"] == 4
    assert gs["iou_median"] == pytest.approx(0.55)


def test_geometry_summary_empty_none():
    assert _geometry_summary([]) is None


# ---------------------------------------------------------------------------
# build_scorecard — self-check + aggregate
# ---------------------------------------------------------------------------


def _mk_tile(H, W, gt_boxes, pred_boxes):
    lab = np.zeros((H, W), np.int64)
    pr = np.zeros((H, W), np.float32)
    for (r0, r1, c0, c1) in gt_boxes:
        lab[r0:r1, c0:c1] = 1
    for (r0, r1, c0, c1, v) in pred_boxes:
        pr[r0:r1, c0:c1] = v
    return lab, pr


def test_build_scorecard_selfcheck_and_signals():
    H = W = 32
    tids, probs, labels, regions = [], [], [], {}

    def add(tid, region, gt, pred):
        lab, pr = _mk_tile(H, W, gt, pred)
        tids.append(tid); regions[tid] = region
        labels.append(lab); probs.append(pr)

    add("A1", "RegA", [(4, 12, 4, 12)], [(4, 12, 4, 12, 0.9)])                    # clean TP
    add("A2", "RegA", [(4, 12, 4, 20)], [(4, 12, 4, 9, 0.9), (4, 12, 13, 20, 0.85)])  # split
    add("A3", "RegA", [(4, 12, 4, 12)], [(4, 12, 4, 12, 0.1)])                    # invisible
    add("B1", "RegB", [(4, 10, 4, 10), (4, 10, 14, 20)], [(4, 10, 4, 20, 0.9)])   # merge
    add("B2", "RegB", [], [(0, 1, 0, 1, 0.95)])                                   # speckle FP

    probs_a = np.stack(probs); labels_a = np.stack(labels)
    tid_region = dict(regions)
    sc = build_scorecard(
        probs_a, labels_a, tids, tid_region,
        ignore_index=255, thr=0.65, min_blob=4, iou_thr=0.3,
        low_thr=0.3, overlap_frac=0.1, n_boot=200, seed=42,
    )
    assert sc["self_check"]["detail_vs_score_by_region_counts_match"] is True
    agg = sc["aggregate"]
    assert agg["n_gt_objects"] == 5
    assert agg["obj_recall"] == pytest.approx(0.6)        # 3 of 5
    assert agg["obj_precision"] == pytest.approx(0.75)    # 3 of 4
    assert agg["n_splits"] >= 1 and agg["n_merges"] >= 1
    assert agg["typology"]["counts"]["perception_invisible"] >= 1
    # Low-sample regions flagged; per-region typology floor present.
    assert sc["per_region"]["RegA"]["low_sample"] is True
    assert sc["per_region"]["RegA"]["typology"]["invisible_floor"] == pytest.approx(1 / 3, abs=1e-3)


# ---------------------------------------------------------------------------
# sample_region_stratified
# ---------------------------------------------------------------------------


def _meta(rows):
    m = pd.DataFrame(rows, columns=["Tile_ID", "RegionName"])
    m["centroid_lat"] = 70.0; m["centroid_lon"] = 100.0
    m["TrainClass"] = "positive"; m["UIDs"] = ""
    return m


def test_sampler_caps_dense_regions_keeps_sparse():
    rows = [(f"A{i}", "RegA") for i in range(10)] + \
           [(f"B{i}", "RegB") for i in range(3)] + [("C0", "RegC")]
    meta = _meta(rows)
    cand = [r[0] for r in rows]
    out = sample_region_stratified(meta, cand, per_region_cap=4, seed=42)
    counts = meta.set_index("Tile_ID").loc[out, "RegionName"].value_counts().to_dict()
    assert counts["RegA"] == 4          # capped
    assert counts["RegB"] == 3          # kept
    assert counts["RegC"] == 1          # no region dropped
    assert out == sorted(out)
    assert out == sample_region_stratified(meta, cand, 4, 42)   # deterministic


def test_sampler_cap_above_sizes_keeps_all():
    rows = [(f"A{i}", "RegA") for i in range(3)] + [("B0", "RegB")]
    meta = _meta(rows)
    cand = [r[0] for r in rows]
    assert set(sample_region_stratified(meta, cand, 100, 1)) == set(cand)


# ---------------------------------------------------------------------------
# classify_invisible_change (D2 probe)
# ---------------------------------------------------------------------------


def test_change_probe_bright_blank_and_excludes_detected():
    H = W = 20
    lab1, p1 = _mk_tile(H, W, [(4, 8, 4, 8)], [])   # invisible (no pred)
    p1[:] = 0.05
    chg1 = np.zeros((H, W), np.float32); chg1[4:8, 4:8] = 0.9       # bright

    lab2, p2 = _mk_tile(H, W, [(4, 8, 4, 8)], [])
    p2[:] = 0.05
    chg2 = np.zeros((H, W), np.float32)                            # blank

    lab3, p3 = _mk_tile(H, W, [(4, 8, 4, 8)], [(5, 6, 5, 6, 0.9)])  # detected -> excluded
    chg3 = np.zeros((H, W), np.float32)

    probs = np.stack([p1, p2, p3]); labels = np.stack([lab1, lab2, lab3])
    change = np.stack([chg1, chg2, chg3])
    rows, summary = classify_invisible_change(
        probs, labels, change, ["t1", "t2", "t3"],
        invisible_thr=0.30, bright_bg_percentile=90.0,
    )
    assert summary["n_invisible_objects"] == 2                     # t3 excluded
    assert summary["change_blank_fraction"] == pytest.approx(0.5)
    cls = {r["tile_id"]: r["change_class"] for r in rows}
    assert cls["t1"] == "change_bright"
    assert cls["t2"] == "change_blank"


def test_change_probe_no_invisible_objects():
    H = W = 12
    lab, pr = _mk_tile(H, W, [(2, 6, 2, 6)], [(2, 6, 2, 6, 0.9)])   # detected only
    chg = np.zeros((H, W), np.float32)
    _, summary = classify_invisible_change(
        pr[None], lab[None], chg[None], ["t"], invisible_thr=0.30,
    )
    assert summary["n_invisible_objects"] == 0
    assert summary["change_blank_fraction"] is None


# ---------------------------------------------------------------------------
# seed_noise
# ---------------------------------------------------------------------------


def test_seed_noise_stats():
    scs = [
        {"aggregate": {"obj_recall": 0.44, "obj_f1": 0.46, "obj_precision": 0.50}},
        {"aggregate": {"obj_recall": 0.40, "obj_f1": 0.43, "obj_precision": 0.52}},
        {"aggregate": {"obj_recall": 0.47, "obj_f1": 0.48, "obj_precision": 0.49}},
    ]
    sn = seed_noise(scs)
    assert sn["n_seeds"] == 3
    assert sn["obj_recall"]["mean"] == pytest.approx(0.4367, abs=1e-3)
    assert sn["obj_recall"]["spread"] == pytest.approx(0.07)
    assert sn["obj_recall"]["std"] > 0


def test_seed_noise_handles_none_metric():
    scs = [{"aggregate": {"obj_recall": None, "obj_f1": 0.4, "obj_precision": 0.5}},
           {"aggregate": {"obj_recall": 0.4, "obj_f1": 0.42, "obj_precision": 0.5}}]
    sn = seed_noise(scs)
    assert sn["obj_recall"]["values"] == [0.4]      # the None is dropped


# ---------------------------------------------------------------------------
# find_invisible_objects (contact-sheet selection logic)
# ---------------------------------------------------------------------------


def test_find_invisible_objects_selects_only_below_threshold():
    H = W = 16
    lab1, p1 = _mk_tile(H, W, [(2, 6, 2, 6)], [])   # invisible
    p1[:] = 0.05
    lab2, p2 = _mk_tile(H, W, [(2, 6, 2, 6)], [])   # detected
    p2[:] = 0.05; p2[3, 3] = 0.9
    objs = find_invisible_objects(
        np.stack([p1, p2]), np.stack([lab1, lab2]), ["t1", "t2"],
        ignore_index=255, invisible_thr=0.30,
    )
    assert len(objs) == 1
    assert objs[0]["tile_id"] == "t1"
    assert objs[0]["area_px"] == 16
    assert objs[0]["bbox"] == (2, 5, 2, 5)

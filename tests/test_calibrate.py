"""Unit tests for the GPU-free calibration helpers in scripts/calibrate.py.

The forward-pass / model-loading paths need GPU + checkpoints and are exercised
by the live Phase-D run, not here. These cover the math: temperature fitting,
threshold selection, PR-AUC geomean, and precision@threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.calibrate import (
    _logit,
    _sigmoid,
    fit_temperature,
    pr_auc_geomean,
    precision_at_threshold,
    select_threshold,
)


def _make_per_tile(n_pos: int, n_neg: int, pix: int = 2000, seed: int = 0) -> dict:
    """Synthetic per-tile dict: positive tiles have separable signal, negatives noise."""
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n_pos):
        labels = (rng.random(pix) < 0.3).astype(np.uint8)
        probs = np.where(labels == 1, rng.uniform(0.6, 0.99, pix), rng.uniform(0.0, 0.4, pix))
        out[f"pos{i}"] = {"probs": probs.astype(np.float32), "labels": labels, "is_pos": True}
    for i in range(n_neg):
        out[f"neg{i}"] = {"probs": rng.uniform(0.0, 0.3, pix).astype(np.float32),
                          "labels": np.zeros(pix, np.uint8), "is_pos": False}
    return out


def test_logit_sigmoid_roundtrip():
    p = np.array([0.01, 0.25, 0.5, 0.75, 0.99], dtype=np.float32)
    assert np.allclose(_sigmoid(_logit(p)), p, atol=1e-5)


def test_fit_temperature_recovers_known_scaling():
    # Build logits whose calibrated form needs T=2 to match the labels' frequency.
    rng = np.random.default_rng(1)
    true_logits = rng.normal(0, 2.0, 200_000)
    probs = _sigmoid(true_logits)            # well-calibrated at T=1
    labels = (rng.random(probs.shape) < probs).astype(np.uint8)
    overconfident = true_logits * 2.0        # inflate => optimal T ~ 2
    T = fit_temperature(overconfident, labels)
    assert 1.6 < T < 2.4, f"expected ~2.0, got {T}"


def test_fit_temperature_bounds():
    logits = np.array([5.0, -5.0, 3.0, -3.0] * 1000)
    labels = np.array([1, 0, 1, 0] * 1000, dtype=np.uint8)
    T = fit_temperature(logits, labels)
    assert 0.25 <= T <= 5.0


def test_pr_auc_geomean_separable_is_high():
    per_tile = _make_per_tile(n_pos=20, n_neg=400, seed=2)
    gm, per_r = pr_auc_geomean(per_tile, ratios=[5, 10, 20])
    assert gm > 0.8
    assert set(per_r) == {5, 10, 20}
    # higher prevalence ratio (more negatives) cannot increase PR-AUC
    assert per_r[5] >= per_r[20] - 1e-6


def test_pr_auc_geomean_empty_returns_zero():
    only_neg = {"neg0": {"probs": np.zeros(10, np.float32), "labels": np.zeros(10, np.uint8),
                         "is_pos": False}}
    gm, per_r = pr_auc_geomean(only_neg, ratios=[5, 10])
    assert gm == 0.0


def test_precision_at_threshold_monotone():
    per_tile = _make_per_tile(n_pos=10, n_neg=50, seed=3)
    p_lo = precision_at_threshold(per_tile, 0.3)
    p_hi = precision_at_threshold(per_tile, 0.7)
    assert 0.0 <= p_lo <= 1.0 and 0.0 <= p_hi <= 1.0
    assert p_hi >= p_lo - 1e-6  # raising the threshold should not lower precision here


def test_select_threshold_meets_target_when_separable():
    per_tile = _make_per_tile(n_pos=20, n_neg=400, seed=4)
    res = select_threshold(per_tile, ratio=10, target_precision=0.8)
    assert res["target_met"] is True
    assert res["precision"] >= 0.8 - 1e-9
    assert 0.0 < res["threshold"] < 1.0


def test_select_threshold_falls_back_to_f1_when_unreachable():
    # Unseparable: probs independent of labels => precision target 0.99 unreachable.
    rng = np.random.default_rng(5)
    per_tile = {}
    for i in range(10):
        per_tile[f"pos{i}"] = {"probs": rng.uniform(0, 1, 1000).astype(np.float32),
                               "labels": (rng.random(1000) < 0.2).astype(np.uint8), "is_pos": True}
    for i in range(40):
        per_tile[f"neg{i}"] = {"probs": rng.uniform(0, 1, 1000).astype(np.float32),
                               "labels": np.zeros(1000, np.uint8), "is_pos": False}
    res = select_threshold(per_tile, ratio=10, target_precision=0.99)
    assert res["target_met"] is False
    assert 0.0 <= res["threshold"] <= 1.0

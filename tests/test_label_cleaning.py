"""Unit tests for data.label_cleaning.clean_positive_label (data-v1.1 label hygiene)."""

from __future__ import annotations

import numpy as np

from data.label_cleaning import clean_positive_label


def test_removes_sub_mmu_sliver_to_ignore():
    """A blob smaller than min_size becomes 255 (ignore), not 0 (background)."""
    lab = np.zeros((20, 20), np.int64)
    lab[2:8, 2:8] = 1          # 36-px real blob
    lab[15, 15] = 1            # 1-px sliver
    out, st = clean_positive_label(lab, min_size=10, close_radius=0, fill_holes=False)
    assert out[15, 15] == 255              # sliver → ignore
    assert (out[2:8, 2:8] == 1).all()      # real blob kept
    assert st.n_removed_blobs == 1
    assert st.px_removed_to_ignore == 1
    assert st.n_blobs_after == 1


def test_preserves_existing_ignore():
    lab = np.zeros((12, 12), np.int64)
    lab[2:6, 2:6] = 1
    lab[0, 0] = 255            # pre-existing ignore
    out, _ = clean_positive_label(lab, min_size=4, close_radius=0)
    assert out[0, 0] == 255


def test_fill_holes():
    lab = np.zeros((12, 12), np.int64)
    lab[2:10, 2:10] = 1
    lab[5, 5] = 0             # interior hole
    out, st = clean_positive_label(lab, min_size=4, close_radius=0, fill_holes=True)
    assert out[5, 5] == 1
    assert st.px_holes_filled == 1


def test_bridge_merges_fragments_via_close():
    """Two fragments of one object separated by a 1-px gap merge under the 1px close."""
    lab = np.zeros((10, 20), np.int64)
    lab[2:6, 2:8] = 1         # left fragment (24 px)
    lab[2:6, 9:15] = 1        # right fragment (24 px), gap at col 8
    out, st = clean_positive_label(lab, min_size=4, close_radius=1, fill_holes=False)
    assert st.px_bridged >= 1
    # After bridging it's a single connected component.
    from scipy import ndimage
    assert ndimage.label(out == 1)[1] == 1


def test_order_synergy_keeps_bridged_pair_removes_isolated():
    """Two <T fragments that MERGE above T are kept; a truly isolated <T sliver is removed."""
    lab = np.zeros((12, 30), np.int64)
    # Two 6-px fragments 1px apart → merge to ~13px (≥ T=10) → kept.
    lab[2:5, 2:4] = 1
    lab[2:5, 5:7] = 1
    # Isolated 2-px sliver far away → removed.
    lab[10, 20:22] = 1
    out, st = clean_positive_label(lab, min_size=10, close_radius=1, fill_holes=False)
    assert (out[2:5, 2:7] == 1).any()        # merged pair survives
    assert out[10, 20] == 255 and out[10, 21] == 255   # isolated sliver → ignore
    assert st.n_blobs_after == 1


def test_does_not_mutate_input():
    lab = np.zeros((8, 8), np.int64)
    lab[1, 1] = 1
    snapshot = lab.copy()
    clean_positive_label(lab, min_size=10, close_radius=0)
    assert np.array_equal(lab, snapshot)


def test_all_background_is_noop():
    lab = np.zeros((6, 6), np.int64)
    out, st = clean_positive_label(lab, min_size=10)
    assert (out == 0).all()
    assert st.n_blobs_before == 0 and st.n_removed_blobs == 0

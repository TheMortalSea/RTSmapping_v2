"""Unit tests for scripts/package_model._assert_calibration_complete.

Full end-to-end packaging requires an MLflow run, which is exercised via the
train-smoke test. Here we just verify the calibration-guard contract: a
deployment config with null threshold or temperature must be rejected
(plan Step 8 gate).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from package_model import _assert_calibration_complete  # noqa: E402


def test_null_threshold_rejected():
    cfg = {"threshold": None, "temperature": 1.2}
    with pytest.raises(ValueError, match="threshold"):
        _assert_calibration_complete(cfg)


def test_null_temperature_rejected():
    cfg = {"threshold": 0.5, "temperature": None}
    with pytest.raises(ValueError, match="temperature"):
        _assert_calibration_complete(cfg)


def test_both_null_rejected_together():
    cfg = {"threshold": None, "temperature": None}
    with pytest.raises(ValueError, match="threshold"):
        _assert_calibration_complete(cfg)


def test_both_set_accepted():
    cfg = {"threshold": 0.5, "temperature": 1.2}
    # Should not raise.
    _assert_calibration_complete(cfg)


# --- run-dir packaging (no-MLflow path, Phase 2) --------------------------
def _make_run_dir(tmp_path):
    """A minimal fake training run dir + norm stats + deployment config."""
    import torch
    import yaml

    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    torch.save({"model_state_dict": {"w": torch.zeros(2)}, "epoch": 40,
                "best_metric": 0.92, "git_sha": "abc123",
                "channel_names": ["R", "G", "B", "ndvi"],
                "trained_with": {"seed": 43, "config_sha": "deadbeef"}},
               run / "checkpoints" / "best_deployment.pth")
    (run / "config.yaml").write_text(yaml.safe_dump({
        "seed": 43,
        "model": {"architecture": "unetplusplus", "backbone": "efficientnet-b5"},
        "channels": {"rgb": True, "extra": [{"name": "ndvi", "band": 0}]},
        "data": {"tile_size": 512, "crs": "EPSG:3857", "label_ignore_index": 255},
        "loss": {"boundary_handling": "none", "boundary_ignore_width": 3},
    }))
    (run / "requirements_frozen.txt").write_text("torch==2.0\n")
    ns = tmp_path / "normalization_stats_ndvi.json"
    ns.write_text('{"rgb": {"channel_names": ["R","G","B"]}, '
                  '"extra": {"channel_names": ["ndvi"]}}')
    dep = tmp_path / "deployment.yaml"
    dep.write_text("threshold: 0.65\ntemperature: 0.512321\ntta: none\n")
    return run, ns, dep


def test_package_from_rundir_writes_all_files(tmp_path):
    import json

    from package_model import package_model_from_rundir

    run, ns, dep = _make_run_dir(tmp_path)
    out = tmp_path / "pkg"
    package_model_from_rundir(run, ns, dep, out)

    for f in ("weights.pth", "normalization_stats.json", "model_config.yaml",
              "deployment_config.yaml", "run_metadata.json", "requirements_frozen.txt"):
        assert (out / f).exists(), f"missing {f}"

    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["source"] == "run_dir"
    assert meta["seed"] == 43
    assert meta["git_sha"] == "abc123"
    assert meta["channel_names"] == ["R", "G", "B", "ndvi"]
    assert meta["checkpoint_epoch"] == 40


def test_package_from_rundir_rejects_uncalibrated(tmp_path):
    from package_model import package_model_from_rundir

    run, ns, dep = _make_run_dir(tmp_path)
    dep.write_text("threshold: null\ntemperature: null\n")
    with pytest.raises(ValueError, match="threshold"):
        package_model_from_rundir(run, ns, dep, tmp_path / "pkg")


def test_package_from_rundir_missing_input_raises(tmp_path):
    from package_model import package_model_from_rundir

    run, ns, dep = _make_run_dir(tmp_path)
    (run / "checkpoints" / "best_deployment.pth").unlink()
    with pytest.raises(FileNotFoundError, match="best_deployment"):
        package_model_from_rundir(run, ns, dep, tmp_path / "pkg")

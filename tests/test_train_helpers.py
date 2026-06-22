"""Unit tests for scripts/train.py module-level helpers (CPU, no training loop)."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from train import _deploy_state_dict  # noqa: E402


class _FakeEMA:
    """Minimal EMA stub: swap_in temporarily overwrites every param with a sentinel, then
    restores — mirrors EMAModel.swap_in's contract well enough to test the selection logic."""

    def __init__(self, value: float):
        self.value = value

    def swap_in(self, model):
        @contextlib.contextmanager
        def _ctx():
            orig = {k: v.detach().clone() for k, v in model.state_dict().items()}
            with torch.no_grad():
                for p in model.parameters():
                    p.fill_(self.value)
            try:
                yield
            finally:
                model.load_state_dict(orig)
        return _ctx()


def _model() -> torch.nn.Module:
    m = torch.nn.Conv2d(3, 1, 1)
    with torch.no_grad():
        m.weight.fill_(0.5)
    return m


def test_deploy_state_dict_live_weights_when_no_ema():
    """ema=None → live weights (the freeze-phase / permanently-frozen-probe path that would
    otherwise never write a deployment checkpoint)."""
    model = _model()
    state = _deploy_state_dict(model, ema=None)
    assert torch.allclose(state["weight"], torch.full_like(model.weight, 0.5))


def test_deploy_state_dict_uses_ema_and_restores_model():
    """ema present → captures the swapped-in EMA weights; the live model is restored after."""
    model = _model()
    state = _deploy_state_dict(model, ema=_FakeEMA(9.0))
    assert torch.allclose(state["weight"], torch.full_like(model.weight, 9.0))  # EMA captured
    assert torch.allclose(model.weight, torch.full_like(model.weight, 0.5))     # model restored

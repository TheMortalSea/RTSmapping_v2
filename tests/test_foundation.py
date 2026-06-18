"""Forward-path tests for the foundation-encoder segmenter (models/foundation.py).

CPU-only, pretrained=False (no weight download). Validates the ViT
get_intermediate_layers → simple feature pyramid → FPN decoder → logits chain and the
compatibility hooks (.encoder for freeze/LLRD, .segmentation_head[0] for bias-init).
"""

from __future__ import annotations

import math

import torch

from models.foundation import FoundationSegmenter
from models.segmentation import _init_output_bias

BACKBONE = "vit_base_patch16_dinov3"


def test_foundation_forward_shape():
    """ViT encoder → (B, 1, H, W) logits at input resolution."""
    model = FoundationSegmenter(BACKBONE, pretrained=False).eval()
    x = torch.zeros(2, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 64, 64)


def test_foundation_taps_four_blocks_incl_deepest():
    model = FoundationSegmenter(BACKBONE, pretrained=False)
    depth = len(model.encoder.blocks)
    assert len(model.tap_indices) == 4
    assert model.tap_indices[-1] == depth - 1
    assert model.tap_indices == sorted(model.tap_indices)


def test_foundation_exposes_encoder_and_head():
    """freeze.py needs .encoder; _init_output_bias needs .segmentation_head[0].bias."""
    model = FoundationSegmenter(BACKBONE, pretrained=False)
    assert hasattr(model, "encoder")
    assert isinstance(model.segmentation_head[0], torch.nn.Conv2d)
    _init_output_bias(model, 0.01)
    assert math.isclose(model.segmentation_head[0].bias.detach().item(),
                        -math.log((1.0 - 0.01) / 0.01), abs_tol=1e-5)


def test_foundation_output_is_logits():
    model = FoundationSegmenter(BACKBONE, pretrained=False).eval()
    torch.manual_seed(0)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.min().item() < 0.0 or y.max().item() > 1.0

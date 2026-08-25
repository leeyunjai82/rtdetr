# Apache-2.0
"""The network itself: shapes, variants, and the export path for deformable attention."""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import needs_ov, needs_torch

pytestmark = [needs_torch]


@pytest.mark.parametrize(
    ("variant", "decoder_layers", "millions"),
    [("r18", 3, 20.2), ("r34", 4, 31.3), ("r50", 6, 42.9)],
)
def test_each_variant_matches_the_published_shape(variant, decoder_layers, millions):
    from rtdetr.nn import RTDETRNet

    net = RTDETRNet(variant, num_classes=80, pretrained_backbone=False)
    assert len(net.decoder.decoder.layers) == decoder_layers
    assert len(net.decoder.dec_score_head) == decoder_layers
    params = sum(p.numel() for p in net.parameters()) / 1e6
    assert params == pytest.approx(millions, abs=0.3)


def test_eval_returns_the_last_layer_and_training_returns_every_layer():
    import torch

    from rtdetr.nn import RTDETRNet

    net = RTDETRNet("r18", num_classes=5, pretrained_backbone=False)
    x = torch.zeros(1, 3, 128, 128)

    net.eval()
    with torch.no_grad():
        out = net(x)
    assert out["pred_logits"].shape == (1, 300, 5) and out["pred_boxes"].shape == (1, 300, 4)
    assert "aux_outputs" not in out

    net.train()
    out = net(x)
    # 2 intermediate decoder layers + the encoder's own top-k proposals
    assert len(out["aux_outputs"]) == 3
    assert out["aux_outputs"][0]["pred_logits"].shape == (1, 300, 5)


def test_an_input_too_small_for_300_queries_still_runs():
    """8x8 + 4x4 + 2x2 tokens is fewer than num_queries — must not blow up."""
    import torch

    from rtdetr.nn import RTDETRNet

    net = RTDETRNet("r18", num_classes=2, pretrained_backbone=False).eval()
    with torch.no_grad():
        out = net(torch.zeros(1, 3, 64, 64))
    assert out["pred_logits"].shape == (1, 84, 2)


def test_unknown_variants_are_rejected():
    from rtdetr.nn import RTDETRNet

    with pytest.raises(ValueError, match="r18"):
        RTDETRNet("r101", num_classes=80, pretrained_backbone=False)


@needs_ov
def test_deformable_attention_survives_the_trip_to_openvino(tmp_path):
    """grid_sample is the one op the export path has to get right."""
    import openvino as ov
    import torch

    from rtdetr.nn.decoder import MSDeformableAttention

    attn = MSDeformableAttention(embed_dim=64, num_heads=4, num_levels=3, num_points=4).eval()
    shapes = [(8, 8), (4, 4), (2, 2)]
    q = torch.randn(1, 20, 64)
    ref = torch.rand(1, 20, 3, 4)
    value = torch.randn(1, sum(h * w for h, w in shapes), 64)

    class Wrapper(torch.nn.Module):
        def __init__(self, attn):
            super().__init__()
            self.attn = attn

        def forward(self, q, ref, value):
            return self.attn(q, ref, value, shapes)

    with torch.no_grad():
        expected = Wrapper(attn)(q, ref, value)
    onnx_path = tmp_path / "attn.onnx"
    torch.onnx.export(
        Wrapper(attn), (q, ref, value), str(onnx_path), opset_version=16,
        input_names=["q", "ref", "value"], output_names=["out"], dynamo=False,
    )
    compiled = ov.Core().compile_model(ov.convert_model(str(onnx_path)), "CPU")
    got = compiled([q.numpy(), ref.numpy(), value.numpy()])[compiled.output(0)]
    assert np.abs(got - expected.numpy()).max() < 1e-4

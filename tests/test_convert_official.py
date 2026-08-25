# Apache-2.0
"""tools/convert_official.py — the key mapping, and one full conversion."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from .conftest import needs_torch

TOOL = Path(__file__).resolve().parents[1] / "tools" / "convert_official.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("convert_official", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("official", "ours"),
    [
        ("backbone.res_layers.0.blocks.1.branch2a.conv.weight", "backbone.layer1.1.conv1.weight"),
        ("backbone.res_layers.3.blocks.0.short.norm.bias", "backbone.layer4.0.downsample.1.bias"),
        ("encoder.input_proj.0.0.weight", "encoder.input_proj.0.0.weight"),
        (
            "encoder.encoder.0.layers.0.self_attn.in_proj_weight",
            "encoder.aifi.encoder.layers.0.self_attn.in_proj_weight",
        ),
        ("encoder.lateral_convs.1.norm.weight", "encoder.lateral2.bn.weight"),
        ("encoder.downsample_convs.0.conv.weight", "encoder.down1.conv.weight"),
        ("decoder.decoder.layers.2.linear1.weight", "decoder.layers.2.ffn.0.weight"),
        ("decoder.dec_score_head.5.bias", "decoder.dec_score.5.bias"),
        ("decoder.enc_bbox_head.layers.0.weight", "decoder.enc_bbox.layers.0.weight"),
        ("decoder.enc_output.1.weight", "decoder.enc_norm.weight"),
    ],
)
def test_official_layer_names_map_onto_ours(tool, official, ours):
    assert tool.rename(official) == ours


@pytest.mark.parametrize(
    "official",
    [
        "backbone.conv1.conv1_1.conv.weight",  # vd deep stem, we use a 7x7
        "encoder.fpn_blocks.0.conv1.conv.weight",  # CSPRepLayer is wider than our RepBlock
        "decoder.decoder.layers.0.cross_attn.sampling_offsets.weight",  # deformable
        "something.unexpected",
    ],
)
def test_modules_with_no_counterpart_are_reported_not_guessed(tool, official):
    assert tool.rename(official) is None


def test_coco_names_are_the_default_and_come_out_in_order(tool):
    assert len(tool.COCO_NAMES) == 80
    assert tool.COCO_NAMES[0] == "person" and tool.COCO_NAMES[-1] == "toothbrush"
    assert tool._read_names(None) == tool.COCO_NAMES


def test_names_can_be_supplied_as_labels_txt_or_json(tool, tmp_path):
    (tmp_path / "labels.txt").write_text("can\nbottle\n")
    assert tool._read_names(str(tmp_path / "labels.txt")) == ["can", "bottle"]
    (tmp_path / "names.json").write_text('{"1": "b", "0": "a"}')
    assert tool._read_names(str(tmp_path / "names.json")) == ["a", "b"]


@needs_torch
def test_a_full_conversion_lands_the_weights_it_promises(tool, tmp_path, capsys):
    """Feed it an upstream-shaped state_dict and check what survives the trip."""
    import torch

    from rtdetr.nn.rtdetr_net import RTDETRNet

    ours = RTDETRNet("r18", num_classes=80, pretrained_backbone=False).state_dict()
    official = {}
    for key, tensor in ours.items():
        upstream = _to_official(key)
        if upstream:
            official[upstream] = tensor
    torch.save({"ema": {"module": official}}, tmp_path / "official.pth")

    out = tmp_path / "rtdetr-r18.pt"
    assert (
        tool.main(
            [
                "--weights", str(tmp_path / "official.pth"),
                "--variant", "r18",
                "--out", str(out),
                "--min-coverage", "50",
            ]
        )
        == 0
    )
    assert "transferred" in capsys.readouterr().out

    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert ckpt["variant"] == "r18" and ckpt["num_classes"] == 80
    assert ckpt["names"][0] == "person"

    from rtdetr import RTDETR

    model = RTDETR(str(out), verbose=False)
    assert model.names[2] == "car"
    moved = model.net.state_dict()["backbone.layer1.0.conv1.weight"]
    assert torch.equal(moved, ours["backbone.layer1.0.conv1.weight"])


@needs_torch
def test_a_renamed_upstream_release_fails_the_coverage_gate(tool, tmp_path, capsys):
    import torch

    torch.save({"model": {"totally.new.name": torch.zeros(1)}}, tmp_path / "official.pth")
    code = tool.main(
        [
            "--weights", str(tmp_path / "official.pth"),
            "--out", str(tmp_path / "out.pt"),
            "--min-coverage", "50",
        ]
    )
    assert code == 1 and "below --min-coverage" in capsys.readouterr().err


def _to_official(key: str) -> str | None:
    """The inverse of ``rename`` — used to fabricate an upstream checkpoint."""
    m = re.match(r"backbone\.layer(\d)\.(\d+)\.(.+)", key)
    if m:
        stage, block, rest = int(m.group(1)) - 1, m.group(2), m.group(3)
        rest = (
            rest.replace("conv1", "branch2a.conv")
            .replace("bn1", "branch2a.norm")
            .replace("conv2", "branch2b.conv")
            .replace("bn2", "branch2b.norm")
            .replace("conv3", "branch2c.conv")
            .replace("bn3", "branch2c.norm")
            .replace("downsample.0", "short.conv")
            .replace("downsample.1", "short.norm")
        )
        return f"backbone.res_layers.{stage}.blocks.{block}.{rest}"
    if key.startswith("encoder.input_proj."):
        return key
    m = re.match(r"encoder\.aifi\.encoder\.layers\.(\d+)\.(.+)", key)
    if m:
        return f"encoder.encoder.0.layers.{m.group(1)}.{m.group(2)}"
    m = re.match(r"encoder\.lateral(\d)\.(conv|bn)\.(.+)", key)
    if m:
        norm = "conv" if m.group(2) == "conv" else "norm"
        return f"encoder.lateral_convs.{int(m.group(1)) - 1}.{norm}.{m.group(3)}"
    m = re.match(r"encoder\.down(\d)\.(conv|bn)\.(.+)", key)
    if m:
        norm = "conv" if m.group(2) == "conv" else "norm"
        return f"encoder.downsample_convs.{int(m.group(1)) - 1}.{norm}.{m.group(3)}"
    m = re.match(r"decoder\.layers\.(\d+)\.(.+)", key)
    if m:
        rest = m.group(2).replace("ffn.0.", "linear1.").replace("ffn.2.", "linear2.")
        return f"decoder.decoder.layers.{m.group(1)}.{rest}"
    for ours, upstream in (
        ("decoder.dec_score.", "decoder.dec_score_head."),
        ("decoder.dec_bbox.", "decoder.dec_bbox_head."),
        ("decoder.enc_score.", "decoder.enc_score_head."),
        ("decoder.enc_bbox.", "decoder.enc_bbox_head."),
        ("decoder.tgt_proj.", "decoder.enc_output.0."),
        ("decoder.enc_norm.", "decoder.enc_output.1."),
    ):
        if key.startswith(ours):
            return key.replace(ours, upstream)
    if key.startswith("decoder.query_pos_head."):
        return key
    return None

# Apache-2.0
"""tools/convert_official.py — importing a released RT-DETR checkpoint."""

from __future__ import annotations

import importlib.util
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


def test_every_variant_knows_which_release_to_pull(tool):
    assert set(tool.OFFICIAL) == {"r18", "r34", "r50"}
    assert all(name.endswith(".pth") for name in tool.OFFICIAL.values())


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
@pytest.mark.parametrize("wrapper", ["ema", "model", "bare"])
def test_the_state_dict_is_found_whichever_container_the_release_used(tool, tmp_path, wrapper):
    import torch

    tensors = {"backbone.conv1.conv1_1.conv.weight": torch.zeros(2, 2)}
    blob = {
        "ema": {"ema": {"module": tensors}},
        "model": {"model": tensors},
        "bare": tensors,
    }[wrapper]
    torch.save(blob, tmp_path / "ckpt.pth")
    assert list(tool.official_state_dict(tmp_path / "ckpt.pth")) == list(tensors)


@needs_torch
def test_a_release_shaped_checkpoint_loads_whole_and_carries_our_metadata(tool, tmp_path, capsys):
    """The point of matching the reference layout: strict load, no remapping."""
    import torch

    from rtdetr.nn import RTDETRNet

    upstream = RTDETRNet("r18", num_classes=80, pretrained_backbone=False).state_dict()
    torch.save({"ema": {"module": upstream}}, tmp_path / "official.pth")

    out = tmp_path / "rtdetr-r18.pt"
    assert tool.main(["--variant", "r18", "--weights", str(tmp_path / "official.pth"),
                      "--out", str(out)]) == 0
    assert "loaded all" in capsys.readouterr().out

    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert ckpt["variant"] == "r18" and ckpt["num_classes"] == 80
    assert ckpt["names"][2] == "car" and ckpt["imgsz"] == 640
    assert "Apache-2.0" in ckpt["source"]

    from rtdetr import RTDETR

    model = RTDETR(str(out), verbose=False)
    key = "backbone.res_layers.0.blocks.0.branch2a.conv.weight"
    assert torch.equal(model.net.state_dict()[key], upstream[key])


@needs_torch
def test_a_checkpoint_that_is_not_this_architecture_fails_loudly(tool, tmp_path):
    import torch

    torch.save({"model": {"totally.new.name": torch.zeros(1)}}, tmp_path / "official.pth")
    with pytest.raises(RuntimeError, match="Unexpected key|Missing key"):
        tool.main(["--variant", "r18", "--weights", str(tmp_path / "official.pth"),
                   "--out", str(tmp_path / "out.pt")])

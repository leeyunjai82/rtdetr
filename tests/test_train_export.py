# Apache-2.0
"""Training, validating and exporting for real — tiny, but the whole pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from rtdetr import RTDETR
from rtdetr.metrics import DetMetrics

from .conftest import needs_ov, needs_torch

pytestmark = [needs_torch]

TRAIN_KWARGS = dict(epochs=1, imgsz=64, batch=2, workers=0, device="cpu", val=False, amp=False)


@pytest.fixture(scope="module")
def trained(dataset, tmp_path_factory):
    """One real (one-epoch, 64px) training run, reused by the tests below."""
    model = RTDETR("rtdetr-r18", verbose=False)
    model.net = _blank_net(1)
    runs = str(tmp_path_factory.mktemp("runs"))
    best = model.train(data=str(dataset), project=runs, **TRAIN_KWARGS)
    return model, best


def _blank_net(nc):
    from rtdetr.nn.rtdetr_net import RTDETRNet

    return RTDETRNet("r18", nc, pretrained_backbone=False)


def test_training_writes_last_and_best_with_everything_needed_to_resume(trained):
    import torch

    _, best = trained
    weights = best.parent
    assert best.exists() and (weights / "last.pt").exists()
    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    assert ckpt["variant"] == "r18" and ckpt["num_classes"] == 1
    assert ckpt["names"] == {0: "box"} and ckpt["imgsz"] == 64
    assert "optimizer" in ckpt and ckpt["epoch"] == 0


def test_the_model_reloads_its_own_checkpoint(trained):
    _, best = trained
    reloaded = RTDETR(str(best), verbose=False)
    assert reloaded.variant == "r18" and reloaded.names == {0: "box"}
    assert "1 classes" in reloaded.info()


def test_val_reports_map_in_the_shape_yolo_users_read(trained, dataset):
    model, _ = trained
    metrics = model.val(data=str(dataset), imgsz=64, batch=2, device="cpu", workers=0)
    assert isinstance(metrics, DetMetrics)
    assert 0.0 <= metrics.box.map50 <= 1.0 and 0.0 <= metrics.box.map <= 1.0
    assert metrics["map50"] == metrics.box.map50  # dict-style access still works


def test_resuming_continues_the_same_run_directory(dataset, tmp_path):
    model = RTDETR("rtdetr-r18", verbose=False)
    model.net = _blank_net(1)
    first = model.train(data=str(dataset), project=str(tmp_path), **TRAIN_KWARGS)

    resumed = RTDETR(str(first), verbose=False)
    again = resumed.train(
        data=str(dataset), project=str(tmp_path), resume=True, **{**TRAIN_KWARGS, "epochs": 2}
    )
    assert again.parent == first.parent  # same runs/train dir, not train2

    import torch

    assert torch.load(again, map_location="cpu", weights_only=False)["epoch"] == 1


def test_a_dataset_with_different_classes_re_heads_the_network(trained):
    from rtdetr.model import transfer_weights

    model, _ = trained
    fresh = _blank_net(7)
    moved = transfer_weights(fresh, model.net.state_dict())
    assert moved > 100  # backbone/encoder came across
    assert fresh.decoder.dec_score[0].out_features == 7


def test_early_stopping_gives_up_after_patience_epochs(dataset, tmp_path, capsys):
    model = RTDETR("rtdetr-r18", verbose=False)
    model.net = _blank_net(1)
    model.train(
        data=str(dataset),
        project=str(tmp_path),
        epochs=6,
        imgsz=64,
        batch=2,
        workers=0,
        device="cpu",
        amp=False,
        patience=1,
        val=True,
    )
    assert "early stop" in capsys.readouterr().out


def test_training_falls_back_to_an_imagenet_start_when_the_mirror_is_unreachable(
    dataset, tmp_path, monkeypatch
):
    """No mirror, no crash: the run just starts from a fresh backbone."""
    from rtdetr import downloads
    from rtdetr.errors import ModelNotFoundError as NotFound

    monkeypatch.setattr(
        downloads,
        "download_checkpoint",
        lambda name: (_ for _ in ()).throw(NotFound("offline")),
    )
    monkeypatch.setattr(
        "rtdetr.nn.backbone.ResNet._load_imagenet", lambda self, variant: None
    )
    model = RTDETR("rtdetr-r18", verbose=False)
    best = model.train(data=str(dataset), project=str(tmp_path), **TRAIN_KWARGS)
    assert best.exists() and model.net.num_classes == 1


def test_export_writes_onnx_next_to_the_checkpoint(trained):
    model, best = trained
    onnx = model.export(format="onnx", imgsz=64)
    assert onnx.exists() and onnx.parent == best.parent
    assert (best.parent / "labels.txt").read_text().splitlines() == ["box"]


@needs_ov
def test_export_writes_an_ir_that_predicts(trained, tmp_path, image):
    model, _ = trained
    xml = model.export(format="openvino", imgsz=64, half=True, out_dir=tmp_path)
    assert xml.exists() and xml.with_suffix(".bin").exists()
    assert (tmp_path / "labels.txt").read_text().splitlines() == ["box"]

    results = RTDETR(str(xml), device="CPU", verbose=False)(image, conf=0.0, max_det=2)
    assert results[0].names == {0: "box"}
    assert results[0].boxes.xyxy.shape == (2, 4)


@needs_ov
def test_predicting_from_a_pt_exports_an_ir_behind_the_scenes(
    trained, image, tmp_path, monkeypatch
):
    monkeypatch.setenv("RTDETR_HOME", str(tmp_path / "cache"))
    model = RTDETR(str(trained[1]), device="CPU", verbose=False)
    results = model(image, conf=0.0, max_det=1)
    assert model.ir_path is not None and model.ir_path.exists()
    assert results[0].names == {0: "box"}


@needs_ov
def test_a_trained_model_still_decodes_its_own_probabilities_once(trained, tmp_path):
    """End to end: the IR emits probabilities, so conf filtering must be honest."""
    model, _ = trained
    xml = model.export(format="openvino", imgsz=64, out_dir=tmp_path / "ir")
    from rtdetr.predictor import OVPredictor

    predictor = OVPredictor(xml, device="CPU")
    _, scores = predictor.infer(predictor.preprocess(np.zeros((64, 64, 3), np.uint8)))
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    det, _ = predictor(np.zeros((64, 64, 3), np.uint8), conf=float(scores.max()) + 1e-6)
    assert len(det) == 0

# Apache-2.0
"""Decoding rules — including the sigmoid bug that cost us an afternoon once."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rtdetr.predictor import _looks_like_logits, read_names
from rtdetr.utils.ops import cxcywh2xyxy_np

from .conftest import needs_ov, needs_torch


class _Decoder:
    """postprocess() without compiling anything — it is a pure function."""

    from rtdetr.predictor import OVPredictor

    postprocess = OVPredictor.postprocess


def decode(boxes, scores, shape=(100, 100), **kwargs):
    return _Decoder.postprocess(_Decoder(), boxes, scores, shape, **kwargs)


def test_probabilities_from_the_exported_ir_are_not_sigmoided_twice():
    """sigmoid(0.95) = 0.72 — double-squashing silently breaks conf thresholds."""
    boxes = np.full((4, 4), 0.5, np.float32)
    scores = np.zeros((4, 3), np.float32)
    scores[0, 1] = 0.95
    det = decode(boxes, scores, conf=0.9)
    assert len(det) == 1 and det[0, 4] == pytest.approx(0.95)


def test_raw_logits_still_get_their_sigmoid():
    boxes = np.full((4, 4), 0.5, np.float32)
    scores = np.full((4, 3), -8.0, np.float32)
    scores[0, 1] = 3.0
    det = decode(boxes, scores, conf=0.9)
    assert len(det) == 1 and det[0, 4] == pytest.approx(1 / (1 + np.exp(-3.0)), abs=1e-4)
    assert _looks_like_logits(scores) and not _looks_like_logits(np.zeros((2, 2), np.float32))


def test_boxes_come_back_in_pixels_sorted_by_confidence_and_clipped():
    boxes = np.array([[0.5, 0.5, 0.4, 0.4], [0.1, 0.1, 0.6, 0.6]], np.float32)
    scores = np.array([[0.4, 0.0], [0.9, 0.0]], np.float32)
    det = decode(boxes, scores, shape=(200, 100), conf=0.1)
    assert det[:, 4].tolist() == pytest.approx([0.9, 0.4])
    assert det[:, 0].min() >= 0 and det[:, 2].max() <= 100 and det[:, 3].max() <= 200
    assert det[1, :4].tolist() == pytest.approx([30.0, 60.0, 70.0, 140.0])


def test_conf_max_det_and_classes_all_filter():
    boxes = np.full((5, 4), 0.5, np.float32)
    scores = np.array([[0.9, 0.1], [0.8, 0.1], [0.1, 0.7], [0.05, 0.0], [0.6, 0.0]], np.float32)
    assert len(decode(boxes, scores, conf=0.5)) == 4
    assert len(decode(boxes, scores, conf=0.5, max_det=2)) == 2
    only_car = decode(boxes, scores, conf=0.5, classes=[1])
    assert only_car[:, 5].tolist() == [1.0]


def test_cxcywh_to_xyxy_matches_the_hand_worked_example():
    box = np.array([[0.5, 0.5, 0.2, 0.4]], np.float32)
    assert cxcywh2xyxy_np(box) == pytest.approx(np.array([[0.4, 0.3, 0.6, 0.7]]))


def test_class_names_are_read_from_labels_txt_or_the_json_sidecar(tmp_path):
    (tmp_path / "labels.txt").write_text("can\nbottle\n")
    assert read_names(tmp_path / "m.xml") == {0: "can", 1: "bottle"}
    (tmp_path / "m.names.json").write_text(json.dumps({"0": "cat", "1": "dog"}))
    assert read_names(tmp_path / "m.xml") == {0: "cat", 1: "dog"}
    assert read_names(tmp_path / "elsewhere" / "m.xml") == {}


@needs_torch
@needs_ov
def test_a_real_ir_round_trips_from_pixels_to_detections(tiny_ir):
    from rtdetr.predictor import OVPredictor

    predictor = OVPredictor(tiny_ir, device="CPU")
    assert predictor.imgsz == 64 and predictor.names == {0: "can", 1: "bottle"}
    det, speed = predictor(np.zeros((120, 90, 3), np.uint8), conf=0.0, max_det=5)
    assert det.shape == (5, 6)
    assert det[:, 0].max() <= 90 and det[:, 3].max() <= 120
    assert set(speed) == {"preprocess", "inference", "postprocess"}


@needs_ov
class TestCalibrationImages:
    """What INT8 quantisation feeds on — any source, no labels."""

    def _images(self, tmp_path, count):
        import cv2

        for i in range(count):
            cv2.imwrite(str(tmp_path / f"{i}.jpg"), np.full((20, 30, 3), i * 10, np.uint8))
        return tmp_path

    def test_a_folder_of_images_becomes_preprocessed_tensors(self, tmp_path):
        from rtdetr.exporter import calibration_images

        tensors = calibration_images(self._images(tmp_path, 3), imgsz=32, verbose=False)
        assert len(tensors) == 3
        assert tensors[0].shape == (1, 3, 32, 32)
        assert 0.0 <= tensors[0].min() and tensors[0].max() <= 1.0

    def test_it_stops_at_the_sample_cap(self, tmp_path):
        from rtdetr.exporter import calibration_images

        assert len(calibration_images(self._images(tmp_path, 5), 32, samples=2, verbose=False)) == 2

    def test_a_data_yaml_calibrates_on_its_val_split(self, dataset):
        from rtdetr.exporter import calibration_images

        tensors = calibration_images(str(dataset), imgsz=32, verbose=False)
        assert len(tensors) == 2  # the two val images

    def test_an_empty_source_is_an_error_not_an_empty_model(self, tmp_path):
        from rtdetr.exporter import calibration_images

        (tmp_path / "notes.md").write_text("no images here")
        with pytest.raises(FileNotFoundError):
            calibration_images(tmp_path, 32, verbose=False)

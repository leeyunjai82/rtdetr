# Apache-2.0
"""The RTDETR facade: what it accepts, what it refuses, what predict returns."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rtdetr import RTDETR, ModelNotFoundError
from rtdetr.model import _torch_device, increment_path

from .conftest import needs_ov, needs_torch


def test_a_pretrained_name_is_accepted_in_either_spelling():
    assert RTDETR("rtdetr_r18").model_name == "rtdetr-r18"
    assert RTDETR("RTDETR-R50").variant == "r50"


def test_an_unknown_name_lists_the_ones_that_exist():
    with pytest.raises(ModelNotFoundError, match="rtdetr-r18, rtdetr-r34, rtdetr-r50"):
        RTDETR("rtdetr-r101")


def test_a_missing_file_is_a_file_error_not_an_import_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        RTDETR(tmp_path / "best.pt")
    with pytest.raises(FileNotFoundError, match="model not found"):
        RTDETR(tmp_path / "model.xml")


def test_predict_without_a_source_says_what_is_missing():
    with pytest.raises(ValueError, match="needs a source"):
        RTDETR("rtdetr-r18").predict(None)


def test_a_typo_in_a_predict_keyword_is_not_swallowed():
    with pytest.raises(TypeError, match="confidence"):
        RTDETR("rtdetr-r18").predict("bus.jpg", confidence=0.5)


def test_run_directories_increment_instead_of_overwriting(tmp_path):
    first = increment_path(tmp_path / "predict")
    first.mkdir()
    second = increment_path(tmp_path / "predict")
    assert second.name == "predict2"
    second.mkdir()
    assert increment_path(tmp_path / "predict").name == "predict3"


@pytest.mark.parametrize(
    ("given", "expected"), [(0, "cuda:0"), ("0", "cuda:0"), ("cpu", "cpu"), (None, None)]
)
def test_device_shorthands_map_onto_torch_devices(given, expected):
    assert _torch_device(given) == expected


def test_val_without_weights_explains_how_to_get_them():
    model = RTDETR("rtdetr-r18")
    model.model_name = "not-a-mirror-name"  # simulate a local model with no torch side
    with pytest.raises(RuntimeError, match="no torch weights"):
        model.val(data="data.yaml")


def test_export_rejects_formats_it_cannot_write():
    with pytest.raises(ValueError, match="openvino"):
        RTDETR("rtdetr-r18").export(format="tflite")


@needs_torch
@needs_ov
class TestAgainstARealModel:
    """The user-facing checklist, run against an actual compiled IR."""

    def test_predict_answers_with_a_list_of_results(self, tiny_ir, image):
        model = RTDETR(str(tiny_ir), device="CPU")
        results = model(image, conf=0.0, max_det=3)
        assert isinstance(results, list) and len(results) == 1
        result = results[0]
        assert len(result.boxes) == 3
        assert result.names == {0: "can", 1: "bottle"}
        assert result.orig_shape == (80, 80)
        assert result.path.endswith("bus.jpg")

    def test_stream_true_gives_a_generator(self, tiny_ir, image):
        model = RTDETR(str(tiny_ir), device="CPU", verbose=False)
        stream = model.predict(image, conf=0.0, max_det=1, stream=True)
        assert not isinstance(stream, list)
        assert [len(r.boxes) for r in stream] == [1]

    def test_an_ndarray_source_needs_no_file_at_all(self, tiny_ir):
        model = RTDETR(str(tiny_ir), device="CPU", verbose=False)
        results = model(np.zeros((50, 70, 3), np.uint8), conf=0.0, max_det=2)
        assert results[0].orig_shape == (50, 70)

    def test_a_folder_is_processed_in_order(self, tiny_ir, tmp_path):
        import cv2

        for name in ("1.jpg", "2.jpg"):
            cv2.imwrite(str(tmp_path / name), np.zeros((40, 40, 3), np.uint8))
        model = RTDETR(str(tiny_ir), device="CPU", verbose=False)
        results = model.predict(tmp_path, conf=0.0, max_det=1)
        assert [Path(r.path).name for r in results] == ["1.jpg", "2.jpg"]

    def test_save_writes_into_runs_detect_predict(self, tiny_ir, image, tmp_path):
        model = RTDETR(str(tiny_ir), device="CPU", verbose=False)
        model.predict(image, conf=0.0, max_det=2, save=True, project=str(tmp_path / "runs"))
        saved = tmp_path / "runs" / "detect" / "predict" / "bus.jpg"
        assert saved.exists()
        model.predict(image, conf=0.0, max_det=2, save=True, project=str(tmp_path / "runs"))
        assert (tmp_path / "runs" / "detect" / "predict2" / "bus.jpg").exists()

    def test_the_verbose_line_reads_like_ultralytics(self, tiny_ir, image, capsys):
        RTDETR(str(tiny_ir), device="CPU")(image, conf=0.0, max_det=2)
        line = capsys.readouterr().out.strip()
        assert line.startswith("image 1/1 ")
        assert "64x64" in line and line.endswith("ms")

    def test_track_puts_an_id_on_every_box(self, tiny_ir, tmp_path):
        import cv2

        clip = tmp_path / "clip.mp4"
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (48, 48))
        for _ in range(3):
            writer.write(np.zeros((48, 48, 3), np.uint8))
        writer.release()

        model = RTDETR(str(tiny_ir), device="CPU", verbose=False)
        results = model.track(clip, conf=0.0, max_det=2)
        assert len(results) == 3
        assert all(r.boxes.id is not None for r in results)
        assert results[0].boxes.id.tolist() == results[-1].boxes.id.tolist()

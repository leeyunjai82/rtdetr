# Apache-2.0
"""Results/Boxes — the half of the parity checklist users touch every day."""

from __future__ import annotations

import numpy as np
import pytest

from rtdetr.results import Boxes, Results

DET = np.array(
    [[10, 10, 50, 60, 0.9, 0], [0, 0, 20, 20, 0.8, 0], [5, 5, 30, 30, 0.7, 1]], np.float32
)


@pytest.fixture
def result() -> Results:
    img = np.zeros((100, 200, 3), np.uint8)
    return Results(img, "bus.jpg", {0: "person", 1: "car"}, DET)


def test_boxes_expose_every_coordinate_flavour_yolo_users_expect(result):
    boxes = result.boxes
    assert boxes.xyxy.shape == (3, 4)
    assert boxes.xywh[0].tolist() == [30.0, 35.0, 40.0, 50.0]
    assert boxes.xyxyn[0].tolist() == pytest.approx([0.05, 0.1, 0.25, 0.6])
    assert boxes.xywhn[0].tolist() == pytest.approx([0.15, 0.35, 0.2, 0.5])
    assert boxes.conf.tolist() == pytest.approx([0.9, 0.8, 0.7])
    assert boxes.cls.tolist() == [0, 0, 1]
    assert boxes.id is None  # ids only exist after track()
    assert len(boxes) == 3


def test_boxes_are_indexable_and_stay_boxes(result):
    first = result.boxes[0]
    assert isinstance(first, Boxes) and len(first) == 1
    assert [len(b) for b in result.boxes] == [1, 1, 1]


def test_track_ids_appear_when_the_data_has_seven_columns():
    data = np.array([[0, 0, 10, 10, 7, 0.9, 1]], np.float32)
    boxes = Boxes(data, (100, 100))
    assert boxes.is_track and boxes.id.tolist() == [7]
    assert boxes.conf.tolist() == pytest.approx([0.9]) and boxes.cls.tolist() == [1]


def test_empty_predictions_are_a_normal_empty_result():
    result = Results(np.zeros((10, 10, 3), np.uint8))
    assert len(result) == 0 and result.boxes.xyxy.shape == (0, 4)
    assert "no detections" in result.verbose()


def test_the_log_line_counts_and_pluralises_like_ultralytics(result):
    assert result.verbose() == "2 persons, 1 car, "
    assert str(result) == "100x200 2 persons, 1 car"


def test_names_fall_back_to_a_readable_placeholder():
    result = Results(np.zeros((10, 10, 3), np.uint8), boxes=DET)
    assert result.name_of(1) == "class_1"


def test_plot_draws_without_touching_the_original(result):
    before = result.orig_img.copy()
    plotted = result.plot()
    assert plotted.shape == result.orig_img.shape
    assert plotted.any()  # something was drawn
    assert np.array_equal(result.orig_img, before)


def test_save_writes_the_annotated_image(result, tmp_path):
    out = result.save(tmp_path / "out" / "annotated.jpg")
    assert out.exists() and out.stat().st_size > 0


def test_summary_is_json_ready(result):
    rows = result.summary()
    assert rows[0]["name"] == "person" and rows[0]["confidence"] == pytest.approx(0.9)
    assert set(rows[0]["box"]) == {"x1", "y1", "x2", "y2"}


def test_malformed_box_data_is_rejected_early():
    with pytest.raises(ValueError, match=r"\(N, 6\)"):
        Boxes(np.zeros((3, 5), np.float32), (10, 10))

# Apache-2.0
"""Label files and the folder layout the trainer expects."""

from __future__ import annotations

import numpy as np
from labeling import label_path, labels_beside, list_images, read_labels, write_labels

BOXES = [{"cls": 0, "cx": 0.5, "cy": 0.4, "w": 0.2, "h": 0.3}]


def test_labels_round_trip_through_the_yolo_text_format(tmp_path):
    path = tmp_path / "a.txt"
    write_labels(path, BOXES)
    assert path.read_text() == "0 0.500000 0.400000 0.200000 0.300000\n"
    assert read_labels(path) == BOXES


def test_an_image_with_nothing_in_it_is_an_empty_file_not_a_missing_one(tmp_path):
    path = tmp_path / "empty.txt"
    write_labels(path, [])
    assert path.exists() and path.read_text() == ""
    assert read_labels(path) == []
    assert read_labels(tmp_path / "never_written.txt") == []


def test_coordinates_are_clamped_into_the_image(tmp_path):
    path = tmp_path / "a.txt"
    write_labels(path, [{"cls": 1, "cx": 1.4, "cy": -0.2, "w": 0.5, "h": 0.5}])
    box = read_labels(path)[0]
    assert box["cx"] == 1.0 and box["cy"] == 0.0


def test_label_files_mirror_the_image_tree(tmp_path):
    image = tmp_path / "images" / "train" / "a.jpg"
    out = label_path(image, tmp_path / "images", tmp_path / "labels")
    assert out == tmp_path / "labels" / "train" / "a.txt"


def test_labels_land_next_to_images_the_way_the_trainer_expects(tmp_path):
    assert labels_beside(tmp_path / "images" / "train") == tmp_path / "labels" / "train"
    assert labels_beside(tmp_path / "photos") == tmp_path / "labels"


def test_only_images_are_listed(tmp_path):
    import cv2

    (tmp_path / "sub").mkdir()
    cv2.imwrite(str(tmp_path / "a.jpg"), np.zeros((8, 8, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "sub" / "b.png"), np.zeros((8, 8, 3), np.uint8))
    (tmp_path / "notes.txt").write_text("nope")
    assert [p.name for p in list_images(tmp_path)] == ["a.jpg", "b.png"]


def test_a_model_only_contributes_classes_this_project_labels():
    """A COCO model labelling a "person, car" project stays quiet about the rest."""

    class FakeBoxes:
        cls = [0.0, 2.0, 16.0]
        conf = [0.9, 0.8, 0.7]
        xywhn = [(0.1, 0.1, 0.2, 0.2), (0.5, 0.5, 0.3, 0.3), (0.9, 0.9, 0.1, 0.1)]

        def __len__(self):
            return 3

    class FakeResult:
        boxes = FakeBoxes()
        names = {0: "person", 2: "car", 16: "dog"}

        def name_of(self, index):
            return self.names[int(index)]

    class FakeModel:
        def predict(self, *args, **kwargs):
            return [FakeResult()]

    from labeling import predict_boxes

    boxes = predict_boxes(FakeModel(), "x.jpg", ["person", "car"], conf=0.5)
    assert [b["cls"] for b in boxes] == [0, 1]  # the dog is not ours to label
    assert boxes[0]["conf"] == 0.9

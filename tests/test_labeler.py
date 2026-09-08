# Apache-2.0
"""The labelling tool: label files, the folder layout, and the HTTP endpoints."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from rtdetr.labeler import (
    LabelSession,
    free_port,
    label_path,
    read_labels,
    serve,
    write_labels,
)

BOXES = [{"cls": 0, "cx": 0.5, "cy": 0.4, "w": 0.2, "h": 0.3}]


@pytest.fixture
def images(tmp_path):
    import cv2

    folder = tmp_path / "images"
    folder.mkdir()
    for i in range(3):
        cv2.imwrite(str(folder / f"{i}.jpg"), np.full((40, 60, 3), 10 * i + 20, np.uint8))
    (folder / "notes.txt").write_text("not an image")
    return folder


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
    (tmp_path / "images" / "train").mkdir(parents=True)
    import cv2

    cv2.imwrite(str(tmp_path / "images" / "train" / "a.jpg"), np.zeros((8, 8, 3), np.uint8))
    session = LabelSession(tmp_path / "images" / "train", ["a"])
    assert session.labels_root == tmp_path / "labels" / "train"


def test_a_folder_with_no_images_says_so(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no images"):
        LabelSession(tmp_path / "empty", ["a"])
    with pytest.raises(NotADirectoryError):
        LabelSession(tmp_path / "nope", ["a"])


class TestServer:
    """The endpoints the page talks to, driven the way the page drives them."""

    @pytest.fixture
    def base(self, images):
        session = LabelSession(images, ["can", "bottle"])
        server = serve(session, port=free_port(), open_browser=False)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{server.server_port}", session
        server.shutdown()
        server.server_close()

    @staticmethod
    def get(url):
        with urllib.request.urlopen(url) as response:
            return response

    @staticmethod
    def get_json(url):
        with urllib.request.urlopen(url) as response:
            return json.load(response)

    @staticmethod
    def post_json(url, payload):
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            return json.load(response)

    def test_the_page_is_served(self, base):
        url, _ = base
        with urllib.request.urlopen(url) as response:
            body = response.read().decode()
        assert "<canvas" in body and "rtdetr" in body

    def test_state_lists_the_images_and_classes(self, base):
        url, _ = base
        state = self.get_json(f"{url}/api/state")
        assert state["names"] == ["can", "bottle"]
        assert [i["name"] for i in state["images"]] == ["0.jpg", "1.jpg", "2.jpg"]
        assert all(i["labelled"] is False for i in state["images"])
        assert state["can_autolabel"] is False  # no model given

    def test_an_image_comes_back_as_an_image(self, base):
        url, _ = base
        with urllib.request.urlopen(f"{url}/api/image/1") as response:
            assert response.headers["Content-Type"] == "image/jpeg"
            assert len(response.read()) > 100

    def test_saving_boxes_writes_the_file_the_trainer_reads(self, base):
        url, session = base
        assert self.get_json(f"{url}/api/labels/0") == {"boxes": []}
        assert self.post_json(f"{url}/api/labels/0", {"boxes": BOXES})["saved"] is True
        assert read_labels(session.label_file(0)) == BOXES
        assert self.get_json(f"{url}/api/labels/0")["boxes"] == BOXES
        assert self.get_json(f"{url}/api/state")["images"][0]["labelled"] is True

    def test_autolabel_without_a_model_explains_itself(self, base):
        url, _ = base
        with pytest.raises(urllib.error.HTTPError) as caught:
            self.post_json(f"{url}/api/autolabel/0", {})
        assert caught.value.code == 503

    def test_it_can_write_the_data_yaml_for_what_was_labelled(self, base, tmp_path):
        import yaml

        url, _ = base
        target = tmp_path / "data.yaml"
        assert self.post_json(f"{url}/api/data_yaml", {"path": str(target)})["path"] == str(target)
        cfg = yaml.safe_load(target.read_text())
        assert cfg["names"] == {0: "can", 1: "bottle"} and cfg["train"] == "images"

    def test_an_unknown_route_is_a_404(self, base):
        url, _ = base
        with pytest.raises(urllib.error.HTTPError) as caught:
            self.get(f"{url}/api/nope")
        assert caught.value.code == 404

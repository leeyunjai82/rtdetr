# Apache-2.0
"""Zoom-out, crop and flip must move the boxes exactly as they move the pixels."""

from __future__ import annotations

import random

import numpy as np
import pytest

from easydetect.data import augment as aug


def _scene(h=120, w=200):
    """Black picture with two white boxes; labels in cls cx cy w h."""
    img = np.zeros((h, w, 3), np.uint8)
    boxes = [(0, 20, 30, 70, 90), (1, 120, 10, 190, 60)]       # cls, x1, y1, x2, y2
    labels = []
    for c, x1, y1, x2, y2 in boxes:
        img[y1:y2, x1:x2] = 255
        labels.append([c, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h])
    return img, np.array(labels, np.float32)


@pytest.fixture
def no_colour(monkeypatch):
    monkeypatch.setattr(aug, "PHOTO_P", 0.0)    # keep white white, to find the boxes


def test_every_box_still_covers_its_object(no_colour):
    random.seed(0)
    zoomed = cropped = 0
    for _ in range(300):
        img, labels = _scene()
        out, got = aug.apply(img, labels)
        h, w = out.shape[:2]
        zoomed += h > 120
        cropped += h < 120 or w < 200
        assert got.shape[1] == 5 and set(got[:, 0]) <= {0, 1}
        assert (got[:, 1:] >= 0).all() and (got[:, 1:] <= 1 + 1e-6).all()
        for _, cx, cy, bw, bh in got:
            x1, x2 = int(round((cx - bw / 2) * w)), int(round((cx + bw / 2) * w))
            y1, y2 = int(round((cy - bh / 2) * h)), int(round((cy + bh / 2) * h))
            inside = out[y1 + 1:y2 - 1, x1 + 1:x2 - 1]
            if inside.size:
                assert inside.mean() > 240, "a box that no longer sits on its object"
    assert zoomed > 50 and cropped > 50             # both happen, often


def test_the_plain_phase_only_flips(no_colour):
    random.seed(1)
    img, labels = _scene()
    for _ in range(20):
        out, got = aug.apply(img.copy(), labels.copy(), strong=False)
        assert out.shape == img.shape and len(got) == 2
        assert np.allclose(sorted(got[:, 3]), sorted(labels[:, 3]))


def test_a_background_picture_goes_through(no_colour):
    random.seed(2)
    for _ in range(50):
        out, got = aug.apply(np.zeros((80, 80, 3), np.uint8), np.zeros((0, 5), np.float32))
        assert got.shape == (0, 5) and out.ndim == 3 and min(out.shape[:2]) > 0


def test_colour_jitter_keeps_the_picture_valid():
    random.seed(3)
    img = np.random.default_rng(0).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    for _ in range(30):
        out = aug.photometric(img)
        assert out.dtype == np.uint8 and out.shape == img.shape


def test_the_last_tenth_trains_on_plain_pictures():
    from easydetect.trainer import Trainer

    class Stub:                                   # only clean_from and _augment_lines are used
        clean_from = Trainer.clean_from
        _augment_lines = Trainer._augment_lines

    stub = Stub()
    for epochs, clean in ((20, 18), (50, 45), (100, 90), (5, 4), (3, 3), (1, 1)):
        stub.epochs = epochs
        assert stub.clean_from == clean
    stub.epochs, stub.imgsz, stub.augment = 20, 640, True
    assert any("last 2 epoch" in line for line in stub._augment_lines())
    stub.augment = False
    assert stub._augment_lines() == [aug.DESCRIPTION[-1], "resize to 640×640"]

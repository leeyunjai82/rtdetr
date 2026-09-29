# Apache-2.0
"""A small IoU tracker — enough to give boxes stable ``.id`` across frames.

No motion model and no appearance features: detections are greedily matched to
the most-overlapping track of the same class, tracks survive ``max_age`` frames
without a match, and anything unmatched starts a new id. That is exactly the
behaviour ``model.track(...)`` promises and nothing more.
"""

from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) xyxy -> (N,M) IoU."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None] - inter, 1e-9, None)


class Track:
    __slots__ = ("id", "box", "cls", "age", "hits")

    def __init__(self, track_id: int, box: np.ndarray, cls: int) -> None:
        self.id = track_id
        self.box = box
        self.cls = cls
        self.age = 0
        self.hits = 1


class IoUTracker:
    def __init__(self, iou: float = 0.3, max_age: int = 30, match_class: bool = True) -> None:
        self.iou = iou
        self.max_age = max_age
        self.match_class = match_class
        self.tracks: list[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def update(self, det: np.ndarray) -> np.ndarray:
        """``(N,6)`` detections -> ``(N,7)`` with the track id at column 4."""
        det = np.asarray(det, np.float32).reshape(-1, 6)
        for track in self.tracks:
            track.age += 1

        ids = np.zeros(len(det), np.float32)
        if self.tracks and len(det):
            ious = iou_matrix(det[:, :4], np.stack([t.box for t in self.tracks]))
            if self.match_class:
                same = det[:, 5][:, None] == np.array([t.cls for t in self.tracks], np.float32)
                ious = np.where(same, ious, 0.0)
            taken = set()
            # highest-confidence detection picks its best free track first
            for d in np.argsort(-det[:, 4]):
                order = np.argsort(-ious[d])
                for t in order:
                    if t in taken or ious[d, t] < self.iou:
                        break
                    taken.add(int(t))
                    track = self.tracks[int(t)]
                    track.box, track.cls, track.age = det[d, :4], int(det[d, 5]), 0
                    track.hits += 1
                    ids[d] = track.id
                    break

        for d in range(len(det)):
            if ids[d] == 0:
                track = Track(self._next_id, det[d, :4], int(det[d, 5]))
                self._next_id += 1
                self.tracks.append(track)
                ids[d] = track.id

        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        if not len(det):
            return np.zeros((0, 7), np.float32)
        return np.concatenate([det[:, :4], ids[:, None], det[:, 4:]], axis=1)

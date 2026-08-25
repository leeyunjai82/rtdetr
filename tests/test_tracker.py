# Apache-2.0
"""The IoU tracker: ids that survive movement and die when the object leaves."""

from __future__ import annotations

import numpy as np

from rtdetr.tracker import IoUTracker, iou_matrix


def det(*boxes):
    return np.array([[*b, 0.9, 0] for b in boxes], np.float32)


def test_iou_matrix_handles_overlap_and_emptiness():
    a = np.array([[0, 0, 10, 10]], np.float32)
    assert iou_matrix(a, a)[0, 0] == 1.0
    assert iou_matrix(a, np.array([[20, 20, 30, 30]], np.float32))[0, 0] == 0.0
    assert iou_matrix(a, np.zeros((0, 4), np.float32)).shape == (1, 0)


def test_a_slowly_moving_box_keeps_its_id():
    tracker = IoUTracker()
    first = tracker.update(det([0, 0, 20, 20]))
    second = tracker.update(det([2, 2, 22, 22]))
    assert first.shape[1] == 7
    assert first[0, 4] == second[0, 4] == 1


def test_a_box_that_jumps_away_gets_a_new_id():
    tracker = IoUTracker()
    tracker.update(det([0, 0, 20, 20]))
    moved = tracker.update(det([100, 100, 120, 120]))
    assert moved[0, 4] == 2


def test_two_objects_keep_separate_ids_across_frames():
    tracker = IoUTracker()
    tracker.update(det([0, 0, 20, 20], [100, 100, 120, 120]))
    second = tracker.update(det([101, 101, 121, 121], [1, 1, 21, 21]))
    assert second[:, 4].tolist() == [2, 1]


def test_ids_are_not_shared_between_classes():
    tracker = IoUTracker()
    tracker.update(np.array([[0, 0, 20, 20, 0.9, 0]], np.float32))
    other_class = tracker.update(np.array([[0, 0, 20, 20, 0.9, 1]], np.float32))
    assert other_class[0, 4] == 2


def test_a_track_expires_after_max_age_frames():
    tracker = IoUTracker(max_age=2)
    tracker.update(det([0, 0, 20, 20]))
    for _ in range(3):
        tracker.update(np.zeros((0, 6), np.float32))
    assert tracker.update(det([0, 0, 20, 20]))[0, 4] == 2


def test_an_empty_frame_is_an_empty_seven_column_array():
    assert IoUTracker().update(np.zeros((0, 6), np.float32)).shape == (0, 7)


def test_reset_starts_ids_over():
    tracker = IoUTracker()
    tracker.update(det([0, 0, 20, 20]))
    tracker.reset()
    assert tracker.update(det([0, 0, 20, 20]))[0, 4] == 1

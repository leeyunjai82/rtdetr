# Apache-2.0
"""Box drawing. Deliberately small: OpenCV rectangles and labels, no font files."""

from __future__ import annotations

import numpy as np

#: A 20-colour palette (BGR), cycled by class index.
PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
    (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0),
    (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]


def color_for(index: int) -> tuple[int, int, int]:
    return PALETTE[int(index) % len(PALETTE)]


def draw_boxes(
    img: np.ndarray,
    boxes,
    names: dict[int, str] | None = None,
    conf: bool = True,
    labels: bool = True,
    line_width: int | None = None,
) -> np.ndarray:
    """Draw ``boxes`` (a :class:`~rtdetr.results.Boxes`) onto a copy of ``img``."""
    import cv2

    out = np.ascontiguousarray(img.copy())
    names = names or {}
    h, w = out.shape[:2]
    lw = line_width or max(round((h + w) / 2 * 0.003), 2)
    font_scale = lw / 3.0

    ids = boxes.id if boxes.is_track else None
    placed: list[tuple[int, int, int, int]] = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = (int(round(v)) for v in boxes.xyxy[i])
        cls = int(boxes.cls[i])
        color = color_for(cls)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, lw, cv2.LINE_AA)
        if not labels:
            continue
        text = names.get(cls, f"class_{cls}")
        if ids is not None:
            text = f"id:{int(ids[i])} {text}"
        if conf:
            text += f" {float(boxes.conf[i]):.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(lw - 1, 1))
        left, top = _label_position(x1, y1, y2, tw, th + 3, (h, w), placed)
        placed.append((left, top, left + tw, top + th + 3))
        cv2.rectangle(out, (left, top), (left + tw, top + th + 3), color, -1, cv2.LINE_AA)
        cv2.putText(
            out,
            text,
            (left, top + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(lw - 1, 1),
            cv2.LINE_AA,
        )
    return out


def _label_position(x1, y1, y2, tw, th, shape, placed):
    """Where to put one label: inside the frame, and clear of its neighbours.

    Crowded scenes are the normal case for a detector, and labels that land on
    top of each other are unreadable ("personperson 0.87n 0.87"). Try above the
    box, then below it, then just inside it, and take the first free slot.
    """
    h, w = shape
    left = max(0, min(int(x1), w - tw))
    for top in (y1 - th, y2, y1, y1 + th):
        top = max(0, min(int(top), h - th))
        box = (left, top, left + tw, top + th)
        if not any(_overlaps(box, other) for other in placed):
            return left, top
    return left, max(0, min(int(y1 - th), h - th))


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

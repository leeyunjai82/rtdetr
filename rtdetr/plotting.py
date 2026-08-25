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
        outside = y1 - th - 3 >= 0
        top = y1 - th - 3 if outside else y1
        cv2.rectangle(out, (x1, top), (x1 + tw, top + th + 3), color, -1, cv2.LINE_AA)
        cv2.putText(
            out,
            text,
            (x1, top + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(lw - 1, 1),
            cv2.LINE_AA,
        )
    return out

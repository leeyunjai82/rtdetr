# Apache-2.0
"""Where a label lives, and how one line of it reads — shared by everything.

The trainer reads labels, easydetect lab writes them, the exporter packs them.
If any two of those disagree on where ``a.jpg``'s boxes are, training runs on
nothing and nobody is told. So there is one rule, here, and no torch import:
the lab uses it at startup.
"""

from __future__ import annotations

from pathlib import Path, PurePath

import numpy as np


def label_path(image: str | PurePath) -> PurePath:
    """``…/images/train/a.jpg`` -> ``…/labels/train/a.txt``.

    The last folder called ``images`` becomes ``labels``; with none, the label
    sits beside the image. Works on relative paths too, which is how an export
    lays out a zip that the trainer will read after unpacking.
    """
    image = image if isinstance(image, PurePath) else Path(image)
    parts = list(image.parts)
    for i in range(len(parts) - 2, -1, -1):  # folders only, never the file name
        if parts[i] == "images":
            parts[i] = "labels"
            return type(image)(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def label_row_to_box(values: list[str]) -> list[float] | None:
    """One label line -> ``[cls, cx, cy, w, h]``, or None for a line to skip.

    ``cls cx cy w h`` is a box (a trailing sixth value, a confidence, is
    ignored). ``cls x1 y1 x2 y2 x3 y3 ...`` is a segmentation polygon, common
    in exported datasets; reading its first four numbers as a box would train
    on nonsense without a single error, so it becomes its bounding box.
    """
    if len(values) >= 7 and len(values) % 2 == 1:
        xy = np.asarray(values[1:], np.float32).reshape(-1, 2)
        (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
        return [float(values[0]), float(x0 + x1) / 2, float(y0 + y1) / 2,
                float(x1 - x0), float(y1 - y0)]
    if len(values) >= 5:
        return [float(v) for v in values[:5]]
    return None

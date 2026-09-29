# Apache-2.0
"""Training-time augmentation for boxes: the recipe D-FINE trains with.

Photometric jitter, a random zoom-out onto a larger canvas, an IoU-constrained
random crop, and a horizontal flip — the steps of D-FINE's configs with the
probabilities and ranges they use (torchvision's defaults), written here in
NumPy/OpenCV. One difference: the zoom-out canvas is the picture's mean colour
rather than black, so the border is not a cue that says "zoomed".
Zoom-out makes objects small and crop makes them large, so a model sees each
class at many sizes and positions instead of only the framing the photos had.

Boxes are pixel ``xyxy`` float arrays; every function returns new ones.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

ZOOM_OUT_P, ZOOM_OUT_MAX = 0.5, 4.0          # canvas up to 4× the image, half the time
CROP_P = 0.8
CROP_MIN_IOUS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, None)   # None: keep the whole image
CROP_SCALE, CROP_ASPECT, CROP_TRIES = (0.3, 1.0), (0.5, 2.0), 40
PHOTO_P = 0.5

DESCRIPTION = [
    f"photometric jitter (brightness, contrast, saturation, hue; p={PHOTO_P} each)",
    f"zoom-out onto a canvas up to {ZOOM_OUT_MAX:g}× (p={ZOOM_OUT_P})",
    f"IoU-constrained random crop, {CROP_SCALE[0]:g}–{CROP_SCALE[1]:g} of each side (p={CROP_P})",
    "horizontal flip (p=0.5)",
]


def photometric(img: np.ndarray) -> np.ndarray:
    """Brightness, contrast, saturation and hue, each on its own coin flip."""
    out = img.astype(np.float32)
    if random.random() < PHOTO_P:                        # brightness ×0.875–1.125
        out *= random.uniform(0.875, 1.125)
    if random.random() < PHOTO_P:                        # contrast ×0.5–1.5 about the mean grey
        grey = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).mean()
        out = grey + (out - grey) * random.uniform(0.5, 1.5)
    out = np.clip(out, 0, 255).astype(np.uint8)
    saturate, shift = random.random() < PHOTO_P, random.random() < PHOTO_P
    if saturate or shift:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        if saturate:                                     # saturation ×0.5–1.5
            hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.5, 1.5), 0, 255)
        if shift:                                        # hue ±0.05 of the circle (OpenCV: 0–180)
            hsv[..., 0] = (hsv[..., 0] + random.uniform(-9, 9)) % 180
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def zoom_out(img: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Place the image somewhere on a bigger canvas of its mean colour."""
    if random.random() >= ZOOM_OUT_P:
        return img, boxes
    h, w = img.shape[:2]
    s = random.uniform(1.0, ZOOM_OUT_MAX)
    ch, cw = int(h * s), int(w * s)
    top, left = random.randint(0, ch - h), random.randint(0, cw - w)
    canvas = np.empty((ch, cw, 3), img.dtype)
    canvas[:] = img.reshape(-1, 3).mean(0).astype(img.dtype)
    canvas[top:top + h, left:left + w] = img
    return canvas, boxes + np.array([left, top, left, top], np.float32)


def _iou(region: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(region[0], boxes[:, 0])
    y1 = np.maximum(region[1], boxes[:, 1])
    x2 = np.minimum(region[2], boxes[:, 2])
    y2 = np.minimum(region[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    region_area = (region[2] - region[0]) * (region[3] - region[1])
    return inter / np.maximum(area + region_area - inter, 1e-9)


def iou_crop(img: np.ndarray, boxes: np.ndarray,
             classes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut out a region that overlaps some box by at least a random IoU.

    Boxes whose centre falls outside the region are dropped; the rest are
    clipped to it. A picture with no boxes (a background frame) takes any
    region. When no region qualifies, the picture stays whole.
    """
    if random.random() >= CROP_P:
        return img, boxes, classes
    h, w = img.shape[:2]
    min_iou = random.choice(CROP_MIN_IOUS)
    if min_iou is None:
        return img, boxes, classes
    for _ in range(CROP_TRIES):
        cw = w * random.uniform(*CROP_SCALE)
        ch = h * random.uniform(*CROP_SCALE)
        if not CROP_ASPECT[0] <= cw / ch <= CROP_ASPECT[1]:
            continue
        # whole pixels: the boxes must be cut exactly where the picture is
        cw, ch = max(int(cw), 1), max(int(ch), 1)
        left, top = random.randint(0, w - cw), random.randint(0, h - ch)
        region = np.array([left, top, left + cw, top + ch], np.float32)
        if len(boxes):
            if _iou(region, boxes).max() < min_iou:
                continue
            cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
            inside = (cx > region[0]) & (cx < region[2]) & (cy > region[1]) & (cy < region[3])
            if not inside.any():
                continue
            kept = boxes[inside].copy()
            kept[:, [0, 2]] = np.clip(kept[:, [0, 2]], region[0], region[2]) - region[0]
            kept[:, [1, 3]] = np.clip(kept[:, [1, 3]], region[1], region[3]) - region[1]
            boxes, classes = kept, classes[inside]
        return img[top:top + ch, left:left + cw], boxes, classes
    return img, boxes, classes


def hflip(img: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() >= 0.5:
        return img, boxes
    w = img.shape[1]
    flipped = boxes.copy()
    flipped[:, 0], flipped[:, 2] = w - boxes[:, 2], w - boxes[:, 0]
    return np.ascontiguousarray(img[:, ::-1]), flipped


def apply(img: np.ndarray, labels: np.ndarray, strong: bool = True,
          min_side: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Augment one picture and its ``cls cx cy w h`` (normalised) labels.

    ``strong=False`` keeps only the flip — what the last epochs train on, as
    in D-FINE, so the model settles on pictures framed like the ones it will
    see. ``min_side`` drops boxes thinner than that many pixels
    of the returned image.
    """
    h, w = img.shape[:2]
    classes = labels[:, 0].copy()
    boxes = np.empty((len(labels), 4), np.float32)
    boxes[:, 0] = (labels[:, 1] - labels[:, 3] / 2) * w
    boxes[:, 1] = (labels[:, 2] - labels[:, 4] / 2) * h
    boxes[:, 2] = (labels[:, 1] + labels[:, 3] / 2) * w
    boxes[:, 3] = (labels[:, 2] + labels[:, 4] / 2) * h

    if strong:
        img = photometric(img)
        img, boxes = zoom_out(img, boxes)
        img, boxes, classes = iou_crop(img, boxes, classes)
    img, boxes = hflip(img, boxes)

    h, w = img.shape[:2]
    keep = ((boxes[:, 2] - boxes[:, 0]) >= min_side) & ((boxes[:, 3] - boxes[:, 1]) >= min_side)
    boxes, classes = boxes[keep], classes[keep]
    out = np.empty((len(boxes), 5), np.float32)
    out[:, 0] = classes
    out[:, 1] = (boxes[:, 0] + boxes[:, 2]) / 2 / w
    out[:, 2] = (boxes[:, 1] + boxes[:, 3]) / 2 / h
    out[:, 3] = (boxes[:, 2] - boxes[:, 0]) / w
    out[:, 4] = (boxes[:, 3] - boxes[:, 1]) / h
    return img, out

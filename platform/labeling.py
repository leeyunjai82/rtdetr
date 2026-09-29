# Apache-2.0
"""Label files, and the model-assisted first pass over them.

Labels are the same ``labels/<name>.txt`` files the trainer reads: one line per
box, ``cls cx cy w h`` normalised. Nothing here is clever; the value is that the
detector writes the first draft and a person only corrects it.
"""

from __future__ import annotations

from pathlib import Path

from easydetect.data.labels import label_path as _shared_label_path
from easydetect.data.labels import label_row_to_box

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_SUFFIXES)


def label_path(image: Path) -> Path:
    """Where the trainer will read this image's boxes — the only place to write them.

    One rule, shared with the trainer (``easydetect.data.labels``): the last
    ``images`` folder becomes ``labels``, and with none the label sits beside
    the image. Two rules used to exist here, and a folder not called ``images``
    got its labels written where training never looked.
    """
    return Path(_shared_label_path(Path(image)))


def labels_beside(images_root: Path) -> Path:
    """The folder labels for images directly in ``images_root`` land in."""
    return label_path(Path(images_root) / "x.jpg").parent


def read_labels(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = label_row_to_box(line.split())       # boxes, and polygons as their boxes
        if row is not None:
            cls, cx, cy, w, h = row
            boxes.append({"cls": int(cls), "cx": cx, "cy": cy, "w": w, "h": h})
    return boxes


def write_labels(path: Path, boxes: list[dict]) -> None:
    """An image with no objects is a valid label: an empty file, not a missing one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{int(b['cls'])} {_clamp(b['cx']):.6f} {_clamp(b['cy']):.6f} "
        f"{_clamp(b['w']):.6f} {_clamp(b['h']):.6f}"
        for b in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def predict_boxes(model, image: Path, names: list[str], conf: float = 0.35) -> list[dict]:
    """Run a model over one image and keep the boxes whose class we label.

    Classes are matched by name, case-insensitively: a COCO model labelling a
    ``person, car`` project contributes its people and cars and stays quiet
    about the rest.
    """
    result = model.predict(str(image), conf=conf, verbose=False)[0]
    lookup = {name.lower(): i for i, name in enumerate(names)}
    boxes = []
    for i in range(len(result.boxes)):
        name = result.name_of(result.boxes.cls[i]).lower()
        if name not in lookup:
            continue
        cx, cy, w, h = result.boxes.xywhn[i]
        boxes.append(
            {
                "cls": lookup[name],
                "cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h),
                "conf": round(float(result.boxes.conf[i]), 3),
            }
        )
    return boxes

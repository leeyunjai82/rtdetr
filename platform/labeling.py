# Apache-2.0
"""Label files, and the model-assisted first pass over them.

Labels are the same ``labels/<name>.txt`` files the trainer reads: one line per
box, ``cls cx cy w h`` normalised. Nothing here is clever; the value is that the
detector writes the first draft and a person only corrects it.
"""

from __future__ import annotations

from pathlib import Path

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_SUFFIXES)


def labels_beside(images_root: Path) -> Path:
    """``…/images/train`` -> ``…/labels/train``; otherwise a labels/ next door."""
    parts = list(Path(images_root).parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return Path(images_root).parent / "labels"


def label_path(image: Path, images_root: Path, labels_root: Path) -> Path:
    """``images/train/a.jpg`` -> ``labels/train/a.txt`` (mirroring the tree)."""
    return (Path(labels_root) / Path(image).relative_to(images_root)).with_suffix(".txt")


def read_labels(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            cls, cx, cy, w, h = (float(v) for v in parts[:5])
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

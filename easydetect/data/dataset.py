# Apache-2.0
"""Detection dataset: a data.yaml plus images/ and labels/*.txt.

data.yaml:
    path: dataset root (optional)
    train: images dir or txt list
    val: images dir or txt list
    names: {0: person, ...} or [person, ...]
Labels: <images-dir with 'images' replaced by 'labels'>/<stem>.txt
        each line: cls cx cy w h  (normalized)
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from . import augment as aug
from .labels import label_path, label_row_to_box

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_data_yaml(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    yaml_dir = Path(path).resolve().parent
    root = Path(cfg.get("path") or yaml_dir)
    if not root.is_absolute():
        root = yaml_dir / root
    if not root.exists():
        # a dataset downloaded from elsewhere often keeps its author's path
        # (/content/datasets/..., a Windows home folder); the yaml's own folder
        # is the only root that can be right on this machine
        root = yaml_dir
    names = cfg["names"]
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    names = {int(k): str(v) for k, v in names.items()}
    return {
        "root": root,
        "yaml_dir": yaml_dir,
        "train": cfg.get("train"),
        "val": cfg.get("val"),
        "names": names,
        "nc": len(names),
    }


def _locate(root: Path, yaml_dir: Path, spec: str) -> Path:
    """Where a train/val entry points, trying the ways data.yaml files are written.

    Relative to ``path:`` first; then to the yaml itself; then with leading
    ``../`` dropped — exports that sit beside their splits still write
    ``train: ../train/images``.
    """
    spec_path = Path(spec)
    if spec_path.is_absolute():
        return spec_path
    tried = [root / spec_path, yaml_dir / spec_path]
    stripped = Path(*[part for part in spec_path.parts if part != ".."] or ["."])
    tried.append(yaml_dir / stripped)
    for candidate in tried:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"train/val entry {spec!r} not found; tried "
        + ", ".join(str(c) for c in dict.fromkeys(tried))
    )


def _list_images(root: Path, spec, yaml_dir: Path | None = None):
    """Images for one split: a folder, a .txt list, or a list of either."""
    if isinstance(spec, (list, tuple)):
        files = [f for s in spec for f in _list_images(root, s, yaml_dir)]
        return list(dict.fromkeys(files))
    p = _locate(root, yaml_dir or root, str(spec))
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)
    if p.suffix == ".txt":
        base = p.parent
        out = []
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                q = Path(line)
                out.append(q if q.is_absolute() else base / q)
        return out
    raise FileNotFoundError(f"train/val entry not found: {p}")


_label_path = label_path  # older name, kept for callers


class DetDataset(Dataset):
    def __init__(self, data_yaml, split="train", imgsz=640, augment=True):
        cfg = load_data_yaml(data_yaml)
        self.names, self.nc = cfg["names"], cfg["nc"]
        self.imgsz = imgsz
        self.augment = augment and split == "train"
        # zoom-out, crop and colour jitter; the trainer turns them off for the
        # last epochs (the flip stays)
        self.strong = True
        if cfg[split] is None:
            raise ValueError(
                f"{data_yaml} has no '{split}:' entry. Add one — it may point at the "
                f"training images, but then mAP only measures memorisation."
            )
        self.files = _list_images(cfg["root"], cfg[split], cfg["yaml_dir"])
        if not self.files:
            raise FileNotFoundError(f"no images for split '{split}'")

    def __len__(self):
        return len(self.files)

    def _load_labels(self, img_file):
        lp = _label_path(img_file)
        if not lp.exists():
            return np.zeros((0, 5), np.float32)
        rows = []
        for line in lp.read_text().splitlines():
            row = label_row_to_box(line.split())
            if row is not None:
                rows.append(row)
        return np.asarray(rows, np.float32) if rows else np.zeros((0, 5), np.float32)

    def __getitem__(self, i):
        f = self.files[i]
        img = cv2.imread(str(f))
        if img is None:
            raise FileNotFoundError(f)
        labels = self._load_labels(f)  # cls, cx, cy, w, h (normalized)

        if self.augment:
            img, labels = aug.apply(img, labels, strong=self.strong)

        img = cv2.resize(img, (self.imgsz, self.imgsz))  # plain resize, as D-FINE trains
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()

        target = {
            "labels": torch.as_tensor(labels[:, 0], dtype=torch.long),
            "boxes": torch.as_tensor(labels[:, 1:5], dtype=torch.float32),  # cxcywh 0..1
        }
        return tensor, target

    @staticmethod
    def collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        targets = [b[1] for b in batch]
        return imgs, targets

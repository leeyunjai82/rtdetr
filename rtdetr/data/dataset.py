# Apache-2.0
"""YOLO-format detection dataset (same data.yaml layout Ultralytics users know).

data.yaml:
    path: dataset root (optional)
    train: images dir or txt list
    val: images dir or txt list
    names: {0: person, ...} or [person, ...]
Labels: <images-dir with 'images' replaced by 'labels'>/<stem>.txt
        each line: cls cx cy w h  (normalized)
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_data_yaml(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", Path(path).parent))
    if not root.is_absolute():
        root = Path(path).parent / root
    names = cfg["names"]
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    names = {int(k): str(v) for k, v in names.items()}
    return {
        "root": root,
        "train": cfg.get("train"),
        "val": cfg.get("val"),
        "names": names,
        "nc": len(names),
    }


def _list_images(root: Path, spec: str):
    p = (root / spec) if not Path(spec).is_absolute() else Path(spec)
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


def _label_path(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return img.with_suffix(".txt")


class DetDataset(Dataset):
    def __init__(self, data_yaml, split="train", imgsz=640, augment=True):
        cfg = load_data_yaml(data_yaml)
        self.names, self.nc = cfg["names"], cfg["nc"]
        self.imgsz = imgsz
        self.augment = augment and split == "train"
        self.files = _list_images(cfg["root"], cfg[split])
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
            v = line.split()
            if len(v) >= 5:
                rows.append([float(x) for x in v[:5]])
        return np.asarray(rows, np.float32) if rows else np.zeros((0, 5), np.float32)

    def __getitem__(self, i):
        f = self.files[i]
        img = cv2.imread(str(f))
        if img is None:
            raise FileNotFoundError(f)
        labels = self._load_labels(f)  # cls, cx, cy, w, h (normalized)

        if self.augment:
            # hflip
            if random.random() < 0.5:
                img = img[:, ::-1]
                if len(labels):
                    labels[:, 1] = 1.0 - labels[:, 1]
            # HSV jitter
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[..., 0] = (hsv[..., 0] + random.randint(-8, 8)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-30, 30), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] + random.randint(-30, 30), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        img = cv2.resize(img, (self.imgsz, self.imgsz))  # plain resize (RT-DETR default)
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

# Apache-2.0
"""OpenVINO inference for an exported RT-DETR IR (or ONNX).

The exported graph is ``images -> (boxes cxcywh 0..1, scores)``. The scores it
emits are **already sigmoid'd** — squashing them a second time silently turns a
0.95 detection into 0.72 and breaks every conf threshold, so the decoder only
applies a sigmoid when the tensor is clearly still logits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .utils.ops import cxcywh2xyxy_np


def preprocess_image(img: np.ndarray, imgsz: int) -> np.ndarray:
    """BGR HWC uint8 -> NCHW float32 RGB 0..1, plain-resized to imgsz.

    Calibration and inference must agree here, or an INT8 model is quantised
    against a distribution it never sees.
    """
    import cv2

    resized = cv2.resize(img, (imgsz, imgsz))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[None]


def _looks_like_logits(scores: np.ndarray) -> bool:
    """True when the head still emits raw logits (anything outside 0..1)."""
    return bool(scores.size) and (scores.min() < 0.0 or scores.max() > 1.0)


def read_names(model_path: Path) -> dict[int, str]:
    """Class names from ``labels.txt`` or ``<stem>.names.json`` beside the model."""
    model_path = Path(model_path)
    sidecar = model_path.with_suffix(".names.json")
    if sidecar.exists():
        table = json.loads(sidecar.read_text(encoding="utf-8"))
        return {int(k): str(v) for k, v in table.items()}
    labels = model_path.parent / "labels.txt"
    if labels.exists():
        lines = [ln.strip() for ln in labels.read_text(encoding="utf-8").splitlines()]
        return {i: n for i, n in enumerate(lines) if n}
    return {}


class OVPredictor:
    """Compiles an IR once, then runs images through it."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "AUTO",
        imgsz: int | None = None,
        names: dict[int, str] | None = None,
    ) -> None:
        import openvino as ov

        self.model_path = Path(model_path)
        self.device = device
        core = ov.Core()
        model = core.read_model(str(self.model_path))
        self.compiled = core.compile_model(model, device)
        self.input = self.compiled.input(0)
        self.names = names if names else read_names(self.model_path)
        self.imgsz = imgsz or self._input_size()

    def _input_size(self) -> int:
        shape = self.input.partial_shape
        try:
            h, w = shape[2], shape[3]
            if h.is_static and w.is_static:
                return int(max(h.get_length(), w.get_length()))
        except Exception:  # dynamic or unusual layout
            pass
        return 640

    # -- pipeline -----------------------------------------------------------

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """BGR HWC uint8 -> NCHW float32 RGB 0..1, plain-resized to imgsz."""
        return preprocess_image(img, self.imgsz)

    def infer(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``(boxes (Q,4) cxcywh 0..1, scores (Q,K))`` for one image."""
        outputs = self.compiled(tensor)
        arrays = [np.asarray(outputs[out]) for out in self.compiled.outputs]
        boxes = next((a for a in arrays if a.shape[-1] == 4), None)
        scores = next((a for a in arrays if a is not boxes), None)
        if boxes is None or scores is None:
            shapes = [a.shape for a in arrays]
            raise ValueError(f"unexpected model outputs {shapes}; expected boxes + scores")
        return boxes[0], scores[0]

    def postprocess(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        orig_shape: tuple[int, int],
        conf: float = 0.25,
        max_det: int = 300,
        classes: list[int] | None = None,
    ) -> np.ndarray:
        """-> ``(N, 6)`` array of ``x1 y1 x2 y2 conf cls`` in pixels."""
        if _looks_like_logits(scores):
            scores = 1.0 / (1.0 + np.exp(-scores))
        cls = scores.argmax(-1)
        best = scores.max(-1)
        keep = best >= conf
        if classes is not None:
            keep &= np.isin(cls, np.asarray(classes, np.int64))
        boxes, best, cls = boxes[keep], best[keep], cls[keep]
        if len(best) > max_det:
            top = np.argsort(-best)[:max_det]
            boxes, best, cls = boxes[top], best[top], cls[top]
        order = np.argsort(-best)
        boxes, best, cls = boxes[order], best[order], cls[order]

        h, w = orig_shape
        xyxy = cxcywh2xyxy_np(boxes.astype(np.float32)) * np.array([w, h, w, h], np.float32)
        xyxy[:, 0::2] = xyxy[:, 0::2].clip(0, w)
        xyxy[:, 1::2] = xyxy[:, 1::2].clip(0, h)
        return np.concatenate(
            [xyxy, best.astype(np.float32)[:, None], cls.astype(np.float32)[:, None]], axis=1
        )

    def __call__(
        self,
        img: np.ndarray,
        conf: float = 0.25,
        max_det: int = 300,
        classes: list[int] | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Run one image; returns ``(detections (N,6), speed dict in ms)``."""
        import time

        t0 = time.perf_counter()
        tensor = self.preprocess(img)
        t1 = time.perf_counter()
        boxes, scores = self.infer(tensor)
        t2 = time.perf_counter()
        det = self.postprocess(boxes, scores, img.shape[:2], conf, max_det, classes)
        t3 = time.perf_counter()
        speed = {
            "preprocess": (t1 - t0) * 1e3,
            "inference": (t2 - t1) * 1e3,
            "postprocess": (t3 - t2) * 1e3,
        }
        return det, speed

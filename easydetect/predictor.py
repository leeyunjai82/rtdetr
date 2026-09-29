# Apache-2.0
"""OpenVINO inference for an exported D-FINE IR (or ONNX).

The exported graph is ``images -> (boxes cxcywh 0..1, scores)``. The scores it
emits are **already sigmoid'd** — squashing them a second time silently turns a
0.95 detection into 0.72 and breaks every conf threshold, so the decoder only
applies a sigmoid when the tensor is clearly still logits.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from .utils.ops import cxcywh2xyxy_np


def preprocess_image(img: np.ndarray, imgsz: int) -> np.ndarray:
    """BGR HWC uint8 -> NCHW float32 RGB 0..1, plain-resized to imgsz."""
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


#: Default for ``iou``: above this IoU, two boxes are one object.
IOU = 0.7


def drop_duplicates(xyxy: np.ndarray, iou: float = IOU) -> np.ndarray:
    """Indices of the boxes to keep, from boxes sorted best first.

    A DETR is trained to give each object one query, and mostly does; but
    below its confident answers a second query can land on the same object,
    often under a neighbouring class — one vehicle as both ``truck`` 0.83 and
    ``car`` 0.57 (IoU 0.93). Measured on dfine-s at ``conf=0.25``, every pair
    overlapping by more than 0.7 was one object twice, and several pairs
    were two classes, so this ignores the class: a box that covers a better
    one by more than ``iou`` goes. Two separate objects that overlap this
    much in a picture are rare; pass ``iou=None`` to keep every box.
    """
    x1, y1, x2, y2 = xyxy.T
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    alive = np.ones(len(xyxy), bool)
    for i in range(len(xyxy)):
        if not alive[i]:
            continue
        rest = np.nonzero(alive[i + 1:])[0] + i + 1
        if not len(rest):
            break
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        overlap = inter / np.maximum(area[i] + area[rest] - inter, 1e-9)
        alive[rest[overlap > iou]] = False
    return np.nonzero(alive)[0]


class OVPredictor:
    """Compiles an IR once, then runs images through it."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "AUTO",
        imgsz: int | None = None,
        names: dict[int, str] | None = None,
        precision: str | None = None,
    ) -> None:
        """``precision`` overrides what OpenVINO runs the graph in.

        Left alone, OpenVINO picks: on a CPU that supports it that means
        bfloat16, which is much faster but shifts scores slightly. Pass
        ``"f32"`` when you want output that matches PyTorch exactly.
        """
        import openvino as ov

        self.model_path = Path(model_path)
        self.device = device
        self.precision = precision
        core = ov.Core()
        model = core.read_model(str(self.model_path))
        config = {"INFERENCE_PRECISION_HINT": precision} if precision else {}
        self.compiled = core.compile_model(model, device, config)
        self.input = self.compiled.input(0)
        # CompiledModel.__call__ reuses one InferRequest, so a second thread
        # calling it gets "Infer Request is busy". One request per thread.
        self._local = threading.local()
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
        request = getattr(self._local, "request", None)
        if request is None:
            request = self._local.request = self.compiled.create_infer_request()
        request.infer({0: tensor})
        # copy: the tensor data is the request's own buffer, overwritten next call
        arrays = [
            np.array(request.get_output_tensor(i).data)
            for i in range(len(self.compiled.outputs))
        ]
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
        iou: float | None = IOU,
    ) -> np.ndarray:
        """-> ``(N, 6)`` array of ``x1 y1 x2 y2 conf cls`` in pixels.

        ``iou`` drops a box that covers a higher-scoring one by more than that
        IoU, whatever the two classes (``None`` keeps them all): see
        :func:`drop_duplicates`.
        """
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
        if iou is not None and len(xyxy) > 1:
            kept = drop_duplicates(xyxy, iou)
            xyxy, best, cls = xyxy[kept], best[kept], cls[kept]
        return np.concatenate(
            [xyxy, best.astype(np.float32)[:, None], cls.astype(np.float32)[:, None]], axis=1
        )

    def __call__(
        self,
        img: np.ndarray,
        conf: float = 0.25,
        max_det: int = 300,
        classes: list[int] | None = None,
        iou: float | None = IOU,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Run one image; returns ``(detections (N,6), speed dict in ms)``."""
        import time

        t0 = time.perf_counter()
        tensor = self.preprocess(img)
        t1 = time.perf_counter()
        boxes, scores = self.infer(tensor)
        t2 = time.perf_counter()
        det = self.postprocess(boxes, scores, img.shape[:2], conf, max_det, classes, iou)
        t3 = time.perf_counter()
        speed = {
            "preprocess": (t1 - t0) * 1e3,
            "inference": (t2 - t1) * 1e3,
            "postprocess": (t3 - t2) * 1e3,
        }
        return det, speed

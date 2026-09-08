# Apache-2.0
"""Source resolution: everything a model can be pointed at.

Accepted::

    "bus.jpg"                 a single image
    "frames/*.jpg"            a glob
    "dataset/images"          a folder (recursed, images + videos)
    "list.txt"                a text file of paths, one per line
    "https://…/bus.jpg"       an image URL
    "https://…/clip.mp4"      a video URL / rtsp:// / rtmp:// stream
    "clip.mp4"                a video file
    0                         a webcam index
    ndarray | PIL.Image       pixels you already have
    [ …any of the above… ]    a list, processed in order
"""

from __future__ import annotations

import glob as _glob
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

IMG_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".dng", ".pfm"}
VID_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v", ".wmv", ".webm", ".gif"}
_URL_RE = re.compile(r"^(https?|rtsp|rtmp|tcp|udp)://", re.IGNORECASE)


def is_url(value: Any) -> bool:
    return isinstance(value, (str, Path)) and bool(_URL_RE.match(str(value)))


class Frame:
    """One unit of work handed to the predictor."""

    __slots__ = ("img", "path", "index", "total", "kind", "frame", "frames", "fps")

    def __init__(self, img, path, index, total, kind, frame=0, frames=0, fps=0.0):
        self.img = img  # BGR ndarray
        self.path = str(path)
        self.index = index  # 1-based source index
        self.total = total  # number of sources, 0 when unbounded
        self.kind = kind  # "image" | "video" | "stream"
        self.frame = frame  # 1-based frame number within a video
        self.frames = frames  # total frames, 0 when unknown
        self.fps = fps  # frames per second after vid_stride, 0 when unknown

    def prefix(self) -> str:
        """``"image 1/3 bus.jpg: "`` — the head of the verbose log line."""
        total = self.total or 1
        if self.kind == "image":
            return f"image {self.index}/{total} {self.path}: "
        where = f"(frame {self.frame}/{self.frames})" if self.frames else f"(frame {self.frame})"
        return f"{self.kind} {self.index}/{total} {where} {self.path}: "


def _pil_to_bgr(img) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"))
    return arr[:, :, ::-1].copy()


def _read_url_image(url: str) -> np.ndarray:
    import cv2

    from .downloads import read_url_bytes

    buf = np.frombuffer(read_url_bytes(url), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode an image from {url}")
    return img


def _expand(source: Any) -> list[tuple[str, Any]]:
    """Flatten one user-supplied source into ``(kind, payload)`` entries."""
    if isinstance(source, (list, tuple)):
        return [e for s in source for e in _expand(s)]
    if isinstance(source, np.ndarray):
        return [("array", source)]
    if hasattr(source, "convert") and hasattr(source, "size"):  # PIL.Image
        return [("array", _pil_to_bgr(source))]
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        return [("stream", int(source))]
    if is_url(source):
        url = str(source)
        kind = "url-image" if Path(url.split("?")[0]).suffix.lower() in IMG_FORMATS else "stream"
        return [(kind, url)]

    p = Path(str(source))
    if any(ch in str(source) for ch in "*?[") and not p.exists():
        files = sorted(_glob.glob(str(source), recursive=True))
        if not files:
            raise FileNotFoundError(f"no files match {source!r}")
        return [e for f in files for e in _expand(f)]
    if p.is_dir():
        files = sorted(
            f for f in p.rglob("*") if f.suffix.lower() in IMG_FORMATS | VID_FORMATS
        )
        if not files:
            raise FileNotFoundError(f"no images or videos under {p}")
        return [e for f in files for e in _expand(f)]
    if not p.exists():
        raise FileNotFoundError(f"source not found: {p}")
    suffix = p.suffix.lower()
    if suffix in IMG_FORMATS:
        return [("image", p)]
    if suffix in VID_FORMATS:
        return [("video", p)]
    if suffix == ".txt":
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return [e for ln in lines for e in _expand(ln if Path(ln).is_absolute() else p.parent / ln)]
    raise ValueError(f"unsupported source type: {p}")


class SourceLoader:
    """Iterates a resolved source as :class:`Frame` objects."""

    def __init__(self, source: Any, vid_stride: int = 1) -> None:
        self.entries = _expand(source)
        self.vid_stride = max(int(vid_stride), 1)
        self.total = len(self.entries)
        self.is_stream = any(k == "stream" for k, _ in self.entries)

    def __len__(self) -> int:
        return self.total

    def __iter__(self) -> Iterator[Frame]:
        import cv2

        for i, (kind, payload) in enumerate(self.entries, start=1):
            if kind == "array":
                yield Frame(payload, "image.jpg", i, self.total, "image")
            elif kind == "url-image":
                yield Frame(_read_url_image(payload), payload, i, self.total, "image")
            elif kind == "image":
                img = cv2.imread(str(payload))
                if img is None:
                    raise ValueError(f"could not read image: {payload}")
                yield Frame(img, payload, i, self.total, "image")
            else:  # video file, webcam index, or network stream
                cap = cv2.VideoCapture(payload if isinstance(payload, int) else str(payload))
                if not cap.isOpened():
                    raise ValueError(f"could not open {payload!r}")
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if kind == "video" else 0
                frames = max(frames, 0) // self.vid_stride
                fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                # a stride of 2 keeps every other frame, so playback halves too
                fps = fps / self.vid_stride if fps > 0 else 0.0
                label = "video" if kind == "video" else "stream"
                n = 0
                try:
                    while True:
                        ok, img = cap.read()
                        if not ok:
                            break
                        n += 1
                        if (n - 1) % self.vid_stride:
                            continue
                        yield Frame(
                            img,
                            f"webcam{payload}" if isinstance(payload, int) else payload,
                            i,
                            self.total,
                            label,
                            frame=(n - 1) // self.vid_stride + 1,
                            frames=frames,
                            fps=fps,
                        )
                finally:
                    cap.release()

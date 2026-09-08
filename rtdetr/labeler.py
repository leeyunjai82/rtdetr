# Apache-2.0
"""A small box-labelling tool: ``rtdetr label source=images/ names=can,bottle``.

It serves one page on localhost, lists the images in a folder, and writes the
same ``labels/*.txt`` files the trainer reads. The reason it exists rather than
pointing you at a real annotation suite: this package already has a detector, so
the first pass can be machine-made — hit *Auto-label*, fix what is wrong, move
on. Correcting boxes is much faster than drawing them.

Nothing here talks to the network beyond the browser on your own machine, and
the only dependencies are the standard library plus what the package already
needs.
"""

from __future__ import annotations

import json
import mimetypes
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .sources import IMG_FORMATS

PAGE = Path(__file__).with_name("label_app.html")


def label_path(image: Path, images_root: Path, labels_root: Path) -> Path:
    """``images/train/a.jpg`` -> ``labels/train/a.txt`` (mirroring the tree)."""
    return (labels_root / image.relative_to(images_root)).with_suffix(".txt")


def read_labels(path: Path) -> list[dict]:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{int(b['cls'])} {_clamp(b['cx']):.6f} {_clamp(b['cy']):.6f} "
        f"{_clamp(b['w']):.6f} {_clamp(b['h']):.6f}"
        for b in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class LabelSession:
    """The state one labelling run needs: which images, which classes, where to write."""

    def __init__(self, source, names, labels_dir=None, model=None, device="AUTO"):
        self.images_root = Path(source)
        if not self.images_root.is_dir():
            raise NotADirectoryError(f"{self.images_root} is not a folder of images")
        self.images = sorted(
            p for p in self.images_root.rglob("*") if p.suffix.lower() in IMG_FORMATS
        )
        if not self.images:
            raise FileNotFoundError(f"no images under {self.images_root}")
        self.names = list(names)
        self.labels_root = Path(labels_dir) if labels_dir else _sibling_labels(self.images_root)
        self.model_name = model
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    def label_file(self, index: int) -> Path:
        return label_path(self.images[index], self.images_root, self.labels_root)

    def state(self) -> dict:
        return {
            "names": self.names,
            "images": [
                {
                    "name": str(p.relative_to(self.images_root)),
                    "labelled": self.label_file(i).exists(),
                }
                for i, p in enumerate(self.images)
            ],
            "labels_dir": str(self.labels_root),
            "can_autolabel": self.model_name is not None,
        }

    def autolabel(self, index: int, conf: float = 0.35) -> list[dict]:
        """Predict boxes for one image, mapped onto this session's class list."""
        with self._lock:
            if self._model is None:
                from .model import RTDETR

                self._model = RTDETR(self.model_name, device=self.device, verbose=False)
            result = self._model.predict(str(self.images[index]), conf=conf, verbose=False)[0]

        lookup = {name.lower(): i for i, name in enumerate(self.names)}
        boxes = []
        for i in range(len(result.boxes)):
            name = result.name_of(result.boxes.cls[i]).lower()
            if name not in lookup:  # a class this project does not label
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

    def write_data_yaml(self, path: Path) -> Path:
        """Emit a data.yaml pointing at what was just labelled."""
        import yaml

        path.write_text(
            yaml.safe_dump(
                {
                    "path": str(self.images_root.parent.resolve()),
                    "train": self.images_root.name,
                    "val": self.images_root.name,
                    "names": dict(enumerate(self.names)),
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        return path


def _sibling_labels(images_root: Path) -> Path:
    """``…/images/train`` -> ``…/labels/train``; otherwise a labels/ next door."""
    parts = list(images_root.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return images_root.parent / "labels"


class Handler(BaseHTTPRequestHandler):
    session: LabelSession = None  # set by serve()

    def log_message(self, *args):  # keep the console for the app's own output
        pass

    # -- routing ------------------------------------------------------------

    def do_GET(self):
        route = urlparse(self.path).path
        if route in ("/", "/index.html"):
            return self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        if route == "/api/state":
            return self._json(self.session.state())
        if route.startswith("/api/image/"):
            index = self._index(route)
            path = self.session.images[index]
            kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            return self._send(path.read_bytes(), kind)
        if route.startswith("/api/labels/"):
            return self._json({"boxes": read_labels(self.session.label_file(self._index(route)))})
        self.send_error(404)

    def do_POST(self):
        route = urlparse(self.path).path
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0) or 0)
        payload = json.loads(body or b"{}")
        if route.startswith("/api/labels/"):
            index = self._index(route)
            write_labels(self.session.label_file(index), payload.get("boxes", []))
            return self._json({"saved": True, "path": str(self.session.label_file(index))})
        if route.startswith("/api/autolabel/"):
            index = self._index(route)
            try:
                boxes = self.session.autolabel(index, float(payload.get("conf", 0.35)))
            except Exception as exc:  # a missing model should not kill the tool
                return self._json({"error": str(exc)}, status=503)
            return self._json({"boxes": boxes})
        if route == "/api/data_yaml":
            target = Path(payload.get("path") or (self.session.images_root.parent / "data.yaml"))
            return self._json({"path": str(self.session.write_data_yaml(target))})
        self.send_error(404)

    # -- plumbing -----------------------------------------------------------

    def _index(self, route: str) -> int:
        index = int(unquote(route.rsplit("/", 1)[-1]))
        if not 0 <= index < len(self.session.images):
            raise IndexError(index)
        return index

    def _json(self, payload, status=200):
        self._send(json.dumps(payload).encode("utf-8"), "application/json", status)

    def _send(self, body: bytes, kind: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def serve(session: LabelSession, host="127.0.0.1", port=8000, open_browser=True):
    """Run the labelling server until interrupted. Returns the bound server."""
    handler = type("BoundHandler", (Handler,), {"session": session})
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        raise OSError(f"could not bind {host}:{port} ({exc}) — try port=8001") from exc
    url = f"http://{host}:{server.server_port}"
    print(f"[rtdetr] labelling {len(session.images)} images -> {session.labels_root}")
    print(f"[rtdetr] open {url}   (Ctrl+C to stop)")
    if open_browser:
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    return server


def run(session: LabelSession, host="127.0.0.1", port=8000, open_browser=True) -> int:
    server = serve(session, host, port, open_browser)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[rtdetr] stopped")
    finally:
        server.server_close()
    return 0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

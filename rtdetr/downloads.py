# Apache-2.0
"""Pretrained-weight download and cache — stdlib only (urllib), cached in ~/.rtdetr.

``RTDETR("rtdetr-r18")`` has to just work, so the first predict pulls the
OpenVINO IR for that name from the public mirror and keeps it under
``~/.rtdetr/`` (override with ``$RTDETR_HOME``). The mirror base URL is
``$RTDETR_ASSETS_URL``-overridable, which is also how an offline site points
the package at an internal copy.
"""

from __future__ import annotations

import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .errors import DownloadError, ModelNotFoundError

#: Where the released weights live. Layout: ``<base>/<name>/<file>``.
DEFAULT_ASSETS_URL = "https://huggingface.co/leeyunjai/ovkit-models/resolve/main"

#: Names the mirror knows, with the spellings users type mapped onto them.
MODEL_NAMES = ("rtdetr-r18", "rtdetr-r34", "rtdetr-r50")

_TIMEOUT = 30.0


def normalize_name(name: str) -> str:
    """``rtdetr_r18`` / ``RTDETR-R18`` / ``rtdetr-r18.pt`` -> ``rtdetr-r18``."""
    stem = str(name).strip().lower()
    for suffix in (".pt", ".xml", ".onnx"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.replace("_", "-")


def is_model_name(name: str) -> bool:
    return normalize_name(name) in MODEL_NAMES


def assets_url() -> str:
    return os.environ.get("RTDETR_ASSETS_URL", DEFAULT_ASSETS_URL).rstrip("/")


def cache_dir() -> Path:
    root = Path(os.environ.get("RTDETR_HOME", Path.home() / ".rtdetr")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def read_url_bytes(url: str, timeout: float = _TIMEOUT) -> bytes:
    """Fetch a URL into memory (used for image URLs as well as weights)."""
    request = urllib.request.Request(url, headers={"User-Agent": "rtdetr"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.URLError as exc:
        raise DownloadError(f"could not fetch {url}: {exc}") from exc


def download(url: str, dest: Path, progress: bool = True) -> Path:
    """Download ``url`` to ``dest`` (atomically). Existing files are reused."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "rtdetr"})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response, open(tmp, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while chunk := response.read(1 << 16):
                out.write(chunk)
                done += len(chunk)
                if progress and total and sys.stderr.isatty():
                    pct = 100 * done / total
                    print(f"\rdownloading {dest.name} {pct:5.1f}%", end="", file=sys.stderr)
        if progress and sys.stderr.isatty():
            print(f"\rdownloading {dest.name}  done ", file=sys.stderr)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"could not fetch {url}: {exc}") from exc
    shutil.move(str(tmp), str(dest))
    return dest


def _asset(name: str, filename: str, required: bool = True) -> Path | None:
    dest = cache_dir() / name / filename
    if dest.exists() and dest.stat().st_size:
        return dest
    url = f"{assets_url()}/{name}/{filename}"
    try:
        return download(url, dest)
    except DownloadError:
        if required:
            raise
        return None


def download_ir(name: str) -> Path:
    """Fetch ``<name>.xml`` (+ ``.bin``, + ``labels.txt``); returns the .xml path."""
    name = normalize_name(name)
    if name not in MODEL_NAMES:
        raise ModelNotFoundError(_unknown_name_message(name))
    try:
        xml = _asset(name, f"{name}.xml")
        _asset(name, f"{name}.bin")
    except DownloadError as exc:
        raise ModelNotFoundError(
            f"'{name}' is not on the mirror yet ({exc}). Pass a .pt/.xml path, "
            f"point $RTDETR_ASSETS_URL at your own copy, or train it yourself: "
            f"RTDETR('{name}').train(data='data.yaml')"
        ) from exc
    _asset(name, "labels.txt", required=False)
    return xml


def download_checkpoint(name: str) -> Path:
    """Fetch ``<name>.pt`` — the torch weights used to fine-tune or validate."""
    name = normalize_name(name)
    if name not in MODEL_NAMES:
        raise ModelNotFoundError(_unknown_name_message(name))
    try:
        return _asset(name, f"{name}.pt")
    except DownloadError as exc:
        raise ModelNotFoundError(
            f"no pretrained checkpoint for '{name}' on the mirror ({exc}). "
            f"Training starts from an ImageNet backbone instead: "
            f"RTDETR('{name}').train(data='data.yaml')"
        ) from exc


def _unknown_name_message(name: str) -> str:
    return (
        f"unknown model '{name}'. Known names: {', '.join(MODEL_NAMES)}. "
        f"You can also pass a path to your own .pt / .xml / .onnx."
    )

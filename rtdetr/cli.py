# Apache-2.0
"""``rtdetr <mode> key=value ...`` — the command line for this package.

    rtdetr predict model=rtdetr-r18 source=bus.jpg conf=0.5
    rtdetr train   model=rtdetr-r18 data=data.yaml epochs=100 imgsz=640
    rtdetr val     model=best.pt data=data.yaml
    rtdetr export  model=best.pt format=openvino half=true
  rtdetr label   source=images/ names=can,bottle model=rtdetr-r18
  rtdetr studio  port=8080
    rtdetr track   model=best.pt source=video.mp4
"""

from __future__ import annotations

import sys
from typing import Any

MODES = ("predict", "track", "train", "val", "export", "label", "studio")

HELP = """rtdetr — real-time detection transformer, Apache-2.0

Usage:  rtdetr <mode> key=value ...

Modes:
  predict   run detection on an image / folder / video / url / camera
  track     predict and keep an id on each box across frames
  train     train on a data.yaml dataset (images/ + labels/*.txt)
  val       COCO-style mAP50 / mAP50-95 on the val split
  export    write an OpenVINO IR (or ONNX) next to the checkpoint
  label     draw boxes in the browser, saving labels the trainer can read
  studio    the web app: upload a dataset, train, watch, download

Common keys:
  model=rtdetr-r18|best.pt|model.xml   source=bus.jpg|dir|video.mp4|0|url
  data=data.yaml  epochs=100  imgsz=640  batch=8  conf=0.25  device=0|cpu|AUTO
  project=runs  name=predict  save=true  show=false  format=openvino  half=true

Examples:
  rtdetr predict model=rtdetr-r18 source=bus.jpg conf=0.5
  rtdetr train   model=rtdetr-r18 data=data.yaml epochs=100
  rtdetr val     model=best.pt data=data.yaml
  rtdetr export  model=best.pt format=openvino half=true
  rtdetr label   source=images/ names=can,bottle model=rtdetr-r18
  rtdetr studio  port=8080
"""


def parse_value(text: str) -> Any:
    """``true`` -> True, ``3`` -> 3, ``0.5`` -> 0.5, ``0,1`` -> [0, 1], else str."""
    low = text.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    stripped = text.strip("[]")
    if "," in stripped:
        return [parse_value(part.strip()) for part in stripped.split(",") if part.strip()]
    return text


def parse_args(argv: list[str]) -> tuple[str, dict[str, Any]]:
    """-> ``(mode, overrides)``. The mode may be positional or ``mode=predict``."""
    mode = ""
    overrides: dict[str, Any] = {}
    for arg in argv:
        if "=" in arg:
            key, _, value = arg.partition("=")
            key = key.strip().lstrip("-")
            if key == "mode":
                mode = str(value)
            else:
                overrides[key] = parse_value(value)
        elif not mode:
            mode = arg
        else:
            raise SystemExit(f"unexpected argument {arg!r} — arguments are key=value pairs")
    return mode, overrides


def _take(overrides: dict[str, Any], key: str, default: Any = None) -> Any:
    return overrides.pop(key, default)


def _label(overrides: dict[str, Any], model_name: Any, device: Any, verbose: Any) -> int:
    """``rtdetr label source=images/ names=a,b`` — the browser labelling tool."""
    from .labeler import LabelSession, run

    source = _take(overrides, "source")
    if source is None:
        print("label needs source=path/to/images", file=sys.stderr)
        return 2
    names = _take(overrides, "names")
    if names is None:
        print("label needs names=class1,class2 (the classes you are drawing)", file=sys.stderr)
        return 2
    names = [str(n) for n in (names if isinstance(names, list) else [names])]

    # model= doubles as the auto-labeller; autolabel=false turns that off
    autolabel_with = model_name if _take(overrides, "autolabel", True) else None
    session = LabelSession(
        source,
        names,
        labels_dir=_take(overrides, "labels"),
        model=autolabel_with,
        device=device or "AUTO",
    )
    return run(
        session,
        host=str(_take(overrides, "host", "127.0.0.1")),
        port=int(_take(overrides, "port", 8000)),
        open_browser=bool(_take(overrides, "open", True)),
    )


def _studio(overrides: dict[str, Any]) -> int:
    """``rtdetr studio`` — the web app (needs the [studio] extra)."""
    try:
        import uvicorn
    except ImportError:
        print("the studio needs its extra: pip install 'rtdetr[studio]'", file=sys.stderr)
        return 2

    import os

    if "data" in overrides:
        os.environ["RTDETR_STUDIO_HOME"] = str(_take(overrides, "data"))
    host = str(_take(overrides, "host", "127.0.0.1"))
    port = int(_take(overrides, "port", 8080))
    print(f"[rtdetr] studio on http://{host}:{port}")
    uvicorn.run("rtdetr.studio.app:app", host=host, port=port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(HELP)
        return 0
    if argv[0] in ("-v", "--version", "version"):
        from . import __version__

        print(__version__)
        return 0

    mode, overrides = parse_args(argv)
    if mode not in MODES:
        print(f"unknown mode {mode!r}. Modes: {', '.join(MODES)}\n", file=sys.stderr)
        print(HELP, file=sys.stderr)
        return 2

    from .model import RTDETR

    model_name = _take(overrides, "model", "rtdetr-r18")
    device = _take(overrides, "device")
    verbose = _take(overrides, "verbose", True)

    if mode in ("predict", "track"):
        model = RTDETR(model_name, device=device or "AUTO", verbose=verbose)
        source = _take(overrides, "source")
        if source is None:
            print(
                "predict needs source=… (image, folder, video, url, camera index)",
                file=sys.stderr,
            )
            return 2
        overrides.setdefault("save", True)
        overrides["stream"] = False
        runner = model.predict if mode == "predict" else model.track
        results = runner(source, **overrides)
        print(f"{len(results)} result(s)")
        return 0

    if mode == "label":
        return _label(overrides, model_name, device, verbose)
    if mode == "studio":
        return _studio(overrides)

    model = RTDETR(model_name, verbose=verbose)
    if mode == "train":
        data = _take(overrides, "data")
        if data is None:
            print("train needs data=path/to/data.yaml", file=sys.stderr)
            return 2
        best = model.train(data=data, device=device, **overrides)
        print(best)
        return 0
    if mode == "val":
        data = _take(overrides, "data")
        if data is None:
            print("val needs data=path/to/data.yaml", file=sys.stderr)
            return 2
        metrics = model.val(data=data, device=device, **overrides)
        if not verbose:  # model.val() already printed the line when verbose
            print(f"mAP50 {metrics.box.map50:.4f}  mAP50-95 {metrics.box.map:.4f}")
        return 0

    path = model.export(**overrides)
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

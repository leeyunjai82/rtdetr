#!/usr/bin/env python3
# Apache-2.0
"""Does D-FINE run on this machine's OpenVINO devices? A one-off probe.

    python tools/dfine_probe.py                 # D-FINE-S, every device OpenVINO sees
    python tools/dfine_probe.py --size n        # n / s / m / l / x
    python tools/dfine_probe.py --image my.jpg

Fetches the D-FINE code and its COCO checkpoint (both Apache-2.0) from GitHub
into ~/.rtdetr/dfine-probe/, exports two IRs — the whole model with its
post-processing, and the network alone — and runs each on CPU, GPU and NPU.
Every device is checked against the PyTorch boxes, then timed.

Needs only what `pip install "rtdetr[train]"` already brought: torch,
torchvision, openvino, opencv, numpy, pyyaml. D-FINE's training-only imports
(tensorboard, faster-coco-eval, calflops, transformers, loguru) are stubbed.
"""

from __future__ import annotations

import argparse
import importlib.abc
import importlib.machinery
import io
import statistics
import sys
import time
import types
import urllib.request
import warnings
import zipfile
from pathlib import Path

CODE = "https://codeload.github.com/Peterande/D-FINE/zip/refs/heads/master"
WEIGHTS = "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_{}_coco.pth"
SAMPLE = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"
STUBBED = ("tensorboard", "torch.utils.tensorboard", "faster_coco_eval", "calflops",
           "transformers", "loguru")


class _Anything(types.ModuleType):
    """Stands in for a training-only package: every attribute is a do-nothing class."""

    __path__: list = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {"__init__": lambda self, *a, **k: None,
                               "__call__": lambda self, *a, **k: None,
                               "__getattr__": lambda self, n: (lambda *a, **k: None)})


class _Stubs(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if any(name == s or name.startswith(s + ".") for s in STUBBED):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        return _Anything(spec.name)

    def exec_module(self, module):
        pass


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"downloading {url}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            dest.write_bytes(r.read())
    return dest


def fetch_code(root: Path) -> Path:
    src = root / "D-FINE-master"
    if not src.exists():
        print(f"downloading {CODE}")
        with urllib.request.urlopen(CODE, timeout=120) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(root)
    return src


def iou(a, b) -> float:
    import numpy as np

    x1, y1 = np.maximum(a[0], b[0]), np.maximum(a[1], b[1])
    x2, y2 = np.minimum(a[2], b[2]), np.minimum(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def agreement(ref, got, conf: float) -> str:
    """How closely one device's boxes match the reference, above ``conf``."""
    import numpy as np

    (rl, rb, rs), (gl, gb, gs) = ref, got
    rk, gk = rs > conf, gs > conf
    ious, dconf = [], []
    for label, box, score in zip(rl[rk], rb[rk], rs[rk], strict=True):
        same = [(iou(box, b), abs(score - s))
                for l2, b, s in zip(gl[gk], gb[gk], gs[gk], strict=True) if l2 == label]
        if same:
            best = max(same)
            ious.append(best[0])
            dconf.append(best[1])
    if not ious:
        return f"reference {int(rk.sum())} boxes / this device {int(gk.sum())} — nothing to match"
    return (f"boxes {int(rk.sum())} -> {int(gk.sum())}, matched {len(ious)}, "
            f"mean IoU {np.mean(ious):.3f} (worst {np.min(ious):.3f}), "
            f"conf within {np.max(dconf):.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", default="s", choices=list("nsmlx"))
    parser.add_argument("--image", help="a picture to test on (default: an OpenCV sample frame)")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--devices", help="comma-separated, default: every device OpenVINO sees")
    parser.add_argument("--dir", default=str(Path.home() / ".rtdetr" / "dfine-probe"))
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")      # the tracer's notes about D-FINE's shape checks

    import cv2
    import numpy as np
    import openvino as ov
    import torch
    import torch.nn as nn

    root = Path(args.dir).expanduser()
    src = fetch_code(root)
    ckpt_path = fetch(WEIGHTS.format(args.size), root / f"dfine_{args.size}_coco.pth")
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"could not read {args.image}")
    else:
        cap = cv2.VideoCapture(str(fetch(SAMPLE, root / "vtest.avi")))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("could not read the sample clip")

    sys.meta_path.insert(0, _Stubs())
    sys.path.insert(0, str(src))
    from src.core import YAMLConfig  # noqa: E402 — D-FINE's own loader

    cfg = YAMLConfig(str(src / "configs" / "dfine" / f"dfine_hgnetv2_{args.size}_coco.yml"))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg.model.load_state_dict(ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"])
    net, post = cfg.model.deploy().eval(), cfg.postprocessor.deploy().eval()

    class Full(nn.Module):          # network + D-FINE's own top-k decode
        def __init__(self):
            super().__init__()
            self.net, self.post = net, post

        def forward(self, images, sizes):
            return self.post(self.net(images), sizes)

    class Raw(nn.Module):           # network only: decode happens outside
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, images):
            out = self.net(images)
            return out["pred_logits"], out["pred_boxes"]

    h, w = frame.shape[:2]
    rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (640, 640))
    x = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1) / 255)[None]
    size = torch.tensor([[w, h]])

    def decode_raw(logits, boxes, top=300):
        """Sigmoid scores, best (query, class) pairs, cxcywh -> pixel xyxy."""
        scores = 1 / (1 + np.exp(-logits[0]))
        flat = scores.reshape(-1)
        idx = np.argsort(-flat)[:top]
        q, cls = np.divmod(idx, scores.shape[1])
        cx, cy, bw, bh = boxes[0][q].T
        xyxy = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h,
                         (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
        return cls, xyxy, flat[idx]

    with torch.no_grad():
        labels, boxes, scores = Full()(x, size)
        ref_full = (labels[0].numpy(), boxes[0].numpy(), scores[0].numpy())
        logits, pboxes = Raw()(x)
        ref_raw = decode_raw(logits.numpy(), pboxes.numpy())
    keep = ref_full[2] > args.conf
    print(f"\nD-FINE-{args.size.upper()} in PyTorch: {int(keep.sum())} boxes above {args.conf}, "
          f"classes {sorted(set(ref_full[0][keep].tolist()))}")

    core = ov.Core()
    irs = {}
    for name, module, inputs, in_names, out_names in (
        ("full", Full(), (x, size), ["images", "orig_target_sizes"], ["labels", "boxes", "scores"]),
        ("raw", Raw(), (x,), ["images"], ["pred_logits", "pred_boxes"]),
    ):
        onnx_path = root / f"dfine_{args.size}_{name}.onnx"
        torch.onnx.export(module, inputs, str(onnx_path), input_names=in_names,
                          output_names=out_names, opset_version=17, do_constant_folding=True,
                          dynamo=False)
        irs[name] = core.read_model(str(onnx_path))
        ov.save_model(irs[name], str(root / f"dfine_{args.size}_{name}.xml"))
    print(f"IRs written to {root}")

    available = core.available_devices
    wanted = args.devices.split(",") if args.devices else [d for d in ("CPU", "GPU", "NPU") if any(
        a.startswith(d) for a in available)]
    print(f"OpenVINO {ov.__version__}; devices {available}\n")

    for device in wanted:
        for name in ("full", "raw"):
            label = f"{device:<4} {name:<4}"
            try:
                t0 = time.perf_counter()
                compiled = core.compile_model(irs[name], device)
                first = time.perf_counter() - t0
                req = compiled.create_infer_request()
                feed = {"images": x.numpy()}
                if name == "full":
                    feed["orig_target_sizes"] = size.numpy()
                res = req.infer(feed)
                outs = [res[compiled.output(i)] for i in range(len(compiled.outputs))]
                if name == "full":
                    got = (outs[0][0], outs[1][0], outs[2][0])
                    verdict = agreement(ref_full, got, args.conf)
                else:
                    got = decode_raw(outs[0], outs[1])
                    verdict = agreement(ref_raw, got, args.conf)
                for _ in range(5):
                    req.infer(feed)
                times = []
                for _ in range(30):
                    t = time.perf_counter()
                    req.infer(feed)
                    times.append((time.perf_counter() - t) * 1000)
                ms = statistics.median(times)
                print(f"{label}  {ms:6.1f} ms ({1000 / ms:5.1f} FPS)  "
                      f"compile {first:5.1f}s  |  {verdict}")
            except Exception as exc:  # noqa: BLE001 — the point is to see what fails
                message = str(exc).strip().splitlines()
                print(f"{label}  FAILED: {message[0] if message else type(exc).__name__}")
                for line in message[1:6]:
                    print(f"{'':12}{line}")
    print("\n'full' = network + D-FINE's decode inside the IR; "
          "'raw' = network only, decoded in numpy.")
    print("IoU/conf compare each device with PyTorch on the same frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

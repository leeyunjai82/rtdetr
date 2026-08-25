#!/usr/bin/env python3
# Apache-2.0
"""Build the weight-mirror layout that ``RTDETR("rtdetr-r18")`` downloads from.

    python tools/build_mirror.py --out mirror                 # all variants
    python tools/build_mirror.py --out mirror --variants r18   # just one

For each variant this downloads the Apache-2.0 COCO checkpoint from the original
RT-DETR release, loads it (strict — this package's layout matches), and writes::

    mirror/rtdetr-r18/rtdetr-r18.pt      torch weights, for train/val/export
    mirror/rtdetr-r18/rtdetr-r18.xml     OpenVINO IR, what predict downloads
    mirror/rtdetr-r18/rtdetr-r18.bin
    mirror/rtdetr-r18/labels.txt         COCO class names, one per line

Upload that tree to the mirror named by ``$RTDETR_ASSETS_URL`` (by default the
Hugging Face repo in rtdetr/downloads.py), keeping the directory names:

    huggingface-cli upload leeyunjai/rtdetr mirror . --repo-type=model

Everything runs on CPU in a few minutes; there is no training involved.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import convert_official  # noqa: E402  (same directory)


def build(variant: str, out_root: Path, imgsz: int, half: bool, keep_onnx: bool) -> Path:
    from rtdetr import RTDETR

    name = f"rtdetr-{variant}"
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)

    checkpoint = out / f"{name}.pt"
    convert_official.main(["--variant", variant, "--out", str(checkpoint), "--imgsz", str(imgsz)])

    model = RTDETR(str(checkpoint), verbose=False)
    xml = model.export(format="openvino", imgsz=imgsz, half=half, out_dir=out)
    if not keep_onnx:
        (out / f"{name}.onnx").unlink(missing_ok=True)
    print(f"{name}: {', '.join(sorted(p.name for p in out.iterdir()))}")
    return xml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="mirror", help="output directory")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(convert_official.OFFICIAL),
        choices=("r18", "r34", "r50"),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="FP16 IR (smaller, same accuracy)")
    parser.add_argument("--keep-onnx", action="store_true", help="leave the intermediate .onnx")
    args = parser.parse_args(argv)

    out_root = Path(args.out)
    for variant in args.variants:
        build(variant, out_root, args.imgsz, args.half, args.keep_onnx)
    print(f"\nmirror ready at {out_root}/ — upload it keeping these directory names")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

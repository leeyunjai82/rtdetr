#!/usr/bin/env python3
# Apache-2.0
"""Convert lyuwenyu/RT-DETR (Apache-2.0) COCO weights into an rtdetr checkpoint.

    python tools/convert_official.py --weights rtdetr_r18vd_dec3_6x_coco_from_paddle.pth \
        --variant r18 --out rtdetr-r18.pt

The original release and this package share the RT-DETR design but not every
module: our decoder uses plain multi-head cross-attention instead of deformable
attention (it keeps ONNX/OpenVINO export dependency-free), and our CCFF fusion
block is narrower. So the mapping below is honest about what it can move:

  * backbone residual stages          -> transferred
  * encoder input projections + AIFI  -> transferred
  * FPN/PAN lateral + downsample convs-> transferred
  * CCFF fusion blocks                -> shapes differ, skipped
  * decoder self-attention, FFN, heads-> transferred
  * decoder cross-attention           -> deformable, no counterpart, skipped

The script prints exactly how many tensors landed. A converted checkpoint is a
*warm start*: fine-tune it on COCO (or your own data) before publishing it as
"pretrained". Use --min-coverage in CI if you want the conversion to fail loudly
when a future upstream rename silently drops half the weights.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def rename(key: str) -> str | None:
    """Official key -> our key, or ``None`` when nothing corresponds."""
    k = key

    # ---- backbone: PResNet res_layers -> torchvision-style layer1..layer4 ----
    m = re.match(r"backbone\.res_layers\.(\d)\.blocks\.(\d+)\.(.+)", k)
    if m:
        stage, block, rest = int(m.group(1)) + 1, m.group(2), m.group(3)
        rest = (
            rest.replace("branch2a.conv", "conv1")
            .replace("branch2a.norm", "bn1")
            .replace("branch2b.conv", "conv2")
            .replace("branch2b.norm", "bn2")
            .replace("branch2c.conv", "conv3")
            .replace("branch2c.norm", "bn3")
            .replace("short.conv", "downsample.0")
            .replace("short.norm", "downsample.1")
        )
        if rest.startswith("branch") or rest.startswith("short"):
            return None
        return f"backbone.layer{stage}.{block}.{rest}"

    # the vd stem (three 3x3 convs) has no counterpart in a torchvision stem
    if k.startswith("backbone.conv1"):
        return None

    # ---- encoder ----
    if k.startswith("encoder.input_proj."):
        return k  # Conv+BN Sequential, identical layout
    m = re.match(r"encoder\.encoder\.0\.layers\.(\d+)\.(.+)", k)
    if m:
        return f"encoder.aifi.encoder.layers.{m.group(1)}.{m.group(2)}"
    m = re.match(r"encoder\.lateral_convs\.(\d)\.(conv|norm)\.(.+)", k)
    if m:
        which = "conv" if m.group(2) == "conv" else "bn"
        return f"encoder.lateral{int(m.group(1)) + 1}.{which}.{m.group(3)}"
    m = re.match(r"encoder\.downsample_convs\.(\d)\.(conv|norm)\.(.+)", k)
    if m:
        which = "conv" if m.group(2) == "conv" else "bn"
        return f"encoder.down{int(m.group(1)) + 1}.{which}.{m.group(3)}"
    if k.startswith(("encoder.fpn_blocks.", "encoder.pan_blocks.")):
        return None  # CSPRepLayer vs our narrower RepBlock — shapes disagree

    # ---- decoder ----
    m = re.match(r"decoder\.decoder\.layers\.(\d+)\.(.+)", k)
    if m:
        i, rest = m.group(1), m.group(2)
        if rest.startswith("cross_attn"):
            return None  # deformable attention, nothing to map onto
        rest = rest.replace("linear1.", "ffn.0.").replace("linear2.", "ffn.2.")
        return f"decoder.layers.{i}.{rest}"
    m = re.match(r"decoder\.dec_score_head\.(\d+)\.(.+)", k)
    if m:
        return f"decoder.dec_score.{m.group(1)}.{m.group(2)}"
    m = re.match(r"decoder\.dec_bbox_head\.(\d+)\.(.+)", k)
    if m:
        return f"decoder.dec_bbox.{m.group(1)}.{m.group(2)}"
    if k.startswith("decoder.enc_score_head."):
        return k.replace("decoder.enc_score_head.", "decoder.enc_score.")
    if k.startswith("decoder.enc_bbox_head."):
        return k.replace("decoder.enc_bbox_head.", "decoder.enc_bbox.")
    if k.startswith("decoder.query_pos_head."):
        return k
    if k.startswith("decoder.enc_output.0."):
        return k.replace("decoder.enc_output.0.", "decoder.tgt_proj.")
    if k.startswith("decoder.enc_output.1."):
        return k.replace("decoder.enc_output.1.", "decoder.enc_norm.")
    return None


def load_official_state(path: Path) -> dict:
    import torch

    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    for key in ("ema", "model", "state_dict"):
        if isinstance(blob, dict) and key in blob:
            blob = blob[key]
            if isinstance(blob, dict) and "module" in blob:  # ema wrapper
                blob = blob["module"]
            break
    if not isinstance(blob, dict):
        raise SystemExit(f"{path} does not contain a state_dict")
    return {k: v for k, v in blob.items() if hasattr(v, "shape")}


def convert(args: argparse.Namespace) -> int:
    import torch

    from rtdetr.nn.rtdetr_net import RTDETRNet

    weights = Path(args.weights)
    if str(args.weights).startswith(("http://", "https://")):
        from rtdetr.downloads import cache_dir, download

        weights = download(str(args.weights), cache_dir() / "official" / Path(args.weights).name)

    official = load_official_state(weights)
    names = _read_names(args.names)
    net = RTDETRNet(args.variant, num_classes=len(names), pretrained_backbone=False)
    target = net.state_dict()

    moved, shape_mismatch, unmapped = {}, [], []
    for key, tensor in official.items():
        mapped = rename(key)
        if mapped is None or mapped not in target:
            unmapped.append(key)
            continue
        if tuple(target[mapped].shape) != tuple(tensor.shape):
            shape_mismatch.append((key, tuple(tensor.shape), tuple(target[mapped].shape)))
            continue
        moved[mapped] = tensor

    target.update(moved)
    net.load_state_dict(target)

    coverage = 100.0 * len(moved) / max(len(target), 1)
    print(f"transferred {len(moved)}/{len(target)} tensors ({coverage:.1f}% of our model)")
    print(f"  official tensors with no counterpart: {len(unmapped)}")
    print(f"  mapped but shape-incompatible:        {len(shape_mismatch)}")
    if args.report:
        for key in unmapped[: args.report]:
            print(f"    unmapped  {key}")
        for key, got, want in shape_mismatch[: args.report]:
            print(f"    mismatch  {key}: {got} != {want}")
    if coverage < args.min_coverage:
        print(
            f"coverage {coverage:.1f}% is below --min-coverage {args.min_coverage}% — "
            f"upstream layer names probably changed",
            file=sys.stderr,
        )
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": net.state_dict(),
            "variant": args.variant,
            "num_classes": len(names),
            "names": {i: n for i, n in enumerate(names)},
            "epoch": -1,
            "imgsz": args.imgsz,
            "source": f"converted from {weights.name}",
        },
        out,
    )
    print(f"wrote {out}")
    print(
        "this is a warm start, not a finished model: the decoder's cross-attention "
        "and the CCFF blocks are freshly initialised, so fine-tune before release."
    )
    return 0


def _read_names(spec: str | None) -> list[str]:
    if not spec:
        return COCO_NAMES
    path = Path(spec)
    if path.suffix == ".json":
        table = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(table, dict):
            return [table[k] for k in sorted(table, key=int)]
        return list(table)
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", required=True, help="official .pth path or URL")
    parser.add_argument("--variant", default="r18", choices=("r18", "r34", "r50"))
    parser.add_argument("--out", default="rtdetr-converted.pt")
    parser.add_argument("--names", help="labels.txt or names.json (default: COCO 80)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--min-coverage", type=float, default=0.0, help="fail below this %%")
    parser.add_argument("--report", type=int, default=0, help="print N example keys per bucket")
    return convert(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

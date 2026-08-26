# Apache-2.0
"""Export: RTDETRNet -> ONNX (opset 17) -> OpenVINO IR, static shape, FP16 optional."""

from __future__ import annotations

import json
import warnings
from pathlib import Path


def _write_labels(out_dir: Path, fname: str, names) -> dict[int, str]:
    """Write ``labels.txt`` (one class per line) + ``<fname>.names.json``.

    labels.txt beside the IR is how downstream runtimes (ovkit among them)
    discover class names, so a freshly trained model answers "my-class 0.91"
    with no extra wiring.
    """
    table = {int(k): str(v) for k, v in (names or {}).items()}
    (out_dir / f"{fname}.names.json").write_text(
        json.dumps(table, ensure_ascii=False), encoding="utf-8"
    )
    if table:
        lines = [table.get(i, f"class_{i}") for i in range(max(table) + 1)]
        (out_dir / "labels.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table


def export_onnx(net, names, imgsz=640, out_dir=".", fname="rtdetr", half=False, verbose=True):
    """Write ``<out_dir>/<fname>.onnx`` (plus labels). Returns the .onnx path."""
    import torch

    from .nn.rtdetr_net import DeployWrapper

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{fname}.onnx"

    wrapper = DeployWrapper(net).eval().cpu()
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    with warnings.catch_warnings():
        # The graph is exported at a fixed input size on purpose, so the tracer
        # baking shape-dependent constants (the query count) in is what we want.
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_path),
            input_names=["images"],
            output_names=["boxes", "scores"],
            opset_version=17,
            dynamo=False,
        )
    _write_labels(out_dir, fname, names)
    if verbose:
        print(f"[rtdetr] exported: {onnx_path}")
    return onnx_path


def export_openvino(net, names, imgsz=640, out_dir=".", fname="rtdetr", half=False, verbose=True):
    """Write ``<out_dir>/<fname>.xml`` (+ .bin, + labels). Returns the .xml path."""
    import openvino as ov

    out_dir = Path(out_dir)
    onnx_path = export_onnx(
        net, names, imgsz=imgsz, out_dir=out_dir, fname=fname, verbose=False
    )
    xml_path = out_dir / f"{fname}.xml"

    model = ov.convert_model(str(onnx_path))
    ov.save_model(model, str(xml_path), compress_to_fp16=half)
    if verbose:
        print(f"[rtdetr] exported: {xml_path} ({'FP16' if half else 'FP32'})")
    return xml_path

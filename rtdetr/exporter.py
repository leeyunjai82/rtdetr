# Apache-2.0
"""Export: RTDETRNet -> ONNX (opset 17) -> OpenVINO IR, static shape, FP16 optional."""

from __future__ import annotations

import json
import sys
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


def calibration_images(data, imgsz, samples=300, verbose=True):
    """Preprocessed tensors for INT8 calibration, from anything we can read.

    ``data`` may be a data.yaml (its val split is used), a folder, a glob, a
    video, a camera index, or a list of paths. No labels are involved — INT8
    calibration only watches activation ranges.
    """
    from .predictor import preprocess_image
    from .sources import SourceLoader

    source = data
    if isinstance(data, (str, Path)) and str(data).endswith((".yaml", ".yml")):
        from .data.dataset import load_data_yaml

        cfg = load_data_yaml(data)
        split = cfg.get("val") or cfg.get("train")
        if split is None:
            raise ValueError(f"{data} has neither a val nor a train split to calibrate on")
        source = str(Path(cfg["root"]) / split)

    tensors = []
    for frame in SourceLoader(source, vid_stride=1):
        tensors.append(preprocess_image(frame.img, imgsz))
        if len(tensors) >= samples:
            break
    if not tensors:
        raise ValueError(f"no images to calibrate on in {data!r}")
    if verbose:
        print(f"[rtdetr] calibrating on {len(tensors)} images from {data}")
    if len(tensors) < 100:
        print(
            f"[rtdetr] warning: only {len(tensors)} calibration images. INT8 accuracy "
            f"follows what it saw — measured here, calibrating on street frames alone "
            f"dropped an unrelated photo from 0.95 to 0.35 confidence. Feed it 100-300 "
            f"frames covering the scenes you will actually run on.",
            file=sys.stderr,
        )
    return tensors


def quantize_int8(model, data, imgsz, samples=300, verbose=True):
    """Post-training INT8 quantisation of an OpenVINO model (needs nncf)."""
    try:
        import nncf
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(
            "INT8 export needs nncf: pip install 'rtdetr[int8]'"
        ) from exc

    if data is None:
        raise ValueError(
            "int8=True needs calibration images: pass data=... — a folder, a glob, "
            "a video, a camera clip, or a data.yaml. Labels are not used; 100-300 "
            "frames from the scene you will deploy in work best."
        )
    tensors = calibration_images(data, imgsz, samples, verbose=verbose)
    dataset = nncf.Dataset(tensors)
    # TRANSFORMER keeps attention in a form the quantiser handles sanely
    return nncf.quantize(
        model, dataset, model_type=nncf.ModelType.TRANSFORMER, subset_size=len(tensors)
    )


def export_openvino(
    net,
    names,
    imgsz=640,
    out_dir=".",
    fname="rtdetr",
    half=False,
    verbose=True,
    int8=False,
    data=None,
    calib_samples=300,
):
    """Write ``<out_dir>/<fname>.xml`` (+ .bin, + labels). Returns the .xml path."""
    import openvino as ov

    out_dir = Path(out_dir)
    onnx_path = export_onnx(
        net, names, imgsz=imgsz, out_dir=out_dir, fname=fname, verbose=False
    )
    xml_path = out_dir / f"{fname}.xml"

    model = ov.convert_model(str(onnx_path))
    if int8:
        model = quantize_int8(model, data, imgsz, calib_samples, verbose=verbose)
    ov.save_model(model, str(xml_path), compress_to_fp16=half and not int8)
    if verbose:
        precision = "INT8" if int8 else ("FP16" if half else "FP32")
        print(f"[rtdetr] exported: {xml_path} ({precision})")
    return xml_path

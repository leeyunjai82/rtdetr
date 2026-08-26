# Apache-2.0
"""RT-DETR — an Ultralytics-style detector that is 100% Apache-2.0.

    pip install rtdetr

    from rtdetr import RTDETR

    model = RTDETR("rtdetr-r18")                     # weights fetched from the mirror
    results = model("bus.jpg")                       # list[Results]
    results[0].boxes.xyxy, results[0].boxes.conf     # plain numpy
    results[0].save()

    model.train(data="data.yaml", epochs=100)        # YOLO-format labels
    model.val(data="data.yaml").box.map50            # COCO-style mAP
    model.export(format="openvino", half=True)       # IR + labels.txt

Network, loss, trainer, validator and exporter are original implementations —
no Ultralytics code and no AGPL weights anywhere, so this package can ship
inside a product. Inference needs numpy/opencv/openvino/pyyaml; training adds
``pip install "rtdetr[train]"`` (torch, torchvision, scipy, onnx).
"""

from __future__ import annotations

__version__ = "0.3.0"

from .errors import DownloadError, ModelNotFoundError, RTDETRError
from .metrics import BoxMetrics, DetMetrics
from .model import RTDETR
from .results import Boxes, Results

__all__ = [
    "RTDETR",
    "Results",
    "Boxes",
    "DetMetrics",
    "BoxMetrics",
    "RTDETRError",
    "ModelNotFoundError",
    "DownloadError",
    "__version__",
]

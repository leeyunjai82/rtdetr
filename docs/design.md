# Design notes and provenance

Back to the [README](../README.md).

## Design notes

* **Backbone** PResNet-vd 18/34/50 — three-conv stem, average-pooled shortcuts.
* **Encoder** AIFI transformer on the top level + CCFF (CSPRepLayer) FPN/PAN fusion.
* **Decoder** two-stage: dense heads pick the top-300 encoder tokens as queries,
  3/4/6 layers refine them with multi-scale deformable cross-attention.
* **Loss** Hungarian matching, varifocal + L1 + GIoU, over every decoder layer
  and the encoder's own proposals.
* **Validation** COCO-style mAP50 / mAP50-95, 101-point interpolation, no
  pycocotools dependency.
* **Inference** OpenVINO, plain resize (no letterbox) to match training.

The network definition under `rtdetr/nn/` is adapted from the RT-DETR reference
implementation ([lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR),
Apache-2.0) precisely so its released weights load here unchanged; see
[NOTICE](NOTICE). Everything around it — packaging, API, trainer, validator,
exporter, predictor, CLI — is this project's own. Nothing here derives from an
AGPL-licensed project.

Deformable attention is the `grid_sample` formulation, so the export path stays
plain ONNX (opset 16+) → OpenVINO; a test pins it to within 1e-4 of eager
PyTorch, and the assembled model matches the reference implementation to ~1e-5.


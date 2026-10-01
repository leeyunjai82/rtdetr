# Changelog

## 0.2.2

- **The learning rate follows the batch size.** `lr0=None`, now the default,
  means `1e-4 × √(batch / 4)`: unchanged at batch 4, 1.4e-4 at the default 8,
  2.8e-4 at 32. With one fixed rate a bigger batch took fewer steps and
  learned less in the same epochs (0.76 mAP50-95 at batch 4, under 0.5 at 32).
  **Changed:** a run at the default batch 8 now trains at 1.4e-4 instead of
  1e-4; `lr0=1e-4` keeps the old rate. `run.json` says which rate was used and
  why.

## 0.2.1

- **The browser app moved** to its own repository,
  [easydetect lab](https://github.com/themakerrobot/easydetect-lab). It runs on
  this package from PyPI; `platform/` is gone from here. A data folder from
  `platform/` keeps working: `python run.py --data <easydetect-platform folder>`.
- `easydetect.data.dataset.list_images` (was `_list_images`): the images of one
  split of a `data.yaml`, public because the lab uses it.

## 0.2.0

- **Two runtimes.** `pip install easydetect` now brings ONNX Runtime beside
  OpenVINO. `Detector(..., backend="onnxruntime")` runs the `.onnx` on any CPU
  (a Raspberry Pi included) with the same preprocessing and decoding, so the
  boxes match OpenVINO's; OpenVINO stays the default and the way to an Intel
  GPU or NPU. `$EASYDETECT_BACKEND` sets the default. A named model downloads
  the `.onnx` for ONNX Runtime (the mirror now carries one per size), a `.pt`
  exports whichever the runtime reads, and an `.xml` without OpenVINO runs
  from the `.onnx` beside it. `pip install "easydetect[train]"` adds training.
- **Validation on large sets.** `val` (and the per-epoch validation) no longer
  runs out of file descriptors on COCO-sized validation sets ("received 0
  items of ancdata").
- **Faster augmentation.** The zoom, crop and flip are rendered once at the
  training size: 3.1 ms a picture at 320, down from 25.4.
- **Tools.** `tools/coco2yolo.py` converts COCO 2017 for fine-tuning the COCO
  models at another input size; `tools/compare_yolo.py` scores a YOLO ONNX
  export and D-FINE checkpoints with one evaluator.

## 0.1.2

- `predict` and `track` drop a box that covers a higher-scoring one by more
  than `iou=0.7`, whatever the class — D-FINE's occasional second box on one
  object (a vehicle as both truck and car). `iou=None` keeps every box.
- **Changed:** `track(iou=)` is now that duplicate filter, as in Ultralytics;
  the tracker's matching threshold is `match_iou=0.3`.

## 0.1.1

- **Augmentation** from D-FINE's recipe — colour jitter, zoom-out, IoU crop,
  flip — with the last tenth of the epochs on plain pictures;
  `train(augment=False)` keeps only the flip.
- Platform: early stopping per run, the backbone-freeze choice in plain sight,
  a training run that can always be deleted, cut-off runs that keep their best
  epoch, and the themaker-ui design with a Korean / English switch.

## 0.1.0

- First release: D-FINE (n/s/m/l/x) with its Apache-2.0 COCO weights, training,
  OpenVINO export and inference, and the browser platform.

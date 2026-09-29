<div align="center">

# rtdetr

**Real-time object detection you can actually ship.**
PyTorch training, OpenVINO inference, and a browser app to run the whole loop —
Apache-2.0 from the code to the model you export.

[![PyPI](https://img.shields.io/pypi/v/rtdetr?color=2b7489)](https://pypi.org/project/rtdetr/)
[![Python](https://img.shields.io/pypi/pyversions/rtdetr)](https://pypi.org/project/rtdetr/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/leeyunjai82/rtdetr/blob/main/LICENSE)
[![CI](https://github.com/leeyunjai82/rtdetr/actions/workflows/ci.yml/badge.svg)](https://github.com/leeyunjai82/rtdetr/actions/workflows/ci.yml)

![detections on a street scene](https://raw.githubusercontent.com/leeyunjai82/rtdetr/main/docs/assets/demo.jpg)

</div>

## Why

* **One licence, all the way down.** Apache-2.0 code, Apache-2.0 COCO weights,
  Apache-2.0 exports. Nothing to clear with legal before you ship a product.
* **24 FPS on four CPU cores.** No GPU, no CUDA, no drivers — OpenVINO runs the
  IR on the machine you already have. A GPU makes training quick; inference does
  not need one.
* **Train and deploy in the same tool.** `train()` writes the checkpoint,
  `export()` writes the OpenVINO IR and its `labels.txt`, and inference reads
  that IR. No conversion script to maintain between them.
* **Transformer detection, no NMS.** RT-DETR predicts a fixed set of queries, so
  there is no non-maximum suppression to tune and latency does not rise with the
  number of objects in frame.

## Install

```bash
pip install rtdetr              # inference
pip install "rtdetr[train]"     # + training
```

## Detect

```python
from rtdetr import RTDETR

model = RTDETR("rtdetr-r18")          # COCO weights, downloaded on first use
r = model("bus.jpg", conf=0.5)[0]

r.boxes.xyxy, r.boxes.conf, r.boxes.cls   # plain numpy
r.names[int(r.boxes.cls[0])]              # "person"
r.plot(); r.save(); r.show()
```

Point it at an image, a folder, a glob, a video, an RTSP stream, a webcam index,
or a numpy array. Long sources stream, so memory stays flat:

```python
for r in model.predict(0, stream=True, show=True):   # webcam, q or Esc quits
    print(r.boxes.xyxyn)

model.track("clip.mp4")        # adds r.boxes.id
```

A webcam viewer with an FPS counter is in [`examples/webcam.py`](https://github.com/leeyunjai82/rtdetr/blob/main/examples/webcam.py)
— `python examples/webcam.py --track`.

## Train

```python
model = RTDETR("rtdetr-r18")
model.train(data="data.yaml", epochs=100, imgsz=640, batch=8, device=0)
model.val(data="data.yaml").box.map50
model.export(format="openvino", half=True)     # IR + labels.txt
```

```yaml
# data.yaml
path: /data/cans
train: images/train
val: images/val
names: {0: can, 1: bottle}
```

Labels are one `.txt` per image, `cls cx cy w h` normalised — the layout every
labelling tool already exports. A YOLO-format download (Roboflow's included)
trains as it comes; [training.md](https://github.com/leeyunjai82/rtdetr/blob/main/docs/training.md#datasets-from-elsewhere)
lists the variations it accepts. `freeze="backbone"` trains roughly twice as fast
on a small set, and `resume=True` picks a killed run back up.

## Label, train and watch it in a browser

`platform/` is a web app on top of this package: drop images in, label them (the
model drafts the boxes, you correct them), queue a training run, watch the curve,
then run the result over a folder, a video or your webcam — on one machine, with
nothing leaving it.

```bash
pip install -r platform/requirements.txt
python platform/run.py          # http://<this machine>:8080
```

![the platform](https://raw.githubusercontent.com/leeyunjai82/rtdetr/main/docs/assets/platform.jpg)

## Speed

Median of 20 frames, whole pipeline (preprocess, inference, decode), on
**4 CPU cores** with nothing else running:

| setup | latency | FPS |
| --- | --- | --- |
| `rtdetr-r18` @640 | 42 ms | **23.6** |
| `rtdetr-r34` @640 | 64 ms | 15.7 |
| `rtdetr-r50` @640 | 80 ms | 12.6 |
| `rtdetr-r18` @480 | 25 ms | 39.9 |
| `rtdetr-r18` @320 | 18 ms | 55.8 |

Input size dominates everything else, so shrinking it is the first thing to try:
[performance.md](https://github.com/leeyunjai82/rtdetr/blob/main/docs/performance.md) has the rest, including when to ask for
exact fp32 instead of the CPU's default bfloat16.

## Models

| name | params | COCO AP | notes |
| --- | --- | --- | --- |
| `rtdetr-r18` | 20M | 46.4 | fastest — the default |
| `rtdetr-r34` | 31M | 48.9 | |
| `rtdetr-r50` | 43M | 53.1 | most accurate |

## Compared with YOLO

Published COCO val2017 numbers at 640 px, as each project reports them — not
re-measured here:

| model | params | COCO mAP50-95 | NMS | license |
| --- | --- | --- | --- | --- |
| RT-DETR r18 | 20M | 46.5 (paper), 46.4 (these weights) | none | Apache-2.0 |
| RT-DETR r34 | 31M | 48.9 | none | Apache-2.0 |
| RT-DETR r50 | 43M | 53.1 | none | Apache-2.0 |
| YOLOv8 s / m / l | | 44.9 / 50.2 / 52.9 | needed | AGPL-3.0 |
| YOLO11 s / m / l | 9.4M / 20.1M / 25.3M | 47.0 / 51.5 / 53.4 | needed | AGPL-3.0 |

Read it plainly:

* **On COCO accuracy per parameter, recent YOLO is ahead.** RT-DETR beat
  YOLOv8 when it came out; YOLO11-s reaches r18's score with half the
  parameters, and YOLO11-l matches r50 with about 60% of them.
* **The license is the difference that usually decides.** Ultralytics YOLO is
  AGPL-3.0: a product that ships it, or serves it over a network, must publish
  its source or buy a commercial license. Everything here is Apache-2.0 — code
  and weights — so it goes into closed products as it is.
* **No NMS.** RT-DETR emits one box per object, so there is no IoU threshold to
  tune and latency does not climb with the number of objects in the frame.
  Published speed tables often leave NMS out, and the two projects measure
  on different setups, so their FPS figures do not compare directly.
* **The whole model runs on an Intel NPU** — 24 FPS for r18 on a Core Ultra,
  leaving the CPU free ([measured](https://github.com/leeyunjai82/rtdetr/blob/main/docs/performance.md#on-a-desktop-with-an-npu)).
* **Where YOLO fits better:** nano-size models for very small devices,
  segmentation and pose in the same tool, and a far larger ecosystem.

COCO is a guide, not your answer. Fine-tune both on your own data, then compare
mAP on the same validation images and latency on the same device, NMS included.

## Command line

```bash
rtdetr predict model=rtdetr-r18 source=bus.jpg conf=0.5
rtdetr train   model=rtdetr-r18 data=data.yaml epochs=100
rtdetr val     model=best.pt data=data.yaml
rtdetr export  model=best.pt format=openvino half=true
```

## Docs

* [Using the model](https://github.com/leeyunjai82/rtdetr/blob/main/docs/usage.md) — sources, results, tracking, saving
* [Training](https://github.com/leeyunjai82/rtdetr/blob/main/docs/training.md) — datasets, validation, export, CLI
* [Platform](https://github.com/leeyunjai82/rtdetr/blob/main/platform/README.md) — the browser app: labelling, jobs, artefacts
* [Weights](https://github.com/leeyunjai82/rtdetr/blob/main/docs/weights.md) — the mirror, building it, offline use
* [Performance](https://github.com/leeyunjai82/rtdetr/blob/main/docs/performance.md) — measured speeds and how to improve them
* [Design](https://github.com/leeyunjai82/rtdetr/blob/main/docs/design.md) — architecture and provenance
* [Development](https://github.com/leeyunjai82/rtdetr/blob/main/docs/contributing.md) — tests, releases

## What it does not do

Boxes only — no segmentation, pose or classification. One training process, one
machine; multi-GPU and distributed training are out of scope. Inference targets
OpenVINO on CPU and Intel GPU, so a CUDA deployment means exporting to ONNX and
taking it from there.

## 한국어

```python
from rtdetr import RTDETR

model = RTDETR("rtdetr-r18")               # COCO 사전학습 가중치 자동 다운로드
model("bus.jpg", conf=0.5)[0].save()       # 결과 이미지 저장
model.train(data="data.yaml", epochs=100)  # 내 데이터로 학습
model.export(format="openvino", half=True) # 배포용 IR + labels.txt
```

CPU 4코어에서 640px 기준 24 FPS. GPU 없이 산업용 미니 PC에서 바로 돌아갑니다.
라벨링부터 학습·추론까지 브라우저로 하려면 `python platform/run.py`.
가중치 캐시는 `~/.rtdetr/`, 사내 미러는 `RTDETR_ASSETS_URL` 환경변수로 지정합니다.

YOLO와 비교하면: COCO 정확도는 최신 YOLO(YOLO11)가 같은 크기에서 조금 앞서지만,
코드·가중치가 모두 Apache-2.0이라 AGPL 걱정 없이 상용 제품에 넣을 수 있고, NMS가
없어 물체가 많아도 지연이 늘지 않으며, Intel NPU에서 모델 전체가 돕니다. 자세한
수치는 위의 [Compared with YOLO](#compared-with-yolo)를 보세요.

## Credits

The architecture and the COCO weights come from
[lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) (Apache-2.0), published
as [arXiv:2304.08069](https://arxiv.org/abs/2304.08069). This package is an
independent implementation of that network with its own training loop, inference
stack and tooling — see [NOTICE](https://github.com/leeyunjai82/rtdetr/blob/main/NOTICE).

## License

Apache-2.0 — see [LICENSE](https://github.com/leeyunjai82/rtdetr/blob/main/LICENSE) and [NOTICE](https://github.com/leeyunjai82/rtdetr/blob/main/NOTICE).

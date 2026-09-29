---
license: apache-2.0
library_name: rtdetr
pipeline_tag: object-detection
tags:
  - object-detection
  - rt-detr
  - detr
  - openvino
  - npu
  - coco
---

# RT-DETR weights for `pip install rtdetr`

Real-time DETR detectors, ready to run: **OpenVINO IR** for inference on a CPU,
an Intel GPU or an NPU, and the **PyTorch checkpoint** to fine-tune from. These
are the files the [rtdetr](https://github.com/leeyunjai82/rtdetr) package
downloads on first use — you never have to fetch them by hand.

Apache-2.0 end to end: the code, the COCO weights, and the models fine-tuned here.

```bash
pip install rtdetr
```

```python
from rtdetr import RTDETR

model = RTDETR("rtdetr-r18")                # downloads rtdetr-r18/ from this repo once
results = model("photo.jpg", conf=0.25)
results[0].save("result.jpg")

for r in model.predict(0, stream=True, show=True):   # webcam; q or Esc quits
    pass
```

No NMS step to tune: RT-DETR predicts its set of boxes end to end, and the
package handles resize, decode and drawing.

## COCO models — 80 classes

| folder | backbone | decoder | params | COCO mAP50-95 |
| --- | --- | --- | --- | --- |
| [`rtdetr-r18`](./rtdetr-r18) | PResNet-18 (ResNet-18-vd) | 3 layers | 20M | 46.4 |
| [`rtdetr-r34`](./rtdetr-r34) | PResNet-34 (ResNet-34-vd) | 4 layers | 31M | 48.9 |
| [`rtdetr-r50`](./rtdetr-r50) | PResNet-50 (ResNet-50-vd) | 6 layers | 43M | 53.1 |

Every folder holds the same four files:

| file | what it is |
| --- | --- |
| `<name>.xml` + `<name>.bin` | OpenVINO IR — what `RTDETR("<name>")` runs |
| `<name>.pt` | PyTorch checkpoint — the starting point for `model.train(...)` |
| `labels.txt` | class names, one per line |

`rtdetr-r18` at 640 × 640, whole pipeline (preprocess, inference, decode):

| machine | device | latency |
| --- | --- | --- |
| 4 CPU cores with bfloat16 | `CPU` | 42 ms (23.6 FPS) |
| Core Ultra 5 250K Plus | `NPU` | 41 ms (24.3 FPS) — the CPU stays free |
| Core Ultra 5 250K Plus | `CPU` | 85 ms (11.8 FPS) |

More in [performance](https://github.com/leeyunjai82/rtdetr/blob/main/docs/performance.md).

## Fine-tuned models — [`models/`](./models)

Detectors trained on one job, each with its own README: classes, scores, the data
it learned from, and how it was trained.

| folder | finds | base | val mAP50-95 |
| --- | --- | --- | --- |
| [`models/rock-paper-scissors`](./models/rock-paper-scissors) | rock, paper, scissors | r18 | see its README |

<!-- add a row per model folder; its README.md already has the numbers -->

Use one with `huggingface_hub`:

```python
from huggingface_hub import snapshot_download
from rtdetr import RTDETR

root = snapshot_download("leeyunjai/rtdetr", allow_patterns="models/rock-paper-scissors/*")
model = RTDETR(f"{root}/models/rock-paper-scissors/best.xml", device="AUTO")   # CPU · GPU · NPU
model.predict("photo.jpg", save=True)
```

## Train your own

```bash
pip install "rtdetr[train]"
```

```python
from rtdetr import RTDETR

model = RTDETR("rtdetr-r18")                       # starts from the COCO weights above
model.train(data="data.yaml", epochs=50, imgsz=640, freeze="backbone")
model.export(format="openvino")                    # best.xml + best.bin + labels.txt
```

`data.yaml` is the common images/ + labels/ layout — a Roboflow export works as
it is. Or do it all in a browser: the
[platform](https://github.com/leeyunjai82/rtdetr/tree/main/platform) collects and
labels images, trains, shows the numbers, and writes the upload folder for
`models/` with its README filled in from the run.

## Layout

```
rtdetr-r18/   rtdetr-r18.xml  rtdetr-r18.bin  rtdetr-r18.pt  labels.txt
rtdetr-r34/   …
rtdetr-r50/   …
models/
  rock-paper-scissors/   best.xml  best.bin  labels.txt  README.md
```

Keep these names: the package builds its download URLs from them
(`<repo>/resolve/main/<name>/<name>.xml`). To host a copy elsewhere, mirror the
same layout and point `$RTDETR_ASSETS_URL` at it.

## License and credit

* The COCO weights are the official RT-DETR checkpoints by Wenyu Lv et al.,
  released under Apache-2.0 at [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR),
  converted unchanged. COCO annotations are CC BY 4.0.
* The package and the fine-tuned models are Apache-2.0. Each model's README says
  where its training images came from; those images keep their own license.

```bibtex
@inproceedings{zhao2024rtdetr,
  title     = {DETRs Beat YOLOs on Real-time Object Detection},
  author    = {Zhao, Yian and Lv, Wenyu and Xu, Shangliang and Wei, Jinman and
               Wang, Guanzhong and Dang, Qingqing and Liu, Yi and Chen, Jie},
  booktitle = {CVPR},
  year      = {2024}
}
```

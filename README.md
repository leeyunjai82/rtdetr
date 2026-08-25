# rtdetr

**Ultralytics-style API, 100% Apache-2.0.** RT-DETR object detection you can ship
inside a product: train with PyTorch, run with OpenVINO, and keep the workflow
your team already knows — `model.train(...)`, `model.predict(...)`,
`results[0].boxes.xyxy`.

Every line of the network, loss, trainer, validator and exporter is an original
implementation. No Ultralytics code, no AGPL weights, no license surprises.

```bash
pip install rtdetr              # inference (numpy, opencv, openvino, pyyaml)
pip install "rtdetr[train]"     # + training (torch, torchvision, scipy, onnx)
```

```python
from rtdetr import RTDETR

model = RTDETR("rtdetr-r18")          # weights from the mirror (see "Pretrained weights")
results = model("bus.jpg", conf=0.5)  # list[Results]

r = results[0]
r.boxes.xyxy, r.boxes.conf, r.boxes.cls    # plain numpy
r.names[int(r.boxes.cls[0])]               # "person"
r.plot(); r.save(); r.show()
```

```python
model = RTDETR("rtdetr-r18")
model.train(data="data.yaml", epochs=100, imgsz=640, batch=8, device=0)
metrics = model.val(data="data.yaml")       # metrics.box.map50 / .box.map
model.export(format="openvino", half=True)  # IR + labels.txt, ready to deploy
```

---

## Why this exists

Ultralytics YOLO is excellent and its API is muscle memory for a lot of people —
but the code and weights are AGPL-3.0, which rules them out of most shipped
products. `rtdetr` keeps the ergonomics and drops the license problem.

## Predicting

Any source a YOLO user would expect:

| Source | Example |
| --- | --- |
| image | `model("bus.jpg")` |
| glob | `model("frames/*.jpg")` |
| folder | `model("dataset/images")` |
| list file | `model("images.txt")` |
| URL | `model("https://example.com/bus.jpg")` |
| video | `model("clip.mp4")` |
| stream | `model("rtsp://camera/live")` |
| webcam | `model(0)` |
| array | `model(numpy_bgr)` / `model(pil_image)` |
| list | `model(["a.jpg", "b.jpg"])` |

```python
for r in model.predict("clip.mp4", conf=0.4, stream=True):   # generator, O(1) memory
    print(r.boxes.xyxyn)

model.predict("bus.jpg", save=True)     # writes runs/detect/predict/bus.jpg
model.track("clip.mp4")                 # IoU tracker -> r.boxes.id
```

The log line reads the way you expect:

```
image 1/1 bus.jpg: 640x640 4 persons, 1 bus, 12.3ms
```

### Results

| Attribute | What you get |
| --- | --- |
| `r.boxes.xyxy` | `(N, 4)` pixel corners |
| `r.boxes.xywh` | `(N, 4)` pixel centre + size |
| `r.boxes.xyxyn` / `r.boxes.xywhn` | the same, normalized 0..1 |
| `r.boxes.conf` / `r.boxes.cls` | `(N,)` scores and class indices |
| `r.boxes.id` | track ids after `model.track(...)`, else `None` |
| `r.names` | `{0: "person", ...}` |
| `r.plot()` | annotated BGR ndarray |
| `r.save()` / `r.show()` | write / display it |
| `r.summary()` | detections as JSON-ready dicts |
| `r.speed` | `{"preprocess": ms, "inference": ms, "postprocess": ms}` |

## Training

YOLO-format labels — the same `data.yaml` and `images/` + `labels/*.txt` layout:

```yaml
path: /data/cans
train: images/train
val: images/val
names:
  0: can
  1: bottle
```

```python
model = RTDETR("rtdetr-r18")
best = model.train(
    data="data.yaml", epochs=100, imgsz=640, batch=8,
    device=0, workers=4, project="runs", name="train",
    resume=False, patience=50, lr0=1e-4, seed=0,
)
# runs/train/weights/best.pt  (last.pt too — resume=True picks the run back up)
```

AdamW with a 10× lower LR on the backbone, linear warmup into cosine decay, AMP
on CUDA, mAP50-95 after every epoch, early stop on `patience`.

## Exporting and deploying

```python
model = RTDETR("runs/train/weights/best.pt")
xml = model.export(format="openvino", half=True, imgsz=640)
```

Writes `best.xml` + `best.bin`, plus **`labels.txt`** (one class name per line)
next to the IR — that is how downstream runtimes discover class names.

The exported graph emits **probabilities, not logits**. Anything that decodes it
must not apply a sigmoid a second time; this package checks the score range and
skips it (`rtdetr/predictor.py`).

## CLI

Same `key=value` shape as the `yolo` command:

```bash
rtdetr predict model=rtdetr-r18 source=bus.jpg conf=0.5
rtdetr train   model=rtdetr-r18 data=data.yaml epochs=100
rtdetr val     model=best.pt data=data.yaml
rtdetr export  model=best.pt format=openvino half=true
rtdetr track   model=best.pt source=clip.mp4
```

## Pretrained weights

> **Status: the mirror is not populated yet.** Until COCO weights are published
> there, `RTDETR("rtdetr-r18")` raises a `ModelNotFoundError` that names the ways
> forward — it never silently falls back to an untrained network. Everything else
> (your own `.pt`/`.xml`, training, val, export) works today.

`RTDETR("rtdetr-r18")` downloads the IR for that name on first use and caches it
in `~/.rtdetr/` (`$RTDETR_HOME` to move it, `$RTDETR_ASSETS_URL` to point at an
internal mirror — handy for air-gapped sites). Known names: `rtdetr-r18`,
`rtdetr-r34`, `rtdetr-r50`. Each mirror entry is `<name>/<name>.xml`, `.bin`,
`.pt` and `labels.txt`.

Publishing an entry takes three steps, and only the first is instant:

1. `tools/convert_official.py` — move what maps from the original release.
2. Fine-tune the result on COCO. The converted decoder cross-attention and CCFF
   blocks start fresh (see below), so this step is what actually earns the
   "COCO pretrained" label.
3. Upload `.pt`, `.xml`, `.bin` and `labels.txt` to the mirror path.

`tools/convert_official.py` converts the original
[lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) Apache-2.0 COCO weights
into this package's layout:

```bash
python tools/convert_official.py --weights rtdetr_r18vd_6x_coco.pth \
    --variant r18 --out rtdetr-r18.pt --report 10
```

It prints exactly what transferred. Note the honest part: this decoder uses plain
multi-head cross-attention rather than deformable attention (that is what keeps
ONNX/OpenVINO export dependency-free), and the CCFF fusion blocks are narrower,
so those tensors have no counterpart upstream. **A converted checkpoint is a warm
start, not a finished COCO model — fine-tune it before publishing.**

## Design notes

* **Backbone** ResNet-18/34/50, torchvision-compatible module names, so ImageNet
  weights load when torchvision is around.
* **Encoder** AIFI transformer on C5 + FPN/PAN cross-scale fusion.
* **Decoder** two-stage: dense heads pick the top-K encoder tokens as queries, six
  layers refine them. No deformable attention, on purpose — see above.
* **Loss** Hungarian matching, varifocal + L1 + GIoU, auxiliary and encoder heads.
* **Validation** COCO-style mAP50 / mAP50-95, 101-point interpolation, no
  pycocotools dependency.
* **Inference** OpenVINO, plain resize (no letterbox) to match training.

## Development

```bash
pip install -e ".[train,dev]"
pytest -q          # the parity checklist lives in tests/
ruff check .
```

## Releasing

Push a `v*` tag and GitHub Actions builds and uploads to PyPI:

```bash
git tag v0.1.0
git push origin v0.1.0
```

`.github/workflows/publish.yml` uses **PyPI trusted publishing** — no API token
lives in the repo. One-time setup on PyPI (Publishing → add a pending publisher):

| Field | Value |
| --- | --- |
| PyPI project | `rtdetr` |
| Owner | `leeyunjai82` |
| Repository | `rtdetr` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

The job refuses to publish when the tag and `pyproject.toml` version disagree, so
bump the version in the same commit you tag.

## 한국어 빠른 시작

Ultralytics YOLO와 사용법이 같지만 라이선스는 Apache-2.0입니다. 제품에 넣어도
문제없습니다.

```python
from rtdetr import RTDETR

model = RTDETR("rtdetr-r18")               # 미러에서 가중치 다운로드 (아래 "Pretrained weights" 참고)
results = model("bus.jpg", conf=0.5)       # list[Results]
results[0].boxes.xyxy                      # numpy 배열
results[0].save()                          # 결과 이미지 저장

model.train(data="data.yaml", epochs=100)  # YOLO 형식 라벨 그대로
model.val(data="data.yaml").box.map50      # mAP50
model.export(format="openvino", half=True) # IR + labels.txt
```

가중치 캐시는 `~/.rtdetr/`, 사내 미러는 `RTDETR_ASSETS_URL` 환경변수로 지정합니다.

## License

Apache-2.0. See [LICENSE](LICENSE).

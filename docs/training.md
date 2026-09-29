# Training, validating, exporting

Back to the [README](../README.md).

## Training

Labels are one `.txt` per image next to a `data.yaml`:

```yaml
path: /data/cans
train: images/train
val: images/val
names:
  0: can
  1: bottle
```

```python
model = Detector("dfine-s")
best = model.train(
    data="data.yaml", epochs=100, imgsz=640, batch=8,
    device=0, workers=4, project="runs", name="train",
    resume=False, patience=50, lr0=1e-4, seed=0,
)
# runs/train/weights/best.pt  (last.pt too — resume=True picks the run back up)
```

D-FINE's recipe, one process: AdamW with the backbone at its own slower rate
(half for n/s, a tenth for m, less for l/x), no weight decay on norms and biases,
linear warmup into cosine decay, AMP on CUDA, and an exponential moving average
of the weights that is what gets validated and saved. The loss is D-FINE's —
varifocal classification, L1 + GIoU boxes, fine-grained localisation and
decoupled distillation, over every decoder layer and the denoising queries.
mAP50-95 after every epoch, early stop on `patience`.

`device=0` uses the first CUDA GPU, `device="cpu"` forces CPU, and leaving it
out picks a GPU when there is one. CPU training is slow but real — at 320 px,
dfine-s manages an epoch of 240 images in a few minutes on 4 cores — and
fine-tuning from the COCO weights is what makes a short run enough.

No dataset yet? `python tools/make_toyset.py toyset` writes 300 labelled images of
squares and circles — enough to see a run learn, and to try every step below.

## Datasets from elsewhere

Any YOLO-format dataset trains as it is — the `.txt` layout above is what
labelling tools export. Downloaded datasets vary in the details, and these all
load without editing:

| in the download | what happens |
| --- | --- |
| `images/train` + `labels/train` | the standard layout |
| `train/images` + `train/labels`, yaml says `train: ../train/images` | a split per folder, as Roboflow exports it (pick a *YOLOv8* / *YOLOv11* TXT format) |
| `path:` naming someone else's machine (`/content/datasets/…`) | falls back to the folder the yaml is in |
| `train: [images/day, images/night]` | several folders make one split |
| polygon rows (`cls x1 y1 x2 y2 x3 y3 …`) | each becomes its bounding box |
| a sixth value on a box row | read as a confidence and ignored |

A label sits where the last `images` folder in the image's path becomes
`labels`, or beside the image when there is none — one rule, used by training,
the platform and its exports alike. A dataset with no `val:` is refused with a
message rather than validated on its training images.

```bash
unzip shapes.v1i.yolov11.zip -d shapes
easydetect train model=dfine-s data=shapes/data.yaml epochs=50 device=0
```

In the [platform](../platform/README.md), the same zip goes in through
*Datasets → zip* without unpacking.

## Starting point

`Detector("dfine-s")` warm-starts from the mirror's COCO weights. For a domain
COCO says nothing about (thermal, medical, satellite), or for a clean baseline,
start from the ImageNet-pretrained HGNetv2 backbone instead:

```python
Detector("dfine-s", pretrained=False).train(data="data.yaml", epochs=200)
```

## Freezing

```python
model.train(data="data.yaml", epochs=50, freeze="backbone")
```

Roughly twice as fast per epoch (measured on CPU: 1.90 s/step → 1.09 s/step at
640, batch 2) and it overfits less on a small dataset, because the features it
starts from are already good. `freeze` also takes `"encoder"`,
`"backbone+encoder"`, or a list of module prefixes. Frozen batch norms are held
in eval mode so their running statistics stop drifting.

## Watching a run

Every epoch appends a row to `runs/train/results.csv` and, if you pass one, calls
`on_epoch_end` with the same numbers — which is all a dashboard needs:

```python
model.train(data="data.yaml", epochs=100, on_epoch_end=lambda row: print(row))
# {'epoch': 1, 'loss': 24.7, 'vfl': .., 'l1': .., 'giou': .., 'map50_95': 0.31,
#  'lr': 0.0001, 'seconds': 12.4, 'epochs': 100, 'save_dir': 'runs/train'}
```

`summary.json` in the same folder holds the final numbers.

## Exporting and deploying

```python
model = Detector("runs/train/weights/best.pt")
xml = model.export(format="openvino", half=True, imgsz=640)
```

Writes `best.xml` + `best.bin`, plus **`labels.txt`** (one class name per line)
next to the IR — that is how downstream runtimes discover class names.

The exported graph emits **probabilities, not logits**. Anything that decodes it
must not apply a sigmoid a second time; this package checks the score range and
skips it (`easydetect/predictor.py`).

## CLI

`key=value` arguments:

```bash
easydetect predict model=dfine-s source=bus.jpg conf=0.5
easydetect train   model=dfine-s data=data.yaml epochs=100
easydetect val     model=best.pt data=data.yaml
easydetect export  model=best.pt format=openvino half=true
easydetect track   model=best.pt source=clip.mp4
```


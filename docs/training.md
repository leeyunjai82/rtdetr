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

`device=0` uses the first CUDA GPU, `device="cpu"` forces CPU, and leaving it
out picks a GPU when there is one. CPU training is slow but real: on 4 cores,
r18 at 640 runs about 1 image/s — 500 images × 50 epochs is a few hours, and
fine-tuning from the COCO weights is what makes that enough.

## Starting point

`RTDETR("rtdetr-r18")` warm-starts from the mirror's COCO weights. For a domain
COCO says nothing about (thermal, medical, satellite), or for a clean baseline,
start from the ImageNet backbone instead:

```python
RTDETR("rtdetr-r18", pretrained=False).train(data="data.yaml", epochs=200)
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
model = RTDETR("runs/train/weights/best.pt")
xml = model.export(format="openvino", half=True, imgsz=640)
```

Writes `best.xml` + `best.bin`, plus **`labels.txt`** (one class name per line)
next to the IR — that is how downstream runtimes discover class names.

The exported graph emits **probabilities, not logits**. Anything that decodes it
must not apply a sigmoid a second time; this package checks the score range and
skips it (`rtdetr/predictor.py`).

## CLI

`key=value` arguments:

```bash
rtdetr predict model=rtdetr-r18 source=bus.jpg conf=0.5
rtdetr train   model=rtdetr-r18 data=data.yaml epochs=100
rtdetr val     model=best.pt data=data.yaml
rtdetr export  model=best.pt format=openvino half=true
rtdetr track   model=best.pt source=clip.mp4
```


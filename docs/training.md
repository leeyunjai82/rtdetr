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


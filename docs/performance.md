# Variants and speed

Back to the [README](../README.md).

## Which variant

| variant | backbone | fusion width | decoder layers | params | COCO AP |
| --- | --- | --- | --- | --- | --- |
| `rtdetr-r18` | PResNet-18 (basic blocks) | 0.5x | 3 | 20M | 46.4 |
| `rtdetr-r34` | PResNet-34 (basic blocks) | 0.5x | 4 | 31M | 48.9 |
| `rtdetr-r50` | PResNet-50 (bottlenecks) | 1.0x | 6 | 43M | 53.1 |

## Measured speed

Median of 20 frames after 5 warmups, whole pipeline (preprocess, inference,
decode), on **4 CPU cores** with nothing else running:

| setup | default | `precision="f32"` |
| --- | --- | --- |
| r18 @640 | **42 ms (23.6 FPS)** | 129 ms (7.7 FPS) |
| r34 @640 | 64 ms (15.7 FPS) | 189 ms (5.3 FPS) |
| r50 @640 | 80 ms (12.6 FPS) | 281 ms (3.6 FPS) |
| r18 @640, FP16 IR | 41 ms (24.6 FPS) | 133 ms (7.5 FPS) |
| r18 @480 | 25 ms (39.9 FPS) | 76 ms (13.2 FPS) |
| r18 @320 | 18 ms (55.8 FPS) | 40 ms (24.7 FPS) |

Compute scales with input size: 61.0 GFLOPs at 640, 35.3 at 480, 16.9 at 320.
Time at 640 splits roughly backbone 55%, encoder 25%, decoder 25%.

## Precision: why two columns

OpenVINO picks the execution precision itself, and on a CPU that supports
bfloat16 that is what it uses — three times faster than fp32, at a small cost in
exactness. Over ten frames of a street clip, bf16 against f32:

* same number of detections in 7 of 10 frames,
* matched boxes at mean IoU 0.964 (worst 0.902),
* confidence differences averaging 0.009, at most 0.066.

That is invisible at a 0.5 threshold and can flip a borderline detection near it.
Ask for exactness when you need it:

```python
RTDETR("rtdetr-r18", precision="f32")   # matches PyTorch to ~1e-5
```

Without the hint, the exported IR differs from the torch model by up to 0.33 in
score on individual queries; with it, by 5e-06. Both produce the same
detections at sane thresholds — but if you are comparing against a reference or
debugging a threshold, use `f32`.

## Making it faster

1. **Shrink the input.** It dominates everything else: 640 -> 320 is 42 ms -> 18 ms.
   `model.export(format="openvino", imgsz=320)`, then load that IR. 320 still
   sees people and cars at conversational distance; 640 is for small or far
   objects.
2. **Use a GPU device.** OpenVINO talks to Intel integrated graphics too:
   `RTDETR("rtdetr-r18", device="GPU")`.
3. **Skip frames.** `predict(0, vid_stride=2, ...)` runs every other frame.
4. **FP16 is about size, not CPU speed.** `half=True` halves the weight file
   (78 MB -> 39 MB) and helps on GPU; on CPU it measured 41 ms against 42 ms.

## Duplicate boxes

There is no NMS: one-to-one matching during training teaches the model to emit a
single query per object. Over 32 frames of a street clip, same-class box pairs
overlapping by more than IoU 0.5:

| conf | boxes | overlapping pairs |
| --- | --- | --- |
| 0.50 | 239 | 0 |
| 0.25 | 477 | 34 |
| 0.10 | 1913 | 342 |

At working thresholds there is nothing to suppress. Below 0.25 the low-scoring
queries start piling onto the same object.

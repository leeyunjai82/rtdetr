# Sizes and speed

Back to the [README](../README.md).

## Which size

| size | backbone | decoder layers | params | COCO mAP50-95 |
| --- | --- | --- | --- | --- |
| `dfine-n` | HGNetv2-B0, two levels | 3 | 4M | 42.8 |
| `dfine-s` | HGNetv2-B0 | 3 | 10M | 48.5 |
| `dfine-m` | HGNetv2-B2 | 4 | 19M | 52.3 |
| `dfine-l` | HGNetv2-B4 | 6 | 31M | 54.0 |
| `dfine-x` | HGNetv2-B5 | 6 | 62M | 55.8 |

`dfine-s` is the default: the accuracy of a much larger model at a size a CPU
handles. `dfine-n` is for small boards; `m`/`l`/`x` when a GPU or an NPU does
the work and every point of mAP matters.

## Measure your machine

```bash
python tools/bench.py                             # dfine-s on CPU, GPU, NPU
python tools/bench.py --models dfine-n dfine-s dfine-m
python tools/bench.py --models runs/train/weights/best.pt --devices CPU NPU
```

Whole pipeline (resize, inference, decode), median of 30 calls, and every
device's boxes checked against the CPU's.

## Measured

`dfine-s` at 640 on a Core Ultra 5 250K Plus (18 cores, no bfloat16) with an RTX
5090, `python tools/bench.py` — whole pipeline, median of 30 calls, each device
checked against the CPU:

| `device=` | latency | FPS | vs CPU |
| --- | --- | --- | --- |
| `"CPU"` | 35.7 ms | 28.0 | reference |
| `"NPU"` | 37.8 ms | 26.5 | mean IoU 0.984 (worst 0.946) |
| `"GPU"` (the RTX 5090, see below) | 13.8 ms | 72.3 | identical |

The network alone is about 2 ms of that less (34 / 37 / 13 ms): resizing and
decoding cost little.

Against RT-DETR r18, which this package used to ship, timed the same way
(network only) on a 4-core cloud CPU without bfloat16: `dfine-s` 116 ms, r18
198 ms — faster, more accurate (48.5 against 46.4 COCO mAP) and half the size.

* **The NPU runs the whole model**, deformable attention and the box
  distribution included, while leaving the CPU free. Its fp16 arithmetic moves
  boxes by a few pixels at most. For a box that runs detection all day, this is
  the device to use.
* **The `"GPU"` row is the 5090 reached through NVIDIA's OpenCL driver**,
  because OpenVINO found no Intel GPU on that machine. It works and matches the
  CPU, but it is not a supported combination and leaves most of the card unused
  — expect lines of `105 warnings generated.` from the OpenCL compiler on first
  load; they are harmless. For inference on an NVIDIA card,
  export to ONNX (`format="onnx"`) and use onnxruntime-gpu or TensorRT.

## Precision

OpenVINO picks the execution precision itself: on a CPU with bfloat16 it uses
that — much faster than fp32, at a small cost in exactness that can flip a
borderline detection near the threshold. Ask for exactness when you need it:

```python
Detector("dfine-s", precision="f32")   # matches PyTorch to float noise
```

On a CPU without bfloat16 (like the one above) this changes nothing. When you
are comparing against a reference or debugging a threshold, use `f32`.

## Making it faster

1. **Shrink the input — after training at that size.** Compute scales with the
   pixel count: 320 is about a quarter of 640. But the COCO weights were trained
   at 640, and exporting them at 320 as they are loses most of their accuracy
   (see [Small inputs](#small-inputs-320) below). Train at the size you will
   run: `model.train(..., imgsz=320)`, and the IR it exports is 320 too.
2. **Pick the size.** `dfine-n` is roughly half of `dfine-s`'s work.
3. **Use another device.** `Detector("dfine-s", device="NPU")` on a Core
   Ultra, `device="GPU"` for Intel graphics. `ov.Core().available_devices`
   lists what OpenVINO can see; see [NPU on Linux](#npu-on-linux) if the NPU
   is missing.
4. **Skip frames.** `predict(0, vid_stride=2, ...)` runs every other frame.
5. **FP16 is about size, not CPU speed.** `half=True` halves the weight file and
   helps on GPU; on a CPU it makes little difference.

## Small inputs (320)

Measured on COCO val2017 (5,000 pictures, `easydetect val` and
`tools/compare_yolo.py`, the same evaluator for every row), all at 320:

| model | how | mAP50-95 | mAP50 | model time, Core Ultra 5 CPU |
| --- | --- | --- | --- | --- |
| dfine-n | COCO weights (trained at 640), run at 320 | 0.099 | 0.302 | |
| dfine-n | + fine-tuned 12 epochs on COCO at 320 | 0.324 | 0.482 | 5.5 ms |
| YOLO11s | its COCO weights, exported at 320 | 0.374 | 0.520 | 7.2 ms + NMS |

For reference, dfine-n at 640 scores 0.419 here. Two lessons:

* **Train at the size you deploy.** Fine-tuning at 320 took the score from
  0.099 to 0.313 in one epoch; the rest of the twelve added 0.011, so a few
  epochs are enough. `tools/coco2yolo.py` converts COCO for this.
* **At 320 on a CPU, a YOLO of similar speed is the more accurate choice.**
  dfine-n is the smaller model (4M parameters against 9.4M) and a quarter
  faster here, but 0.05 lower. D-FINE's lead is at 640 and on GPUs and NPUs.

`tools/compare_yolo.py` repeats the comparison for your own YOLO export and
checkpoints: `python tools/compare_yolo.py --data coco/data.yaml --yolo
yolo11s.onnx --dfine runs/dfine-n-320/weights/best.pt`.

## NPU on Linux

The kernel side ships with recent kernels: `lsmod | grep intel_vpu` and a
`/dev/accel/accel0` owned by the `render` group mean it is there. OpenVINO also
needs Intel's user-space driver and the Level Zero loader; without either, the
NPU is simply absent from `available_devices`, with no error.

```bash
# the ubuntu2404 tarball from https://github.com/intel/linux-npu-driver/releases
tar xzf linux-npu-driver-*-ubuntu2404.tar.gz
sudo apt install -y libtbb12 libze1      # libze1 is the Level Zero loader
sudo dpkg -i intel-fw-npu_*.deb intel-driver-compiler-npu_*.deb intel-level-zero-npu_*.deb
sudo usermod -aG render $USER            # then log in again
python -c "import openvino as ov; print(ov.Core().available_devices)"   # ... 'NPU'
```

`libze1` is the step most often missed: the NPU packages install cleanly
without it, and the device still does not appear. If Ubuntu's `libze1` is too
old for the driver, take `libze1_*.deb` from
[oneapi-src/level-zero releases](https://github.com/oneapi-src/level-zero/releases).

## Duplicate boxes

D-FINE has no NMS step: one-to-one matching during training teaches it to give
each object one query, and mostly it does. Below its confident answers,
though, a second query can land on the same object, often under a
neighbouring class. On eight test pictures, `dfine-s` at `conf=0.25` gave 194
boxes with 20 pairs overlapping by more than IoU 0.7 — one vehicle as `truck`
0.83 and `car` 0.57 (IoU 0.93), one fridge three times — and every pair was
one object twice. Raising `conf` to 0.5 cleared the same-class pairs but not
the truck/car ones.

So `predict` (and `track`) keep the best box and drop any box that covers it by
more than `iou=0.7`, whatever the class — the same name and default as
Ultralytics' NMS threshold, applied here to D-FINE's rare duplicates: 194 boxes became 176 with no pair left, and at
0.5, 66 became 60. Separate objects rarely overlap that much; for a scene where
they do, change or switch it off:

```python
model.predict("crowd.jpg", iou=0.85)   # only near-identical boxes merge
model.predict("crowd.jpg", iou=None)   # every box the model gave
```

IoU has a blind spot: a piece inside a whole. A chair half hidden behind a
person came back from `dfine-m` as the whole chair (0.64) plus its two visible
pieces (0.67, 0.61); each piece overlaps the whole by an IoU of only its share
of the area, so NMS at any usual threshold keeps all three (YOLO's NMS would
too — it measures overlap the same way). `contain=0.8` measures it over the
smaller box instead: a box of the same class with 80% of the smaller one's area
shared is one object. The inner box goes when the enclosing one is about as
sure (within 0.1), so the whole chair stays and its pieces go; the enclosing
one goes when it is much less sure than the box inside, so a loose box around
two people cannot erase them. It is off by default because a real object
inside another of its class — a child held by an adult — goes too:

```python
model.predict(0, stream=True, show=True, contain=0.8)   # one box per chair
```

`val` and the mAP a run reports use the unfiltered boxes, as D-FINE's own
numbers do.

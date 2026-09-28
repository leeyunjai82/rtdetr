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

### On a desktop with an NPU

`rtdetr-r18` at 640 on a Core Ultra 5 250K Plus (18 cores) with an RTX 5090,
same median-of-30 method, each device checked against the CPU's boxes:

| `device=` | latency | FPS | vs CPU |
| --- | --- | --- | --- |
| `"CPU"` | 84.9 ms | 11.8 | reference |
| `"NPU"` | 41.2 ms | **24.3** | mean IoU 0.986, conf within 0.013 |
| `"GPU"` (the RTX 5090, see below) | 23.3 ms | 42.8 | identical |

Two things that are easy to get wrong from the first table:

* **Core count is not the story.** This 18-core CPU is slower than the 4-core
  one above, because it has no bfloat16: OpenVINO runs it in fp32, where the
  4-core machine ran bf16. `precision="f32"` changes nothing here.
* **The NPU runs the whole model**, deformable attention included, at twice the
  CPU's rate while leaving the CPU free. Its fp16 arithmetic moves boxes by a
  few pixels at most. For a box that runs detection all day, this is the device
  to use.

The `"GPU"` row is the 5090 reached through NVIDIA's OpenCL driver, because
OpenVINO found no Intel GPU on that machine. It works and matches the CPU bit
for bit, but it is not a supported combination and it leaves most of the card
unused — expect OpenCL build warnings on first load. For inference on an NVIDIA
card, export to ONNX (`format="onnx"`) and use onnxruntime-gpu or TensorRT.

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
2. **Use another device.** `RTDETR("rtdetr-r18", device="NPU")` on a Core
   Ultra, `device="GPU"` for Intel graphics. `ov.Core().available_devices`
   lists what OpenVINO can see; see [NPU on Linux](#npu-on-linux) if the NPU
   is missing.
3. **Skip frames.** `predict(0, vid_stride=2, ...)` runs every other frame.
4. **FP16 is about size, not CPU speed.** `half=True` halves the weight file
   (78 MB -> 39 MB) and helps on GPU; on CPU it measured 41 ms against 42 ms.

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

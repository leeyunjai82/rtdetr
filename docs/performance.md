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
5090 — network only, median of 30 runs, each device checked against PyTorch:

| `device=` | latency | FPS | vs PyTorch |
| --- | --- | --- | --- |
| `"CPU"` | 34 ms | 29 | identical |
| `"NPU"` | 37 ms | 27 | mean IoU 0.985, conf within 0.04 |
| `"GPU"` (the RTX 5090, see below) | 13 ms | 74 | identical |

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
  — expect OpenCL build warnings on first load. For inference on an NVIDIA card,
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

1. **Shrink the input.** Compute scales with the pixel count: 320 is about a
   quarter of 640. `model.export(format="openvino", imgsz=320)`, then load that
   IR. 320 still sees people and cars at conversational distance; 640 is for
   small or far objects.
2. **Pick the size.** `dfine-n` is roughly half of `dfine-s`'s work.
3. **Use another device.** `Detector("dfine-s", device="NPU")` on a Core
   Ultra, `device="GPU"` for Intel graphics. `ov.Core().available_devices`
   lists what OpenVINO can see; see [NPU on Linux](#npu-on-linux) if the NPU
   is missing.
4. **Skip frames.** `predict(0, vid_stride=2, ...)` runs every other frame.
5. **FP16 is about size, not CPU speed.** `half=True` halves the weight file and
   helps on GPU; on a CPU it makes little difference.

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
single query per object, so at working thresholds (0.25 and up) there is
nothing to suppress. Very low thresholds let the low-scoring queries pile onto
the same object — raise `conf` rather than adding NMS.

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

D-FINE has no NMS step: one-to-one matching during training teaches it to give
each object one query, and mostly it does. Below its confident answers,
though, a second query can land on the same object, often under a
neighbouring class. On eight test pictures, `dfine-s` at `conf=0.25` gave 194
boxes with 20 pairs overlapping by more than IoU 0.7 — one vehicle as `truck`
0.83 and `car` 0.57 (IoU 0.93), one fridge three times — and every pair was
one object twice. Raising `conf` to 0.5 cleared the same-class pairs but not
the truck/car ones.

So `predict` keeps the best box and drops any box that covers it by more than
IoU 0.7, whatever the class: 194 boxes became 176 with no pair left, and at
0.5, 66 became 60. Separate objects rarely overlap that much; for a scene where
they do, change or switch it off:

```python
model.predict("crowd.jpg", overlap=0.85)   # only near-identical boxes merge
model.predict("crowd.jpg", overlap=None)   # every box the model gave
```

`val` and the mAP a run reports use the unfiltered boxes, as D-FINE's own
numbers do.

# Using the model

Everything `RTDETR(...)` can be pointed at, and everything a prediction gives
back. See also [training](training.md), [weights](weights.md),
[performance](performance.md).

## Predicting

Anything you can point it at:

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
model = RTDETR("rtdetr-r18", device="GPU")     # AUTO, CPU, GPU
model = RTDETR("rtdetr-r18", precision="f32")  # exact, ~3x slower on CPU

for r in model.predict("clip.mp4", conf=0.4, stream=True):   # generator, O(1) memory
    print(r.boxes.xyxyn)

model.predict("bus.jpg", save=True)     # writes runs/detect/predict/bus.jpg
model.track("clip.mp4")                 # IoU tracker -> r.boxes.id
for r in model.predict(0, stream=True, show=True):  # webcam window, q or Esc quits
    pass                                           # (a generator only runs when iterated)
```

A webcam loop you can copy, with an FPS counter and optional tracking, lives in
[`examples/webcam.py`](examples/webcam.py):

```bash
python examples/webcam.py --track          # camera 0
python examples/webcam.py --source rtsp://camera/live --conf 0.4 --save
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

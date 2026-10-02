# Keypoints: `task="pose"`

```python
from easydetect import Detector

model = Detector("dfine-s", task="pose")
r = model("people.jpg")[0]
r.keypoints.xy      # (N, 17, 2) pixel positions, one row per box
r.keypoints.conf    # (N, 17) 0-1; all zero for boxes that are not people
r.save("result.jpg")   # boxes with each person's skeleton drawn in
```

The 17 keypoints are COCO's, in COCO's order
(`easydetect.pose.KEYPOINT_NAMES`: nose, eyes, ears, shoulders, elbows,
wrists, hips, knees, ankles). `r.summary()` lists them by name for every
person. `predict`, `track` and the CLI (`task=pose`) all take it.

## How it works

Top-down, in two steps. The detector finds the people; each person box is
widened to a 3:4 shape with a quarter of margin (a raised arm often sticks
out of the box), cut out at 256×192, and a small network places the 17
keypoints in the crop, which are mapped back to the picture. A box gets
keypoints when its class is named `person` — or every box does, when the
model has no such class (a model trained on "worker" or "player").

The keypoint network is easydetect's own (`easydetect/nn/posenet.py`, 3.7 M
parameters): D-FINE's HGNetv2-B0 backbone, started from the COCO detector's
weights so it already knows people, and a head that reads each keypoint's
position as two classifications — over the crop's columns and over its rows,
at half-pixel steps (the SimCC formulation, Li et al., ECCV 2022) — with one
transformer layer across the 17 keypoints so an elbow is placed knowing where
the shoulder and wrist are. A keypoint's confidence is how sharply peaked
those two distributions are.

On a 4-core CPU (OpenVINO) it adds about 7 ms for one person and 20 ms for
four, on top of the detector.

It is trained on COCO 2017's person keypoints (labels CC BY 4.0) and nothing
else, so the weights carry no research-only licence.

## Training it

210 epochs over COCO's 150,000 labelled people. On one RTX 5090 that should
take roughly 4–5 hours — an estimate: reading, cropping and jittering a person
costs about 4 ms of one CPU core, so 16 workers feed some 4,000 a second and
the loading, not the GPU, sets the pace. Give it as many `--workers` as you
have cores to spare.

```bash
git clone https://github.com/themakerrobot/easydetect && cd easydetect
pip install "easydetect[train]"

# COCO 2017: pictures and annotations (skip what you already have)
mkdir -p ~/datasets/coco/images && cd ~/datasets/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q train2017.zip -d images && unzip -q val2017.zip -d images
unzip -q annotations_trainval2017.zip            # annotations/person_keypoints_*.json
cd -

python tools/train_pose.py --coco ~/datasets/coco --workers 16
```

`tools/coco2yolo.py` reads the same folder, so a COCO copy made for it works
as it is. The run writes to `runs/pose/s/`: `results.csv` (loss each epoch,
OKS AP every `--val-every` epochs), `run.json` (the settings), `last.pt`
(everything needed to go on: `--resume runs/pose/s`) and `best.pt` (the EMA
weights with the best AP). At the end `best.pt` is exported to
`pose-s.onnx` and checked against PyTorch on ONNX Runtime.

The AP printed during training is scored with COCO's own person boxes, which
measures the keypoint network alone. What a user gets depends on the detector
too; score the whole pipeline — the detector's boxes, missed people and false
boxes included — with

```bash
python tools/train_pose.py --coco ~/datasets/coco --eval runs/pose/s/pose-s.onnx --detector dfine-m
```

`--size m` trains a larger one on HGNetv2-B2 (from dfine-m's backbone).
`--limit 2000 --epochs 3` is a quick check that everything runs.

## Publishing it

`Detector(..., task="pose")` downloads `pose/pose-s.onnx` from the mirror.
After a run, upload it from the machine that trained it:

```bash
pip install -U huggingface_hub
hf auth login                       # a write token for the mirror's account
hf upload leeyunjai/easydetect runs/pose/s/pose-s.onnx pose/pose-s.onnx
```

Until it is there, `task="pose"` says so and where to put the file instead
(`~/.easydetect/pose/pose-s.onnx`).

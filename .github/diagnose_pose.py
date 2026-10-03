"""Audit of the pose training run (diag branch only, not for main).

    python .github/diagnose_pose.py <run folder> <coco folder>

1. results.csv as saved by the run.
2. OKS AP on val2017 (labelled boxes) of best.pt's EMA, last.pt's EMA, and
   last.pt's raw (non-EMA) weights — does the EMA lag behind the network?
3. How far the EMA sits from the network, weight by weight.
4. The frozen stem and stages: still the detector's weights and BN statistics?
5. Gradient norms and per-batch losses on augmented val crops (train mode):
   how often clipping at 10 bites, and how heavy the worst batches are.
6. Training annotations: how many crops magnify a tiny box enormously.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from train_pose import build, evaluate  # noqa: E402

from easydetect import downloads  # noqa: E402
from easydetect.data.keypoints import HALF_BODY_MIN, UPPER, KeypointDataset, load_coco  # noqa: E402
from easydetect.nn.posenet import simcc_loss  # noqa: E402
from easydetect.pose import INPUT, box_to_crop  # noqa: E402

run, coco = Path(sys.argv[1]), sys.argv[2]
torch.set_num_threads(4)
section = lambda title: print(f"\n=== {title} ===", flush=True)  # noqa: E731

section("1. results.csv")
print((run / "results.csv").read_text())
last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
print(f"last.pt: epoch index {last['epoch']} (epoch {last['epoch'] + 1}), best {last['best']}, "
      f"ema_updates {last['ema_updates']}; best.pt: epoch {best['epoch']} ap {best['ap']}")
size = last["size"]


def net_with(state):
    net = build(size, "none")
    net.load_state_dict(state)
    return net.eval()


section("2. OKS AP on val2017, labelled boxes")
val = KeypointDataset(coco, "val", augment=False)
for name, state in (("best.pt EMA", best["model"]), ("last.pt EMA", last["model"]),
                    ("last.pt raw net", last["net"])):
    started = time.time()
    m = evaluate(net_with(state), val, torch.device("cpu"), workers=3)
    print(f"{name:16s} ap {m['ap']:.4f} ap50 {m['ap50']:.4f} ap75 {m['ap75']:.4f} "
          f"({time.time() - started:.0f} s)", flush=True)

section("3. EMA vs network, relative distance per tensor group")
groups = {}
for k, v in last["net"].items():
    if not v.dtype.is_floating_point:
        continue
    e = last["model"][k].float()
    d = (v.float() - e).norm().item()
    n = v.float().norm().item() + 1e-12
    g = ".".join(k.split(".")[:2]) if k.startswith("backbone") else k.split(".")[0]
    groups.setdefault(g, []).append((d, n))
for g, rows in groups.items():
    d = sum(r[0] ** 2 for r in rows) ** 0.5
    n = sum(r[1] ** 2 for r in rows) ** 0.5
    print(f"{g:28s} |net-ema|/|net| = {d / n:.4f}   |net| = {n:.1f}")

section("4. Frozen layers (stem + first 2 stages) vs the detector checkpoint")
det = torch.load(downloads.download_checkpoint(f"dfine-{size}"), map_location="cpu",
                 weights_only=False)["model"]
worst, checked = 0.0, 0
for k, v in last["net"].items():
    if not (k.startswith("backbone.stem.") or k.startswith("backbone.stages.0.")
            or k.startswith("backbone.stages.1.")):
        continue
    ref = det.get(k)
    if ref is None or ref.shape != v.shape or not v.dtype.is_floating_point:
        continue
    checked += 1
    worst = max(worst, (v.float() - ref.float()).abs().max().item())
print(f"{checked} frozen tensors (weights and BN statistics) compared; "
      f"largest difference from the detector: {worst:.3g}")

section("5. Gradient norms on augmented val crops (train mode, last.pt raw net)")
net = build(size, "none")
net.load_state_dict(last["net"])
net.freeze(2)
net.train()
aug = KeypointDataset(coco, "val", augment=True)
loader = torch.utils.data.DataLoader(aug, batch_size=64, shuffle=True, num_workers=3,
                                     drop_last=True, generator=torch.Generator().manual_seed(0))
norms, losses = [], []
for b, (crops, xy, weight, _) in enumerate(loader):
    if b == 99:
        break
    net.zero_grad(set_to_none=True)
    x, y = net(crops)
    loss = simcc_loss(x, y, xy, weight)
    loss.backward()
    total = torch.norm(torch.stack([p.grad.norm() for p in net.parameters()
                                    if p.grad is not None])).item()
    norms.append(total)
    losses.append(loss.item())
norms, losses = np.array(norms), np.array(losses)
q = lambda a: " ".join(f"p{p}={np.percentile(a, p):.2f}" for p in (5, 50, 95, 100))  # noqa: E731
print(f"{len(norms)} batches of 64")
print(f"grad norm: {q(norms)}; clipped at 10 in {np.mean(norms > 10):.0%} of batches")
print(f"batch loss: {q(losses)}")
print("worst batches (loss, grad norm):",
      sorted(zip(losses.round(3), norms.round(2), strict=True), key=lambda t: -t[1])[:5])

section("6. Training annotations: crop magnification")
_, people = load_coco(Path(coco), "train")
mags, half_mags = [], []
for anns in people.values():
    for ann in anns:
        if ann.get("iscrowd") or ann.get("num_keypoints", 0) <= 0 or \
                ann["bbox"][2] <= 1 or ann["bbox"][3] <= 1:
            continue
        x, y, w, h = ann["bbox"]
        _, s = box_to_crop(np.array([x, y, x + w, y + h]))
        mags.append(INPUT[1] / s[0][0])
        k = np.asarray(ann["keypoints"], np.float64).reshape(-1, 3)
        lab = np.where(k[:, 2] > 0)[0]
        if len(lab) >= HALF_BODY_MIN:
            for part in ([i for i in lab if i in UPPER], [i for i in lab if i not in UPPER]):
                if len(part) >= 2:
                    pts = k[part, :2]
                    _, s = box_to_crop(np.concatenate([pts.min(0), pts.max(0)]))
                    half_mags.append(INPUT[1] / s[0][0])
for name, a in (("whole box", np.array(mags)), ("half body", np.array(half_mags))):
    print(f"{name}: {len(a)} crops, magnification {q(a)}; "
          f">20x {np.mean(a > 20):.1%}, >50x {np.mean(a > 50):.2%}, >100x {np.mean(a > 100):.2%}")
print(json.dumps({"done": True}))

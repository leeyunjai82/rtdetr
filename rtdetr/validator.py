# Apache-2.0
"""Validation: mAP@50 and mAP@50-95 (101-point interpolation), no pycocotools needed."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.dataset import DetDataset
from .utils.ops import cxcywh2xyxy


def _iou_np(a, b):
    """a: (N,4), b: (M,4) xyxy -> (N,M)."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None] - inter, 1e-9, None)


def _ap(recall, precision):
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[1.0], precision, [0.0]])
    p = np.flip(np.maximum.accumulate(np.flip(p)))
    x = np.linspace(0, 1, 101)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.x / 1.x
    return float(trapezoid(np.interp(x, r, p), x))


class Evaluator:
    """Accumulates (pred, gt) pairs per image; computes mAP over IoU 0.5:0.95."""

    def __init__(self, num_classes):
        self.nc = num_classes
        self.iouv = np.linspace(0.5, 0.95, 10)
        self.records = []  # per image: (pred xyxy, pred conf, pred cls, gt xyxy, gt cls)

    def add(self, pred_boxes, pred_conf, pred_cls, gt_boxes, gt_cls):
        self.records.append(
            (
                np.asarray(pred_boxes, np.float32).reshape(-1, 4),
                np.asarray(pred_conf, np.float32).reshape(-1),
                np.asarray(pred_cls, np.int64).reshape(-1),
                np.asarray(gt_boxes, np.float32).reshape(-1, 4),
                np.asarray(gt_cls, np.int64).reshape(-1),
            )
        )

    def compute(self):
        aps = np.zeros((self.nc, len(self.iouv)))
        seen = np.zeros(self.nc, bool)
        for c in range(self.nc):
            scores, matches, n_gt = [], [], 0
            for pb, pc, pk, gb, gk in self.records:
                gsel = gb[gk == c]
                n_gt += len(gsel)
                sel = pk == c
                pbc, pcc = pb[sel], pc[sel]
                order = np.argsort(-pcc)
                pbc, pcc = pbc[order], pcc[order]
                iou = _iou_np(pbc, gsel)
                m = np.zeros((len(pbc), len(self.iouv)), bool)
                for t, thr in enumerate(self.iouv):
                    used = np.zeros(len(gsel), bool)
                    for i in range(len(pbc)):
                        if not len(gsel):
                            break
                        j = int(np.argmax(np.where(used, -1.0, iou[i])))
                        if iou[i, j] >= thr and not used[j]:
                            used[j] = True
                            m[i, t] = True
                scores.append(pcc)
                matches.append(m)
            if n_gt == 0:
                continue
            seen[c] = True
            scores = np.concatenate(scores) if scores else np.zeros(0)
            matches = np.concatenate(matches) if matches else np.zeros((0, len(self.iouv)), bool)
            order = np.argsort(-scores)
            matches = matches[order]
            for t in range(len(self.iouv)):
                tp = np.cumsum(matches[:, t])
                fp = np.cumsum(~matches[:, t])
                recall = tp / n_gt
                precision = tp / np.clip(tp + fp, 1e-9, None)
                aps[c, t] = _ap(recall, precision)
        per_class = {c: float(aps[c].mean()) for c in range(self.nc) if seen[c]}
        if not seen.any():
            return {"map50": 0.0, "map": 0.0, "per_class": {}}
        return {
            "map50": float(aps[seen, 0].mean()),
            "map": float(aps[seen].mean()),
            "per_class": per_class,
        }


@torch.no_grad()
def validate_torch(net, data_yaml, imgsz=640, batch=8, conf=0.01, device=None, workers=2):
    """Evaluate a torch RTDETRNet on the val split. Returns metrics dict."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = DetDataset(data_yaml, "val", imgsz, augment=False)
    dl = DataLoader(ds, batch_size=batch, num_workers=workers, collate_fn=DetDataset.collate)
    net = net.to(device).eval()
    ev = Evaluator(net.num_classes)
    for imgs, targets in dl:
        out = net(imgs.to(device))
        scores = out["pred_logits"].sigmoid()
        boxes = cxcywh2xyxy(out["pred_boxes"])
        for b in range(imgs.shape[0]):
            sc, cl = scores[b].max(-1)
            keep = sc >= conf
            gt = cxcywh2xyxy(targets[b]["boxes"])
            ev.add(
                boxes[b][keep].cpu().numpy(),
                sc[keep].cpu().numpy(),
                cl[keep].cpu().numpy(),
                gt.cpu().numpy(),
                targets[b]["labels"].cpu().numpy(),
            )
    return ev.compute()

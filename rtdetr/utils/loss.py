# Apache-2.0
"""Hungarian matcher and detection criterion (varifocal + L1 + GIoU, incl. aux heads)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .ops import box_iou, cxcywh2xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, alpha=0.25, gamma=2.0):
        super().__init__()
        self.cc, self.cb, self.cg = cost_class, cost_bbox, cost_giou
        self.alpha, self.gamma = alpha, gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, nq = outputs["pred_logits"].shape[:2]
        prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # BQ, K
        boxes = outputs["pred_boxes"].flatten(0, 1)  # BQ, 4

        tgt_cls = torch.cat([t["labels"] for t in targets])
        tgt_box = torch.cat([t["boxes"] for t in targets])
        if tgt_cls.numel() == 0:
            return [
                (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
                for _ in range(bs)
            ]

        # focal-style classification cost
        p = prob[:, tgt_cls]
        pos = self.alpha * ((1 - p) ** self.gamma) * (-(p + 1e-8).log())
        neg = (1 - self.alpha) * (p**self.gamma) * (-(1 - p + 1e-8).log())
        cost_class = pos - neg

        cost_bbox = torch.cdist(boxes, tgt_box, p=1)
        cost_giou = -generalized_box_iou(cxcywh2xyxy(boxes), cxcywh2xyxy(tgt_box))

        C = self.cb * cost_bbox + self.cc * cost_class + self.cg * cost_giou
        C = C.view(bs, nq, -1).cpu()

        sizes = [len(t["labels"]) for t in targets]
        indices = []
        for i, c in enumerate(C.split(sizes, -1)):
            r, col = linear_sum_assignment(c[i])
            indices.append(
                (torch.as_tensor(r, dtype=torch.long), torch.as_tensor(col, dtype=torch.long))
            )
        return indices


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher=None, weight_vfl=1.0, weight_bbox=5.0, weight_giou=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher or HungarianMatcher()
        self.wv, self.wb, self.wg = weight_vfl, weight_bbox, weight_giou

    def _loss_single(self, out, targets, indices, num_boxes):
        logits, boxes = out["pred_logits"], out["pred_boxes"]
        bs, nq, k = logits.shape
        device = logits.device

        idx_b = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        idx_q = torch.cat([src for src, _ in indices])
        tgt_cls = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices, strict=False)])
        tgt_box = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices, strict=False)])

        # --- boxes ---
        src_box = boxes[idx_b, idx_q]
        if src_box.numel():
            l1 = F.l1_loss(src_box, tgt_box, reduction="sum") / num_boxes
            giou = generalized_box_iou(cxcywh2xyxy(src_box), cxcywh2xyxy(tgt_box)).diag()
            lg = (1 - giou).sum() / num_boxes
        else:
            l1 = boxes.sum() * 0.0
            lg = boxes.sum() * 0.0

        # --- varifocal classification ---
        target_score = torch.zeros(bs, nq, k, device=device, dtype=logits.dtype)
        if src_box.numel():
            with torch.no_grad():
                iou = box_iou(cxcywh2xyxy(src_box), cxcywh2xyxy(tgt_box))[0].diag().clamp(0)
            target_score[idx_b, idx_q, tgt_cls] = iou.to(logits.dtype)
        pred = logits.sigmoid().detach()
        weight = (0.75 * (pred**2.0) * (target_score <= 0).to(pred.dtype) + target_score).detach()
        vfl = F.binary_cross_entropy_with_logits(
            logits, target_score, weight=weight, reduction="sum"
        ) / max(num_boxes, 1)

        return self.wv * vfl + self.wb * l1 + self.wg * lg, {
            "vfl": vfl.detach(),
            "l1": l1.detach(),
            "giou": lg.detach(),
        }

    def forward(self, outputs, targets):
        num_boxes = max(sum(len(t["labels"]) for t in targets), 1)
        indices = self.matcher(outputs, targets)
        total, logs = self._loss_single(outputs, targets, indices, num_boxes)

        for aux in outputs.get("aux", []):
            aux_loss, _ = self._loss_single(aux, targets, self.matcher(aux, targets), num_boxes)
            total = total + aux_loss
        if "enc_aux" in outputs:
            enc = outputs["enc_aux"]
            enc_loss, _ = self._loss_single(enc, targets, self.matcher(enc, targets), num_boxes)
            total = total + enc_loss
        return total, logs

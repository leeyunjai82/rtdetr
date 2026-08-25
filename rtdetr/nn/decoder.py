# Apache-2.0
"""RT-DETR-style decoder.

Two-stage: dense per-token class/box heads on the encoder memory select the
top-K tokens as initial queries; 6 decoder layers refine boxes iteratively.
Cross-attention is standard multi-head attention over the flattened multi-scale
memory (deformable attention intentionally not used — keeps ONNX/OpenVINO
export trivial and dependency-free; see README for the accuracy note).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.ops import inverse_sigmoid


class MLP(nn.Module):
    def __init__(self, cin, hidden, cout, layers=3):
        super().__init__()
        dims = [cin] + [hidden] * (layers - 1) + [cout]
        self.layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:], strict=False)
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim=256, heads=8, ffn=1024):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Linear(ffn, dim))
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, tgt, memory, query_pos, mem_pos):
        q = k = tgt + query_pos
        tgt = self.norm1(tgt + self.self_attn(q, k, tgt, need_weights=False)[0])
        tgt = self.norm2(
            tgt + self.cross_attn(tgt + query_pos, memory + mem_pos, memory, need_weights=False)[0]
        )
        return self.norm3(tgt + self.ffn(tgt))


class RTDETRDecoder(nn.Module):
    def __init__(self, num_classes, dim=256, num_queries=300, layers=6, heads=8, ffn=1024):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.layers = nn.ModuleList(DecoderLayer(dim, heads, ffn) for _ in range(layers))

        # encoder-side (stage 1) heads
        self.enc_norm = nn.LayerNorm(dim)
        self.enc_score = nn.Linear(dim, num_classes)
        self.enc_bbox = MLP(dim, dim, 4)

        # decoder-side heads (shared class head, per-layer refinement is via same bbox MLP)
        self.dec_score = nn.ModuleList(nn.Linear(dim, num_classes) for _ in range(layers))
        self.dec_bbox = nn.ModuleList(MLP(dim, dim, 4) for _ in range(layers))

        self.query_pos_head = MLP(4, dim * 2, dim, layers=2)
        self.tgt_proj = nn.Linear(dim, dim)

        self._init_heads()

    def _init_heads(self):
        bias = -math.log((1 - 0.01) / 0.01)  # focal init
        nn.init.constant_(self.enc_score.bias, bias)
        for m in self.dec_score:
            nn.init.constant_(m.bias, bias)
        nn.init.constant_(self.enc_bbox.layers[-1].weight, 0.0)
        nn.init.constant_(self.enc_bbox.layers[-1].bias, 0.0)
        for m in self.dec_bbox:
            nn.init.constant_(m.layers[-1].weight, 0.0)
            nn.init.constant_(m.layers[-1].bias, 0.0)

    @staticmethod
    def build_anchors(shapes, device, dtype, grid_size=0.05, eps=1e-2):
        """Grid anchors (cxcywh, normalized) for each level + validity mask."""
        anchors = []
        for lvl, (h, w) in enumerate(shapes):
            gy, gx = torch.meshgrid(
                torch.arange(h, device=device, dtype=dtype),
                torch.arange(w, device=device, dtype=dtype),
                indexing="ij",
            )
            cxy = torch.stack([(gx + 0.5) / w, (gy + 0.5) / h], -1)
            wh = torch.full_like(cxy, grid_size * (2.0**lvl))
            anchors.append(torch.cat([cxy, wh], -1).reshape(-1, 4))
        anchors = torch.cat(anchors, 0)[None]  # 1, S, 4
        valid = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))  # logit space
        anchors = torch.where(valid, anchors, torch.full_like(anchors, float("inf")))
        return anchors, valid

    def forward(self, feats):
        """feats: list of 3 maps (B, C, H, W). Returns dict of outputs."""
        shapes = [(f.shape[2], f.shape[3]) for f in feats]
        memory = torch.cat([f.flatten(2).permute(0, 2, 1) for f in feats], 1)  # B,S,C

        anchors, valid = self.build_anchors(shapes, memory.device, memory.dtype)
        mem = self.enc_norm(memory * valid.to(memory.dtype))

        enc_logits = self.enc_score(mem)  # B,S,K
        enc_boxes = (self.enc_bbox(mem) + anchors).sigmoid()  # B,S,4

        # a small input can hold fewer tokens than we have queries — ask for what exists
        k = min(self.num_queries, enc_logits.shape[1])
        topk = torch.topk(enc_logits.max(-1).values, k, dim=1).indices  # B,Q
        idx = topk.unsqueeze(-1)
        ref = torch.gather(enc_boxes, 1, idx.expand(-1, -1, 4)).detach()
        tgt = self.tgt_proj(torch.gather(mem, 1, idx.expand(-1, -1, mem.shape[-1])).detach())
        topk_logits = torch.gather(enc_logits, 1, idx.expand(-1, -1, self.num_classes))

        mem_pos = torch.zeros_like(memory)  # positional info carried by anchors/query pos
        dec_logits_all, dec_boxes_all = [], []
        for i, layer in enumerate(self.layers):
            query_pos = self.query_pos_head(ref)
            tgt = layer(tgt, memory, query_pos, mem_pos)
            delta = self.dec_bbox[i](tgt)
            ref_new = (delta + inverse_sigmoid(ref)).sigmoid()
            dec_logits_all.append(self.dec_score[i](tgt))
            dec_boxes_all.append(ref_new)
            ref = ref_new.detach() if self.training else ref_new

        out = {
            "pred_logits": dec_logits_all[-1],
            "pred_boxes": dec_boxes_all[-1],
        }
        if self.training:
            out["aux"] = [
                {"pred_logits": lg, "pred_boxes": bx}
                for lg, bx in zip(dec_logits_all[:-1], dec_boxes_all[:-1], strict=False)
            ]
            out["enc_aux"] = {
                "pred_logits": topk_logits,
                "pred_boxes": torch.gather(enc_boxes, 1, idx.expand(-1, -1, 4)),
            }
        return out

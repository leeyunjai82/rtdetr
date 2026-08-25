# Apache-2.0
"""RT-DETR decoder: deformable cross-attention over the encoder's three levels.

Adapted from the RT-DETR reference implementation
(https://github.com/lyuwenyu/RT-DETR, Apache-2.0); module names match the
released checkpoints. The deformable attention is the pure-PyTorch
``grid_sample`` form, which exports to ONNX (opset 16+) and runs on OpenVINO —
verified to 3e-7 against eager PyTorch. See NOTICE.
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ..utils.ops import inverse_sigmoid
from .common import MLP, bias_init_with_prob, get_activation


def deformable_attention(value, spatial_shapes, sampling_locations, attention_weights):
    """value (B, S, H, C) + sampled locations -> (B, Q, H*C)."""
    bs, _, n_head, c = value.shape
    _, len_q, _, n_levels, n_points, _ = sampling_locations.shape

    value_list = value.split([h * w for h, w in spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampled = []
    for level, (h, w) in enumerate(spatial_shapes):
        level_value = value_list[level].flatten(2).permute(0, 2, 1).reshape(bs * n_head, c, h, w)
        grid = sampling_grids[:, :, :, level].permute(0, 2, 1, 3, 4).flatten(0, 1)
        sampled.append(
            F.grid_sample(
                level_value, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )
        )
    weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, len_q, n_levels * n_points
    )
    merged = torch.stack(sampled, dim=-2).flatten(-2) * weights
    out = merged.sum(-1).reshape(bs, n_head * c, len_q)
    return out.permute(0, 2, 1)


class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=3, num_points=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        total = num_heads * num_levels * num_points
        self.sampling_offsets = nn.Linear(embed_dim, total * 2)
        self.attention_weights = nn.Linear(embed_dim, total)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2 * math.pi / self.num_heads)
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid = grid / grid.abs().max(-1, keepdim=True).values
        grid = grid.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        grid *= torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        self.sampling_offsets.bias.data[...] = grid.flatten()
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(self, query, reference_points, value, value_spatial_shapes):
        bs, len_q = query.shape[:2]
        value = self.value_proj(value).reshape(bs, value.shape[1], self.num_heads, self.head_dim)

        offsets = self.sampling_offsets(query).reshape(
            bs, len_q, self.num_heads, self.num_levels, self.num_points, 2
        )
        weights = self.attention_weights(query).reshape(
            bs, len_q, self.num_heads, self.num_levels * self.num_points
        )
        weights = F.softmax(weights, dim=-1).reshape(
            bs, len_q, self.num_heads, self.num_levels, self.num_points
        )
        # reference points are cxcywh, so offsets scale with the box size
        locations = (
            reference_points[:, :, None, :, None, :2]
            + offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
        )
        return self.output_proj(
            deformable_attention(value, value_spatial_shapes, locations, weights)
        )


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        n_levels=3,
        n_points=4,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, reference_points, memory, memory_spatial_shapes, query_pos_embed=None):
        q = k = tgt if query_pos_embed is None else tgt + query_pos_embed
        attended, _ = self.self_attn(q, k, value=tgt, need_weights=False)
        tgt = self.norm1(tgt + self.dropout1(attended))

        crossed = self.cross_attn(
            tgt if query_pos_embed is None else tgt + query_pos_embed,
            reference_points,
            memory,
            memory_spatial_shapes,
        )
        tgt = self.norm2(tgt + self.dropout2(crossed))

        ffn = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(ffn)
        return self.norm3(tgt.clamp(min=-65504, max=65504))


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(decoder_layer) for _ in range(num_layers))
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        tgt,
        ref_points_unact,
        memory,
        memory_spatial_shapes,
        bbox_head,
        score_head,
        query_pos_head,
    ):
        output = tgt
        out_bboxes, out_logits = [], []
        ref_points_detach = F.sigmoid(ref_points_unact)
        ref_points = ref_points_detach

        for i, layer in enumerate(self.layers):  # noqa: B007 - i indexes the heads
            output = layer(
                output,
                ref_points_detach.unsqueeze(2),
                memory,
                memory_spatial_shapes,
                query_pos_head(ref_points_detach),
            )
            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                out_logits.append(score_head[i](output))
                out_bboxes.append(
                    inter_ref_bbox
                    if i == 0
                    else F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points))
                )
            elif i == self.eval_idx:
                out_logits.append(score_head[i](output))
                out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox

        return torch.stack(out_bboxes), torch.stack(out_logits)


class RTDETRTransformer(nn.Module):
    """Two-stage decoder: encoder tokens propose queries, layers refine them."""

    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=(256, 256, 256),
        feat_strides=(8, 16, 32),
        num_levels=3,
        num_decoder_points=4,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = list(feat_strides)
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.aux_loss = aux_loss

        self._build_input_proj(feat_channels)

        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_decoder_points,
        )
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        self.num_denoising = num_denoising
        if num_denoising > 0:
            # kept so released checkpoints load as-is; contrastive denoising
            # itself is not part of this trainer.
            self.denoising_class_embed = nn.Embedding(
                num_classes + 1, hidden_dim, padding_idx=num_classes
            )

        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)
        self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        self.dec_score_head = nn.ModuleList(
            nn.Linear(hidden_dim, num_classes) for _ in range(num_decoder_layers)
        )
        self.dec_bbox_head = nn.ModuleList(
            MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(num_decoder_layers)
        )
        self._reset_parameters()

    def _build_input_proj(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for c in feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("conv", nn.Conv2d(c, self.hidden_dim, 1, bias=False)),
                            ("norm", nn.BatchNorm2d(self.hidden_dim)),
                        ]
                    )
                )
            )
        channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv",
                                nn.Conv2d(channels, self.hidden_dim, 3, 2, padding=1, bias=False),
                            ),
                            ("norm", nn.BatchNorm2d(self.hidden_dim)),
                        ]
                    )
                )
            )
            channels = self.hidden_dim

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        for score, bbox in zip(self.dec_score_head, self.dec_bbox_head, strict=True):
            init.constant_(score.bias, bias)
            init.constant_(bbox.layers[-1].weight, 0)
            init.constant_(bbox.layers[-1].bias, 0)
        init.xavier_uniform_(self.enc_output[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

    def _get_encoder_input(self, feats):
        proj_feats = [proj(feat) for proj, feat in zip(self.input_proj, feats, strict=False)]
        for i in range(len(feats), self.num_levels):
            proj_feats.append(self.input_proj[i](feats[-1] if i == len(feats) else proj_feats[-1]))

        flatten, spatial_shapes = [], []
        for feat in proj_feats:
            h, w = feat.shape[2], feat.shape[3]
            flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([int(h), int(w)])
        return torch.concat(flatten, 1), spatial_shapes

    def _generate_anchors(self, spatial_shapes, dtype=torch.float32, device="cpu", grid_size=0.05):
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(end=h, dtype=dtype),
                torch.arange(end=w, dtype=dtype),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_wh = torch.tensor([w, h]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_wh
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        return torch.where(valid_mask, anchors, torch.inf), valid_mask

    def _get_decoder_input(self, memory, spatial_shapes):
        anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        memory = valid_mask.to(memory.dtype) * memory

        output_memory = self.enc_output(memory)
        enc_logits = self.enc_score_head(output_memory)
        enc_boxes_unact = self.enc_bbox_head(output_memory) + anchors

        # a small input can hold fewer tokens than we have queries
        k = min(self.num_queries, enc_logits.shape[1])
        _, topk_ind = torch.topk(enc_logits.max(-1).values, k, dim=1)
        index = topk_ind.unsqueeze(-1)

        ref_points_unact = enc_boxes_unact.gather(1, index.repeat(1, 1, enc_boxes_unact.shape[-1]))
        enc_topk_logits = enc_logits.gather(1, index.repeat(1, 1, enc_logits.shape[-1]))
        target = output_memory.gather(1, index.repeat(1, 1, output_memory.shape[-1])).detach()
        return target, ref_points_unact.detach(), F.sigmoid(ref_points_unact), enc_topk_logits

    def forward(self, feats, targets=None):
        memory, spatial_shapes = self._get_encoder_input(feats)
        target, ref_points_unact, enc_topk_bboxes, enc_topk_logits = self._get_decoder_input(
            memory, spatial_shapes
        )
        out_bboxes, out_logits = self.decoder(
            target,
            ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
        )

        out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}
        if self.training and self.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": lg, "pred_boxes": bx}
                for lg, bx in zip(out_logits[:-1], out_bboxes[:-1], strict=True)
            ]
            out["aux_outputs"].append(
                {"pred_logits": enc_topk_logits, "pred_boxes": enc_topk_bboxes}
            )
        return out

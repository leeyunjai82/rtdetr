# Apache-2.0
"""The full network: PResNet backbone + HybridEncoder + RTDETRTransformer.

Module names and per-variant hyper-parameters follow the RT-DETR reference
implementation (https://github.com/lyuwenyu/RT-DETR, Apache-2.0) exactly, so a
released COCO checkpoint loads with ``strict=True`` and no remapping. See NOTICE.
"""

from __future__ import annotations

import torch.nn as nn

from .decoder import RTDETRTransformer
from .hybrid_encoder import HybridEncoder
from .presnet import PResNet

#: Per-variant settings, taken from the upstream configs.
VARIANT_CFG = {
    "r18": {
        "depth": 18,
        "backbone_channels": [128, 256, 512],
        "expansion": 0.5,
        "depth_mult": 1.0,
        "num_decoder_layers": 3,
    },
    "r34": {
        "depth": 34,
        "backbone_channels": [128, 256, 512],
        "expansion": 0.5,
        "depth_mult": 1.0,
        "num_decoder_layers": 4,
    },
    "r50": {
        "depth": 50,
        "backbone_channels": [512, 1024, 2048],
        "expansion": 1.0,
        "depth_mult": 1.0,
        "num_decoder_layers": 6,
    },
}
VARIANTS = tuple(VARIANT_CFG)


class RTDETRNet(nn.Module):
    def __init__(
        self,
        variant="r18",
        num_classes=80,
        num_queries=300,
        hidden_dim=256,
        pretrained_backbone=True,
    ):
        super().__init__()
        if variant not in VARIANT_CFG:
            raise ValueError(f"variant must be one of {VARIANTS}")
        cfg = VARIANT_CFG[variant]
        self.variant = variant
        self.num_classes = num_classes

        self.backbone = PResNet(depth=cfg["depth"], pretrained=pretrained_backbone)
        self.encoder = HybridEncoder(
            in_channels=cfg["backbone_channels"],
            hidden_dim=hidden_dim,
            expansion=cfg["expansion"],
            depth_mult=cfg["depth_mult"],
        )
        self.decoder = RTDETRTransformer(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            feat_channels=[hidden_dim] * 3,
            num_decoder_layers=cfg["num_decoder_layers"],
        )

    def forward(self, x, targets=None):
        return self.decoder(self.encoder(self.backbone(x)), targets)


class DeployWrapper(nn.Module):
    """ONNX/OpenVINO export head: returns (boxes cxcywh 0..1, scores sigmoid)."""

    def __init__(self, net: RTDETRNet):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        return out["pred_boxes"], out["pred_logits"].sigmoid()

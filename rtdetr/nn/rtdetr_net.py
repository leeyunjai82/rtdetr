# Apache-2.0
"""Full RT-DETR network: ResNet backbone + HybridEncoder + RTDETRDecoder."""

import torch.nn as nn

from .backbone import ResNet
from .decoder import RTDETRDecoder
from .encoder import HybridEncoder

VARIANTS = ("r18", "r34", "r50")


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
        assert variant in VARIANTS, f"variant must be one of {VARIANTS}"
        self.variant = variant
        self.num_classes = num_classes
        self.backbone = ResNet(variant, pretrained=pretrained_backbone)
        self.encoder = HybridEncoder(self.backbone.out_channels, hidden_dim)
        self.decoder = RTDETRDecoder(num_classes, hidden_dim, num_queries)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.encoder(feats)
        return self.decoder(feats)


class DeployWrapper(nn.Module):
    """ONNX/OpenVINO export head: returns (boxes cxcywh 0..1, scores sigmoid)."""

    def __init__(self, net: RTDETRNet):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        return out["pred_boxes"], out["pred_logits"].sigmoid()

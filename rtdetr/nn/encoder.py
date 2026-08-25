# Apache-2.0
"""Hybrid encoder: AIFI (transformer encoder on C5) + CCFF-style cross-scale fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=1, s=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RepBlock(nn.Module):
    """Simplified fusion block (CSP-style) used in CCFF."""

    def __init__(self, cin, cout, n=3):
        super().__init__()
        c = cout // 2
        self.cv1 = ConvBNAct(cin, c)
        self.cv2 = ConvBNAct(cin, c)
        self.m = nn.Sequential(
            *[nn.Sequential(ConvBNAct(c, c, 3), ConvBNAct(c, c, 3)) for _ in range(n)]
        )
        self.cv3 = ConvBNAct(2 * c, cout)

    def forward(self, x):
        a = self.m(self.cv1(x))
        b = self.cv2(x)
        return self.cv3(torch.cat([a, b], 1))


class AIFI(nn.Module):
    """Attention-based intra-scale feature interaction on the C5 map."""

    def __init__(self, dim=256, heads=8, ffn=1024, layers=1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            ffn,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.dim = dim

    @staticmethod
    def pos_embed_2d(w, h, dim, device, temperature=10000.0):
        gx = torch.arange(w, device=device, dtype=torch.float32)
        gy = torch.arange(h, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(gy, gx, indexing="ij")
        pdim = dim // 4
        omega = torch.arange(pdim, device=device, dtype=torch.float32) / pdim
        omega = 1.0 / (temperature**omega)
        ox = gx.flatten()[..., None] @ omega[None]
        oy = gy.flatten()[..., None] @ omega[None]
        return torch.cat([ox.sin(), ox.cos(), oy.sin(), oy.cos()], 1)[None]

    def forward(self, x):
        b, c, h, w = x.shape
        pos = self.pos_embed_2d(w, h, self.dim, x.device).to(x.dtype)
        seq = x.flatten(2).permute(0, 2, 1)  # B, HW, C
        seq = self.encoder(seq + pos)
        return seq.permute(0, 2, 1).reshape(b, c, h, w)


class HybridEncoder(nn.Module):
    """Projects C3/C4/C5 to hidden_dim, runs AIFI on C5, fuses FPN + PAN.

    Returns three maps [P3, P4, P5], each (B, hidden_dim, H, W).
    """

    def __init__(self, in_channels, hidden_dim=256):
        super().__init__()
        self.input_proj = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(c, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim))
                for c in in_channels
            ]
        )
        self.aifi = AIFI(hidden_dim)
        # top-down (FPN)
        self.lateral1 = ConvBNAct(hidden_dim, hidden_dim)
        self.fpn1 = RepBlock(hidden_dim * 2, hidden_dim)
        self.lateral2 = ConvBNAct(hidden_dim, hidden_dim)
        self.fpn2 = RepBlock(hidden_dim * 2, hidden_dim)
        # bottom-up (PAN)
        self.down1 = ConvBNAct(hidden_dim, hidden_dim, 3, 2)
        self.pan1 = RepBlock(hidden_dim * 2, hidden_dim)
        self.down2 = ConvBNAct(hidden_dim, hidden_dim, 3, 2)
        self.pan2 = RepBlock(hidden_dim * 2, hidden_dim)

    def forward(self, feats):
        c3, c4, c5 = [proj(f) for proj, f in zip(self.input_proj, feats, strict=False)]
        c5 = self.aifi(c5)

        p5 = self.lateral1(c5)
        p4 = self.fpn1(torch.cat([F.interpolate(p5, scale_factor=2.0, mode="nearest"), c4], 1))
        p4l = self.lateral2(p4)
        p3 = self.fpn2(torch.cat([F.interpolate(p4l, scale_factor=2.0, mode="nearest"), c3], 1))

        n4 = self.pan1(torch.cat([self.down1(p3), p4l], 1))
        n5 = self.pan2(torch.cat([self.down2(n4), p5], 1))
        return [p3, n4, n5]

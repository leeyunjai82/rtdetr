# Apache-2.0
"""PResNet-vd backbone (18/34/50), returning C3, C4, C5.

Adapted from the RT-DETR reference implementation
(https://github.com/lyuwenyu/RT-DETR, Apache-2.0): the module names match the
released checkpoints exactly, which is what lets those weights load here. The
'vd' variant means a three-conv stem and an average-pooled shortcut. See NOTICE.
"""

from __future__ import annotations

from collections import OrderedDict

import torch.nn as nn

from .common import ConvNormLayer, get_activation

RESNET_CFG = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3]}

#: ImageNet-pretrained vd backbones published alongside RT-DETR (Apache-2.0).
IMAGENET_URLS = {
    18: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth",
    34: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth",
    50: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth",
    101: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth",
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="d"):
        super().__init__()
        self.shortcut = shortcut
        if not shortcut:
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                            ("conv", ConvNormLayer(ch_in, ch_out, 1, 1)),
                        ]
                    )
                )
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)
        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = get_activation(act)

    def forward(self, x):
        out = self.branch2b(self.branch2a(x))
        short = x if self.shortcut else self.short(x)
        return self.act(out + short)


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="d"):
        super().__init__()
        width = ch_out
        self.branch2a = ConvNormLayer(ch_in, width, 1, 1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)
        self.shortcut = shortcut
        if not shortcut:
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                            ("conv", ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1)),
                        ]
                    )
                )
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)
        self.act = get_activation(act)

    def forward(self, x):
        out = self.branch2c(self.branch2b(self.branch2a(x)))
        short = x if self.shortcut else self.short(x)
        return self.act(out + short)


class Blocks(nn.Module):
    def __init__(self, block, ch_in, ch_out, count, stage_num, act="relu", variant="d"):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(
                block(
                    ch_in,
                    ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1,
                    shortcut=i != 0,
                    variant=variant,
                    act=act,
                )
            )
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class PResNet(nn.Module):
    def __init__(
        self,
        depth=18,
        variant="d",
        num_stages=4,
        return_idx=(1, 2, 3),
        act="relu",
        pretrained=False,
    ):
        super().__init__()
        block_nums = RESNET_CFG[depth]
        ch_in = 64
        if variant in ("c", "d"):
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]
        self.conv1 = nn.Sequential(
            OrderedDict(
                (name, ConvNormLayer(cin, cout, k, s, act=act))
                for cin, cout, k, s, name in conv_def
            )
        )

        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock
        out_channels = [block.expansion * c for c in ch_out_list]
        out_strides = [4, 8, 16, 32]

        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            self.res_layers.append(
                Blocks(block, ch_in, ch_out_list[i], block_nums[i], i + 2, act=act, variant=variant)
            )
            ch_in = out_channels[i]

        self.return_idx = list(return_idx)
        self.out_channels = [out_channels[i] for i in self.return_idx]
        self.out_strides = [out_strides[i] for i in self.return_idx]
        if pretrained:
            self.load_imagenet(depth)

    def load_imagenet(self, depth):
        """Pull the vd ImageNet weights published with RT-DETR (cached in ~/.rtdetr)."""
        import torch

        from ..downloads import cache_dir, download

        try:
            url = IMAGENET_URLS[depth]
            path = download(url, cache_dir() / "imagenet" / url.rsplit("/", 1)[-1])
            state = torch.load(path, map_location="cpu", weights_only=False)
            missing, unexpected = self.load_state_dict(state, strict=False)
            print(
                f"[rtdetr] loaded ImageNet backbone (r{depth}); "
                f"missing {len(missing)}, unexpected {len(unexpected)}"
            )
        except Exception as exc:  # offline / mirror down -> train from scratch
            print(f"[rtdetr] ImageNet backbone unavailable ({exc}); training from scratch")

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.max_pool2d(x, 3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs

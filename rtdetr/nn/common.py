# Apache-2.0
"""Shared building blocks.

Adapted from the RT-DETR reference implementation
(https://github.com/lyuwenyu/RT-DETR, Apache-2.0) so that the released
Apache-2.0 COCO weights load into this package unchanged. See NOTICE.
"""

from __future__ import annotations

import math

import torch.nn as nn


def get_activation(act):
    if act is None:
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act
    act = act.lower()
    table = {
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
        "hardswish": nn.Hardswish,
    }
    if act not in table:
        raise ValueError(f"unsupported activation: {act}")
    module = table[act]()
    if hasattr(module, "inplace"):
        module.inplace = True
    return module


def bias_init_with_prob(prior_prob=0.01):
    """Focal-loss style bias so training starts at the given foreground prior."""
    return float(-math.log((1 - prior_prob) / prior_prob))


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:], strict=True)
        )
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

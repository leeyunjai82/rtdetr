# Apache-2.0
"""ResNet-18/34/50 backbone returning C3, C4, C5.

Module names match torchvision's ResNet so ImageNet weights can be loaded
(strict=False) when torchvision is installed. torchvision itself is optional.
"""

import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = downsample

    def forward(self, x):
        idn = x if self.downsample is None else self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + idn)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin, cout, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv3 = nn.Conv2d(cout, cout * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        idn = x if self.downsample is None else self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.relu(x + idn)


_CFG = {
    "r18": (BasicBlock, [2, 2, 2, 2]),
    "r34": (BasicBlock, [3, 4, 6, 3]),
    "r50": (Bottleneck, [3, 4, 6, 3]),
}
_TV_NAME = {"r18": "resnet18", "r34": "resnet34", "r50": "resnet50"}


class ResNet(nn.Module):
    def __init__(self, variant="r18", pretrained=True):
        super().__init__()
        block, layers = _CFG[variant]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.out_channels = [128 * block.expansion, 256 * block.expansion, 512 * block.expansion]
        if pretrained:
            self._load_imagenet(variant)

    def _make_layer(self, block, planes, n, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        blocks = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        blocks += [block(self.inplanes, planes) for _ in range(n - 1)]
        return nn.Sequential(*blocks)

    def _load_imagenet(self, variant):
        try:
            import torchvision

            tv = getattr(torchvision.models, _TV_NAME[variant])(weights="IMAGENET1K_V1")
            missing, unexpected = self.load_state_dict(tv.state_dict(), strict=False)
            print(
                f"[rtdetr] loaded ImageNet backbone ({variant}), "
                f"skipped keys: {len(unexpected)}"
            )
        except Exception as e:  # torchvision missing / offline -> train from scratch
            print(f"[rtdetr] backbone pretrained weights unavailable ({e}); training from scratch")

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange

_COMPLEX_LAYERS_PATH = Path(__file__).resolve().parent / "complex" / "complex_layers.py"
_COMPLEX_LAYERS_SPEC = importlib.util.spec_from_file_location(
    "_cvnn_complex_layers_for_resnet_runtime",
    _COMPLEX_LAYERS_PATH,
)
_COMPLEX_LAYERS = importlib.util.module_from_spec(_COMPLEX_LAYERS_SPEC)
exec(
    compile(_COMPLEX_LAYERS_PATH.read_text(encoding="utf-8"), str(_COMPLEX_LAYERS_PATH), "exec"),
    _COMPLEX_LAYERS.__dict__,
)

ComplexBatchNorm2d = _COMPLEX_LAYERS.ComplexBatchNorm2d
ComplexConv2d = _COMPLEX_LAYERS.ComplexConv2d
ComplexMaxPool2d = _COMPLEX_LAYERS.ComplexMaxPool2d
ComplexReLU = _COMPLEX_LAYERS.ComplexReLU
ComplexToReal = _COMPLEX_LAYERS.ComplexToReal


class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = ComplexConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size[0],
            stride=stride[0],
            padding=padding[0],
        )
        self.bn1 = ComplexBatchNorm2d(out_channels)
        self.conv2 = ComplexConv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size[1],
            stride=stride[1],
            padding=padding[1],
        )
        self.bn2 = ComplexBatchNorm2d(out_channels)
        self.relu = ComplexReLU()

    def forward(self, x):
        output = self.conv1(x)
        output = self.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return self.relu(x + output)


class ResNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.extra = nn.Sequential(
            ComplexConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size[0],
                stride=stride[0],
                padding=padding[0],
            ),
            ComplexBatchNorm2d(out_channels),
        )
        self.conv1 = ComplexConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size[1],
            stride=stride[1],
            padding=padding[1],
        )
        self.bn1 = ComplexBatchNorm2d(out_channels)
        self.conv2 = ComplexConv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size[2],
            stride=stride[2],
            padding=padding[2],
        )
        self.bn2 = ComplexBatchNorm2d(out_channels)
        self.relu = ComplexReLU()

    def forward(self, x):
        extra_x = self.extra(x)
        out = self.conv1(x)
        out = self.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(extra_x + out)


class ResNet18(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv1 = ComplexConv2d(3, 64, kernel_size=[111, 3], stride=[8, 1], padding=[0, 0])
        self.bn1 = ComplexBatchNorm2d(64)
        self.maxpool = ComplexMaxPool2d(kernel_size=3, stride=[2, 1], padding=1)
        self.layer1 = nn.Sequential(
            ResNetBasicBlock(64, 64, [3, 3], [1, 1], [1, 1]),
            ResNetBasicBlock(64, 64, [3, 3], [1, 1], [1, 1]),
        )
        self.layer2 = nn.Sequential(
            ResNetDownBlock(64, 128, [1, 3, 3], [[2, 1], [2, 1], 1], [0, 1, 1]),
            ResNetBasicBlock(128, 128, [3, 3], [1, 1], [1, 1]),
        )
        self.layer3 = nn.Sequential(
            ResNetDownBlock(128, 256, [1, 3, 3], [2, 2, 1], [0, 1, 1]),
            ResNetBasicBlock(256, 256, [3, 3], [1, 1], [1, 1]),
        )
        self.layer4 = nn.Sequential(
            ResNetDownBlock(256, 512, [1, 3, 3], [2, 2, 1], [0, 1, 1]),
            ResNetBasicBlock(512, 512, [3, 3], [1, 1], [1, 1]),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.c2r = ComplexToReal()
        self.fc = nn.Linear(512, num_classes)
        self.relu = ComplexReLU()

    def forward(self, x):
        x = rearrange(x, "b l (c d) -> b c l d", c=3)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = rearrange(x, "b d 1 1 -> b d")
        x = self.c2r(x)
        x = self.fc(x)
        return x

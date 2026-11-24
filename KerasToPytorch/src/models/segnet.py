"""Bài 6: Symmetric Encoder-Decoder SegNet for Hand Depth Image Segmentation (96x96).

Ported from 02_KerasExamples.pdf (slides 52-56) to PyTorch 2.x.
Replaces Keras UpSampling2D((2, 2)) with nn.Upsample(scale_factor=2, mode='nearest')
and outputs raw pixel-wise logits (B, C, 96, 96) for nn.CrossEntropyLoss().

Author: TV4 - Vo Huynh Minh Sang (MSSV: 3123411256)
"""

from __future__ import annotations

import torch
from torch import nn


def _conv_block(in_channels: int, out_channels: int, use_bn: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn),
    ]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn))
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class SegNet(nn.Module):
    """Symmetric 3-stage Encoder-Decoder SegNet for 96x96 grayscale depth segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 7,
        base_channels: int = 32,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        # Encoder: 96 -> 48 -> 24 -> 12
        self.enc1 = _conv_block(in_channels, c1, use_bn=use_bn)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = _conv_block(c1, c2, use_bn=use_bn)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = _conv_block(c2, c3, use_bn=use_bn)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck at 12x12
        self.bottleneck = _conv_block(c3, c3, use_bn=use_bn)

        # Decoder: 12 -> 24 -> 48 -> 96 (UpSampling2D -> nn.Upsample mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3 = _conv_block(c3, c2, use_bn=use_bn)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = _conv_block(c2, c1, use_bn=use_bn)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = _conv_block(c1, c1, use_bn=use_bn)

        # Pixel-wise 1x1 classifier returning raw logits (B, num_classes, H, W)
        self.classifier = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        x = self.bottleneck(x)
        x = self.dec3(self.up3(x))
        x = self.dec2(self.up2(x))
        x = self.dec1(self.up1(x))
        return self.classifier(x)

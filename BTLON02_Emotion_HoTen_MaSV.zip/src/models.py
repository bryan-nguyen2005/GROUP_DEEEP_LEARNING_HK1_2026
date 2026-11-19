"""CNN architectures shared by the FER2013 experiments."""

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torchvision.models import VGG16_Weights


def _activation(name):
    factories = {
        "relu": nn.ReLU,
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.1),
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "sigmoid": nn.Sigmoid,
    }
    key = name.lower().replace("-", "_")
    if key == "leakyrelu":
        key = "leaky_relu"
    if key not in factories:
        raise ValueError(f"Unknown activation: {name}")
    return factories[key]()


def _conv(in_channels, out_channels, use_bn):
    layers = [
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=not use_bn)
    ]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    return layers


class _FERModel(nn.Module):
    def forward(self, images):
        return self.classifier(self.features(images))

    def get_first_conv_layer(self) -> nn.Conv2d:
        return next(layer for layer in self.features.modules() if isinstance(layer, nn.Conv2d))

    def get_final_conv_layer(self) -> nn.Conv2d:
        return [layer for layer in self.features.modules() if isinstance(layer, nn.Conv2d)][-1]


class _DemoCNN(_FERModel):
    def __init__(self, blocks, hidden_size, num_classes, activation, use_bn, dropout):
        super().__init__()
        layers = []
        channels = 1
        for block in blocks:
            for out_channels in block:
                layers.extend(_conv(channels, out_channels, use_bn))
                layers.append(_activation(activation))
                channels = out_channels
            layers.extend([nn.MaxPool2d(2), nn.Dropout(dropout)])

        self.features = nn.Sequential(*layers)
        side = 48 // (2 ** len(blocks))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * side * side, hidden_size),
            _activation(activation),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes),
        )


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, use_bn, dropout):
        super().__init__()
        shortcut = [nn.Conv2d(in_channels, out_channels, 1, bias=not use_bn)]
        if use_bn:
            shortcut.append(nn.BatchNorm2d(out_channels))
        self.shortcut = nn.Sequential(*shortcut)
        self.conv1 = nn.Sequential(
            *_conv(in_channels, out_channels, use_bn), _activation(activation)
        )
        self.conv2 = nn.Sequential(*_conv(out_channels, out_channels, use_bn))
        self.output = nn.Sequential(
            _activation(activation), nn.MaxPool2d(2), nn.Dropout2d(dropout)
        )

    def forward(self, images):
        return self.output(self.conv2(self.conv1(images)) + self.shortcut(images))


class ImprovedFERNet(_FERModel):
    def __init__(self, num_classes, activation, use_bn, dropout):
        super().__init__()
        channels = (1, 64, 128, 256, 512)
        self.features = nn.Sequential(*[
            _ResidualBlock(source, target, activation, use_bn, dropout)
            for source, target in zip(channels, channels[1:])
        ])
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels[-1], num_classes)
        )

    def get_first_conv_layer(self) -> nn.Conv2d:
        return self.features[0].conv1[0]

    def get_final_conv_layer(self) -> nn.Conv2d:
        return self.features[-1].conv2[0]


class VGG16Transfer(_FERModel):
    def __init__(self, num_classes, activation, dropout, pretrained):
        super().__init__()
        layers = []
        channels = 3
        for out_channels, depth in ((64, 2), (128, 2), (256, 3), (512, 3), (512, 3)):
            for _ in range(depth):
                layers.extend(_conv(channels, out_channels, use_bn=False))
                layers.append(_activation(activation))
                channels = out_channels
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)

        if pretrained:
            weights = VGG16_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
            features = {
                key.removeprefix("features."): value
                for key, value in weights.items()
                if key.startswith("features.")
            }
            self.features.load_state_dict(features)

        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(512, num_classes)
        )

    def forward(self, images):
        return super().forward(images.expand(-1, 3, -1, -1))


_DEMO_ARCHITECTURES = {
    "SmallCNN": (((64,) * 2, (128,) * 2 + (256,) * 2), 1024),
    "MediumCNN": (((64,) * 2, (128,) * 3, (256,) * 3), 1024),
    "Medium1CNN": (((64,) * 3, (128,) * 4, (256,) * 4), 1024),
    "LargeCNN": (((64,) * 4, (128,) * 4, (256,) * 4, (512,) * 4), 2048),
}
MODEL_NAMES = (*_DEMO_ARCHITECTURES, "ImprovedFERNet", "VGG16Transfer")


def build_model(
    name: str, num_classes: int = 7, activation: str = "relu", use_bn: bool = True,
    dropout: float = 0.2, *, pretrained: bool = True,
) -> nn.Module:
    """Build a model producing logits from a batch of 1×48×48 images.

    pretrained affects VGG16 only. Set it to False when restoring a saved
    checkpoint or running an offline forward check.
    """
    names = {model.lower(): model for model in MODEL_NAMES}
    key = names.get(name.lower())
    if key in _DEMO_ARCHITECTURES:
        blocks, hidden_size = _DEMO_ARCHITECTURES[key]
        return _DemoCNN(blocks, hidden_size, num_classes, activation, use_bn, dropout)
    if key == "ImprovedFERNet":
        return ImprovedFERNet(num_classes, activation, use_bn, dropout)
    if key == "VGG16Transfer":
        # The original VGG16 backbone has no BatchNorm.
        return VGG16Transfer(num_classes, activation, dropout, pretrained)
    raise ValueError(f"Unknown model {name!r}. Choose from {', '.join(MODEL_NAMES)}.")


def main():
    parser = argparse.ArgumentParser(description="Export parameter counts for the default models.")
    parser.add_argument("--output", type=Path, default=Path("logs/tv1_model_summary.csv"))
    output = parser.parse_args().output

    rows = []
    for name in MODEL_NAMES:
        model = build_model(name, pretrained=False)
        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        has_batch_norm = any(isinstance(layer, nn.BatchNorm2d) for layer in model.modules())
        rows.append((name, parameter_count, int(has_batch_norm)))
        del model

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model_name", "trainable_parameters", "uses_batch_norm"))
        writer.writerows(rows)
    print(f"Model summary: {output}")


if __name__ == "__main__":
    main()

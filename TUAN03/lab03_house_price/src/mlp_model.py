"""Configurable PyTorch MLP for a single continuous regression target."""

from dataclasses import dataclass

import torch.nn as nn


@dataclass(frozen=True)
class MLPConfig:
    name: str
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.2
    batch_norm: bool = False
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64


# Three architectures, plus controlled changes to MLP_2's initial LR/dropout.
DEFAULT_CONFIGS = (
    MLPConfig("MLP_1_baseline", (128, 64), dropout=0.0),
    MLPConfig("MLP_2_deeper"),
    MLPConfig("MLP_3_batchnorm", (512, 256, 128, 64),
              batch_norm=True, learning_rate=3e-4),
    MLPConfig("MLP_2_lr_1e-2", learning_rate=1e-2),
    MLPConfig("MLP_2_lr_3e-4", learning_rate=3e-4),
    MLPConfig("MLP_2_dropout_0", dropout=0.0),
    MLPConfig("MLP_2_dropout_01", dropout=0.1),
    MLPConfig("MLP_2_dropout_03", dropout=0.3),
)


class HouseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128, 64),
                 dropout=0.2, batch_norm=False):
        super().__init__()
        if input_dim <= 0 or not hidden_dims or any(dim <= 0 for dim in hidden_dims):
            raise ValueError("Input and hidden dimensions must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1).")

        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        # Linear output: the standardized log target may be negative or positive.
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

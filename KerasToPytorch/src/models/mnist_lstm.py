"""CNN + LSTM dự đoán tổng của chuỗi ảnh MNIST (N, 7, 1, 28, 28) -> (N, 1)."""

import torch
from torch import nn


class TimeDistributed(nn.Module):
    """Áp dụng một module độc lập theo từng bước thời gian của chuỗi (N, T, ...)."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps = inputs.size(0), inputs.size(1)
        flat_inputs = inputs.contiguous().view(batch_size * time_steps, *inputs.shape[2:])
        outputs = self.module(flat_inputs)
        return outputs.contiguous().view(batch_size, time_steps, *outputs.shape[1:])


class MNISTSequenceLSTM(nn.Module):
    """Mô hình TimeDistributed(CNN) + LSTM(128) + Dense(1) tương ứng Slide 46-50 Keras."""

    def __init__(self, time_steps: int = 7, lstm_units: int = 128):
        super().__init__()
        self.time_steps = time_steps
        self.cnn_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
        )
        self.feature_extractor = TimeDistributed(self.cnn_block)
        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_units, batch_first=True)
        self.fc = nn.Linear(lstm_units, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(inputs)  # (N, T, 128)
        lstm_out, _ = self.lstm(features)          # (N, T, 128)
        last_step = lstm_out[:, -1, :]             # (N, 128)
        return self.fc(last_step)                  # (N, 1)

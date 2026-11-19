"""MLP nhận ảnh MNIST (N, 1, 28, 28) và trả logits (N, 10)."""

from torch import nn


class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, inputs):
        features = self.flatten(inputs)
        features = self.dropout(self.relu(self.fc1(features)))
        features = self.dropout(self.relu(self.fc2(features)))
        return self.fc3(features)

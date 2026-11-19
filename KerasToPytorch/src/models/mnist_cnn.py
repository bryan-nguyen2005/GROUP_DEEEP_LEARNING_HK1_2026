"""CNN #1 nhận ảnh MNIST (N, 1, 28, 28) và trả logits (N, 10)."""

from torch import channels_last, nn


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, inputs):
        features = inputs.contiguous(memory_format=channels_last)
        for conv in (self.conv1, self.conv2, self.conv3):
            features = self.dropout(self.pool(self.relu(conv(features))))
        features = self.flatten(features)
        features = self.dropout(self.relu(self.fc1(features)))
        features = self.dropout(self.relu(self.fc2(features)))
        return self.fc3(features)

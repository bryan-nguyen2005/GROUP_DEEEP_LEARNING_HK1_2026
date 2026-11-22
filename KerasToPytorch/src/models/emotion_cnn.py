"""CNN nhận diện 7 cảm xúc theo cấu trúc được mô tả trong slide 38."""
from torch import nn


class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 6 * 6, 1024), nn.ReLU(), nn.Linear(1024, 7),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

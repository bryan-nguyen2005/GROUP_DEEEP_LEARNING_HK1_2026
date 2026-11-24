"""Unit test tự động kiểm tra forward/backward pass và shape hợp đồng cho toàn bộ mô hình.

Tác giả: TV3 - Nguyen Hoang Long (MSSV: 3123411179)
"""

import unittest
import torch
from torch import nn

from src.models import (
    MNISTMLP,
    MNISTCNN,
    GenderCNN,
    EmotionCNN,
    MNISTSequenceLSTM,
    TimeDistributed,
    SegNet,
)


class ModelsContractTests(unittest.TestCase):
    def test_time_distributed_wrapper(self):
        # Kiểm tra wrapper TimeDistributed với layer Linear bất kỳ
        linear = nn.Linear(16, 32)
        td = TimeDistributed(linear)
        x = torch.randn(4, 5, 16)  # (Batch=4, TimeSteps=5, Feat=16)
        out = td(x)
        self.assertEqual(out.shape, (4, 5, 32))
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(linear.weight.grad)

    def test_mnist_mlp_contract(self):
        model = MNISTMLP()
        x = torch.randn(4, 1, 28, 28)
        logits = model(x)
        self.assertEqual(logits.shape, (4, 10))
        loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1, 2, 3]))
        loss.backward()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(params, 669_706)

    def test_mnist_cnn_contract(self):
        model = MNISTCNN()
        x = torch.randn(4, 1, 28, 28)
        logits = model(x)
        self.assertEqual(logits.shape, (4, 10))
        loss = nn.CrossEntropyLoss()(logits, torch.tensor([4, 5, 6, 7]))
        loss.backward()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(params, 1_817_610)

    def test_gender_cnn_contract(self):
        model = GenderCNN()
        x = torch.randn(2, 3, 64, 64)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 2))
        loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1]))
        loss.backward()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(params, 5_270_562)

    def test_emotion_cnn_contract(self):
        model = EmotionCNN()
        x = torch.randn(2, 1, 48, 48)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 7))
        loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 6]))
        loss.backward()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(params, 2_390_791)

    def test_mnist_lstm_contract(self):
        model = MNISTSequenceLSTM(time_steps=7, lstm_units=128)
        x = torch.randn(3, 7, 1, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (3, 1))
        target = torch.tensor([[15.0], [28.0], [42.0]])
        loss = nn.L1Loss()(out, target)
        loss.backward()
        self.assertIsNotNone(model.fc.weight.grad)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(params, 552_577)

    def test_segnet_contract(self):
        model = SegNet(in_channels=1, num_classes=7, base_channels=32, use_bn=True)
        x = torch.randn(2, 1, 96, 96)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 7, 96, 96))
        target = torch.randint(0, 7, (2, 96, 96), dtype=torch.long)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        self.assertIsNotNone(model.classifier.weight.grad)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertGreater(params, 100_000)


if __name__ == "__main__":
    unittest.main()

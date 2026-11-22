"""Unit tests for TV2 utilities, metrics, trainer, checkpointing, and logging."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.metrics import compute_classification_metrics
from src.models import build_model
from src.trainer import EXPERIMENT_RUN_COLUMNS, EarlyStopping, train_model
from src.utils import count_trainable_params, format_duration, seed_everything


class TrainerAndMetricsTests(unittest.TestCase):
    def test_metrics_computation(self):
        y_true = [0, 1, 2, 3, 4, 5, 6, 3]
        y_pred = [0, 1, 2, 3, 4, 5, 0, 3]
        result = compute_classification_metrics(y_true, y_pred)
        self.assertAlmostEqual(result["accuracy"], 7 / 8)
        self.assertEqual(result["confusion_matrix"].shape, (7, 7))
        self.assertEqual(len(result["per_class"]), 7)
        self.assertGreater(result["macro_f1"], 0.7)

    def test_early_stopping_behavior(self):
        stopper = EarlyStopping(patience=2, mode="min")
        self.assertTrue(stopper.step(1.0, epoch=1))
        self.assertFalse(stopper.should_stop)
        self.assertFalse(stopper.step(1.05, epoch=2))
        self.assertFalse(stopper.should_stop)
        self.assertFalse(stopper.step(1.02, epoch=3))
        self.assertTrue(stopper.should_stop)
        self.assertEqual(stopper.best_epoch, 1)

    def test_train_model_contract_and_logs(self):
        seed_everything(42)
        images = torch.randn(28, 1, 48, 48)
        labels = torch.tensor([i % 7 for i in range(28)], dtype=torch.long)
        train_loader = DataLoader(TensorDataset(images, labels), batch_size=7, shuffle=True)
        val_loader = DataLoader(TensorDataset(images[:14], labels[:14]), batch_size=7, shuffle=False)

        model = build_model("ImprovedFERNet", num_classes=7, activation="relu", use_bn=True)
        self.assertGreater(count_trainable_params(model), 1_000_000)
        self.assertEqual(format_duration(75.5), "1m 15.5s")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = train_model(
                model,
                train_loader,
                val_loader,
                config={
                    "exp_id": "EXP-TEST",
                    "model_name": "ImprovedFERNet",
                    "max_epochs": 2,
                    "patience": 2,
                    "device": "cpu",
                    "use_amp": False,
                    "logs_dir": str(root / "logs"),
                    "checkpoint_dir": str(root / "checkpoints"),
                    "verbose": False,
                },
            )

            expected_keys = {
                "exp_id",
                "best_epoch",
                "total_epochs",
                "best_val_loss",
                "best_val_acc",
                "best_val_macro_f1",
                "train_time_sec",
                "num_params",
                "checkpoint_path",
                "history_path",
            }
            self.assertTrue(expected_keys <= set(result.keys()))
            self.assertTrue(Path(result["checkpoint_path"]).is_file())
            self.assertTrue(Path(result["history_path"]).is_file())

            ckpt = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=False)
            self.assertIn("model_state_dict", ckpt)
            self.assertEqual(ckpt["model_name"], "ImprovedFERNet")

            runs_csv = root / "logs" / "experiment_runs.csv"
            self.assertTrue(runs_csv.is_file())
            with runs_csv.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames or []), EXPERIMENT_RUN_COLUMNS)
                rows = list(reader)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["exp_id"], "EXP-TEST")

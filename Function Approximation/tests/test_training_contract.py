import tempfile
import unittest
from pathlib import Path

import torch
import pandas as pd
import numpy as np

from src.trainer import EarlyStopping
from src.data_generator import setup_standard_datasets
from src.run_experiment_A import run_experiment_A


class CheckpointTests(unittest.TestCase):
    def test_saves_small_improvement_without_resetting_patience(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            model = torch.nn.Linear(1, 1)
            stopper = EarlyStopping(patience=2, min_delta=0.1, checkpoint_path=str(path))
            with torch.no_grad():
                model.weight.fill_(1)
            stopper(1.0, model, 1)
            with torch.no_grad():
                model.weight.fill_(2)
            stopper(0.95, model, 2)
            self.assertEqual(stopper.best_epoch, 2)
            self.assertEqual(stopper.counter, 1)
            self.assertEqual(torch.load(path, weights_only=True)["weight"].item(), 2)
            self.assertTrue(stopper(0.96, model, 3))

    def test_experiment_a_records_mae_for_each_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            setup_standard_datasets(data_dir=str(root / "data"))
            run_experiment_A(data_dir=str(root / "data"), logs_dir=str(root / "logs"),
                             figures_dir=str(root / "figures"), checkpoints_dir=str(root / "checkpoints"),
                             sample_sizes=[20], seeds=[42])
            raw = pd.read_csv(root / "logs/raw_runs_exp_A.csv").iloc[0]
            summary = pd.read_csv(root / "logs/exp_A_results.csv").iloc[0]
            for split in ("train", "val", "test"):
                self.assertTrue(np.isfinite(raw[f"{split}_mae"]))
                self.assertAlmostEqual(raw[f"{split}_mae"], summary[f"{split}_mae_mean"])


if __name__ == "__main__":
    unittest.main()

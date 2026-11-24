"""Unit tests for TV4 visualization, filter/Grad-CAM extraction, and evaluation helpers."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.dataset import CLASS_NAMES
from src.evaluate_best_model import _select_stratified_cases
from src.models import build_model
from src.visualize import (
    plot_confusion_matrix,
    plot_experiment_comparison,
    plot_learning_curves,
    plot_prediction_grid,
)
from src.visualize_filters import (
    GradCAM,
    visualize_final_conv_filters,
    visualize_first_conv_filters,
)


class TV4VisualizeAndEvalTests(unittest.TestCase):
    def test_shared_plot_functions_create_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            history = {
                "epoch": [1, 2, 3],
                "train_loss": [1.5, 1.1, 0.9],
                "train_acc": [0.4, 0.55, 0.65],
                "val_loss": [1.4, 1.05, 0.95],
                "val_acc": [0.45, 0.58, 0.66],
                "val_macro_f1": [0.40, 0.52, 0.61],
            }
            p1 = plot_learning_curves(history, out / "lc.png", dpi=100)
            p2 = plot_experiment_comparison(
                ["SmallCNN", "ImprovedFERNet"], [0.667, 0.686], [0.631, 0.663], out / "cmp.png", dpi=100
            )
            cm = np.eye(7, dtype=np.int64) * 10 + 1
            p3 = plot_confusion_matrix(cm, CLASS_NAMES, out / "cm.png", dpi=100)
            imgs = [np.zeros((48, 48), dtype=np.uint8) for _ in range(8)]
            p4 = plot_prediction_grid(
                imgs, [0, 1, 2, 3, 4, 5, 6, 0], [0, 1, 2, 3, 4, 5, 6, 1],
                [0.9] * 8, CLASS_NAMES, out / "grid.png", nrows=2, ncols=4, dpi=100
            )
            for path in (p1, p2, p3, p4):
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 1000)

    def test_filter_viz_and_gradcam_on_improved_fernet(self) -> None:
        model = build_model("ImprovedFERNet", pretrained=False).eval()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            p1 = visualize_first_conv_filters(model, out / "first.png", dpi=100)
            p2 = visualize_final_conv_filters(model, out / "final.png", top_k=16, dpi=100)
            self.assertTrue(p1.is_file() and p1.stat().st_size > 1000)
            self.assertTrue(p2.is_file() and p2.stat().st_size > 1000)

        gradcam = GradCAM(model, model.get_final_conv_layer())
        try:
            x = torch.randn(1, 1, 48, 48)
            cam, pred_idx, conf, fmap = gradcam.compute(x)
            self.assertEqual(cam.shape, (48, 48))
            self.assertTrue(0 <= pred_idx < 7)
            self.assertTrue(0.0 <= conf <= 1.0)
            self.assertEqual(fmap.shape[0], 512)
        finally:
            gradcam.remove()

    def test_stratified_case_selection(self) -> None:
        y_true = np.repeat(np.arange(7), 10)
        y_pred = y_true.copy()
        y_pred[::3] = (y_pred[::3] + 1) % 7
        confs = np.linspace(0.5, 0.99, len(y_true))
        true_ids = _select_stratified_cases(y_true, y_pred, confs, correct=True, limit=14)
        wrong_ids = _select_stratified_cases(y_true, y_pred, confs, correct=False, limit=14)
        self.assertEqual(len(true_ids), 14)
        self.assertEqual(len(wrong_ids), 14)
        self.assertTrue(all(y_true[i] == y_pred[i] for i in true_ids))
        self.assertTrue(all(y_true[i] != y_pred[i] for i in wrong_ids))


if __name__ == "__main__":
    unittest.main()

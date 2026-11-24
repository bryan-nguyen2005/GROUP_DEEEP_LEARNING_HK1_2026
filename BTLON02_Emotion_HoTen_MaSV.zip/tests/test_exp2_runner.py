"""Unit tests for TV3 (EXP-07..EXP-17) training techniques and ablation runner."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.models import build_model
from src.run_exp2_training_techniques import (
    EXP2_SPECS,
    EXP2_SUMMARY_COLUMNS,
    export_exp2_summary_csv,
    generate_simulated_exp2_history,
    plot_exp2_figures,
)
from src.utils import count_trainable_params


class Exp2RunnerTests(unittest.TestCase):
    def test_exp2_specs_contract(self):
        self.assertEqual(len(EXP2_SPECS), 11)
        expected_ids = [f"EXP-{i:02d}" for i in range(7, 18)]
        actual_ids = [spec["exp_id"] for spec in EXP2_SPECS]
        self.assertEqual(actual_ids, expected_ids)

        subgroups = {spec["subgroup"] for spec in EXP2_SPECS}
        self.assertEqual(
            subgroups,
            {"2A_Activation", "2B_Optimizer", "2C_Augmentation", "2D_Regularization"},
        )

        for spec in EXP2_SPECS:
            self.assertEqual(spec["model_name"], "ImprovedFERNet")
            self.assertIn("notes", spec)
            self.assertIn("activation", spec)
            self.assertIn("optimizer", spec)

    def test_model_instantiation_for_specs(self):
        # Verify that all activations and use_bn variants instantiate cleanly
        for spec in EXP2_SPECS:
            model = build_model(
                name=spec["model_name"],
                num_classes=7,
                activation=spec["activation"],
                use_bn=spec["use_bn"],
                dropout=spec["dropout"],
                pretrained=False,
            )
            params = count_trainable_params(model)
            if spec["use_bn"]:
                self.assertEqual(params, 4_863_751)
            else:
                self.assertEqual(params, 4_860_871)
            del model

    def test_summary_and_plotting_pipeline(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            runs_csv = tmp_path / "experiment_runs.csv"
            summary_csv = tmp_path / "exp2_ablation_summary.csv"
            figures_dir = tmp_path / "figures"
            logs_dir = tmp_path / "logs"
            logs_dir.mkdir(parents=True)
            figures_dir.mkdir(parents=True)

            # Create mock history for EXP-05
            mock_base_history = logs_dir / "history_EXP-05.csv"
            with mock_base_history.open("w", encoding="utf-8") as f:
                f.write(
                    "epoch,train_loss,train_acc,val_loss,val_acc,val_macro_f1,lr,epoch_time_sec\n"
                )
                for ep in range(1, 10):
                    f.write(
                        f"{ep},1.0,0.5,0.9,0.65,0.60,0.001,11.0\n"
                    )

            # Generate history for all specs
            for spec in EXP2_SPECS:
                eid = spec["exp_id"]
                hpath = logs_dir / f"history_{eid}.csv"
                generate_simulated_exp2_history(spec, mock_base_history, hpath)
                self.assertTrue(hpath.is_file())

            # Write mock runs_csv
            with runs_csv.open("w", encoding="utf-8") as f:
                f.write(",".join(EXP2_SUMMARY_COLUMNS) + "\n")
                for spec in EXP2_SPECS:
                    eid = spec["exp_id"]
                    f.write(
                        f"{eid},{spec['subgroup']},ImprovedFERNet,{spec['activation']},{spec['optimizer']},{spec['scheduler']},{spec['aug_mode']},{int(spec['use_bn'])},{spec['loss_type']},5,9,0.5,0.7,0.88,0.69,0.67,500.0,checkpoints/{eid}.pth,logs/history_{eid}.csv,Test note\n"
                    )

            summary_rows = export_exp2_summary_csv(runs_csv, summary_csv)
            self.assertEqual(len(summary_rows), 11)
            self.assertTrue(summary_csv.is_file())

            plot_exp2_figures(summary_rows, logs_dir, figures_dir)
            self.assertTrue((figures_dir / "exp2_activations.png").is_file())
            self.assertTrue((figures_dir / "exp2_optimizers.png").is_file())
            self.assertTrue((figures_dir / "exp2_augmentation.png").is_file())
            self.assertTrue((figures_dir / "exp2_regularization_imbalance.png").is_file())


if __name__ == "__main__":
    unittest.main()

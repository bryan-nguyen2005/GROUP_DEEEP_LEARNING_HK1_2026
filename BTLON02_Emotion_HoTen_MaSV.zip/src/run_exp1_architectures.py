"""Run Group 1 (EXP-01..EXP-06) CNN architecture benchmark on FER2013."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dataset import get_dataloaders
from .metrics import save_per_class_metrics_csv
from .models import build_model
from .trainer import train_model
from .utils import seed_everything


EXP1_SPECS: list[dict[str, Any]] = [
    {
        "exp_id": "EXP-01",
        "model_name": "SmallCNN",
        "use_bn": True,
        "dropout": 0.2,
        "lr": 1e-3,
        "pretrained": False,
        "notes": "Demo port: 2 conv blocks (64 -> 128/256) + FC 1024 + BN",
    },
    {
        "exp_id": "EXP-02",
        "model_name": "MediumCNN",
        "use_bn": True,
        "dropout": 0.2,
        "lr": 1e-3,
        "pretrained": False,
        "notes": "Demo port: 3 conv blocks (2x64, 3x128, 3x256) + FC 1024 + BN",
    },
    {
        "exp_id": "EXP-03",
        "model_name": "Medium1CNN",
        "use_bn": True,
        "dropout": 0.2,
        "lr": 1e-3,
        "pretrained": False,
        "notes": "Demo port: deeper 3 conv blocks (3x64, 4x128, 4x256) + FC 1024 + BN",
    },
    {
        "exp_id": "EXP-04",
        "model_name": "LargeCNN",
        "use_bn": True,
        "dropout": 0.2,
        "lr": 1e-3,
        "pretrained": False,
        "notes": "Demo port: 4 conv blocks (4x64..4x512) + FC 2048 + BN",
    },
    {
        "exp_id": "EXP-05",
        "model_name": "ImprovedFERNet",
        "use_bn": True,
        "dropout": 0.2,
        "lr": 1e-3,
        "pretrained": False,
        "notes": "Group proposal: 4 Residual blocks + BN + Dropout2d + GlobalAvgPool",
    },
    {
        "exp_id": "EXP-06",
        "model_name": "VGG16Transfer",
        "use_bn": False,
        "dropout": 0.2,
        "lr": 1e-4,
        "pretrained": True,
        "notes": "Fail Case: fine-tune ImageNet VGG16 (5 MaxPool -> 1x1 on 48x48, lr=1e-4)",
    },
]

SUMMARY_COLUMNS = (
    "exp_id",
    "model_name",
    "num_params",
    "use_bn",
    "lr",
    "best_epoch",
    "total_epochs",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_macro_f1",
    "train_time_sec",
    "checkpoint_path",
    "history_path",
    "notes",
)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def export_summary_csv(runs_csv: Path, summary_csv: Path) -> list[dict[str, str]]:
    """Filter Group 1 experiments (EXP-01..EXP-06) into logs/exp1_architectures_summary.csv."""
    target_ids = {spec["exp_id"] for spec in EXP1_SPECS}
    all_rows = _load_csv_rows(runs_csv)
    exp1_rows = [row for row in all_rows if row.get("exp_id") in target_ids]
    exp1_rows.sort(key=lambda r: r["exp_id"])

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in exp1_rows:
            writer.writerow({col: row.get(col, "") for col in SUMMARY_COLUMNS})
    return exp1_rows


def plot_exp1_figures(
    summary_rows: list[dict[str, str]],
    logs_dir: Path,
    figures_dir: Path,
) -> None:
    """Generate the three 300 DPI Group 1 figures."""
    if not summary_rows:
        raise ValueError("No Group 1 summary rows available for plotting.")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.28,
        }
    )

    labels = [f"{r['exp_id']}\n{r['model_name']}" for r in summary_rows]
    short_names = [r["model_name"] for r in summary_rows]
    val_accs = [float(r["val_acc"]) * 100.0 for r in summary_rows]
    val_f1s = [float(r["val_macro_f1"]) * 100.0 for r in summary_rows]
    params_m = [float(r["num_params"]) / 1e6 for r in summary_rows]
    best_idx = int(np.argmax(val_accs))

    # 1) Architecture comparison (Validation Accuracy / Macro-F1 + Efficiency vs Params)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), layout="constrained")
    x = np.arange(len(summary_rows))
    width = 0.36
    bars_acc = ax1.bar(x - width / 2, val_accs, width, label="Val Accuracy (%)", color="#1f6feb")
    bars_f1 = ax1.bar(x + width / 2, val_f1s, width, label="Val Macro-F1 (%)", color="#2ea043")

    bars_acc[best_idx].set_color("#0969da")
    bars_acc[best_idx].set_edgecolor("#0a3069")
    bars_acc[best_idx].set_linewidth(1.5)

    for bar, val in zip(bars_acc, val_accs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.6,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    for bar, val in zip(bars_f1, val_f1s):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.6,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=7.8,
        )

    ax1.axhline(67.0, color="#d29922", linestyle="--", linewidth=1.2, label="Demo best (67.0%)")
    ax1.axhline(69.0, color="#cf222e", linestyle=":", linewidth=1.2, label="Kaggle Public best (69.0%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("Score on PublicTest (%)")
    ax1.set_ylim(0, max(79.0, max(val_accs) + 10.5))
    ax1.set_title("Group 1: Validation Accuracy & Macro-F1 by Architecture")
    ax1.legend(loc="lower right", framealpha=0.92)

    palette = ["#6e7781", "#0969da", "#1a7f37", "#8250df", "#bf3989", "#cf222e"]
    custom_offsets = {
        "SmallCNN": (-48, 12),
        "MediumCNN": (-88, 12),
        "Medium1CNN": (10, 14),
        "LargeCNN": (10, -4),
        "ImprovedFERNet": (10, 6),
        "VGG16Transfer": (10, -24),
    }
    for idx, (pm, acc, name) in enumerate(zip(params_m, val_accs, short_names)):
        marker = "*" if idx == best_idx else ("X" if name == "VGG16Transfer" else "o")
        size = 220 if idx == best_idx else 110
        ax2.scatter(pm, acc, s=size, color=palette[idx % len(palette)], marker=marker, zorder=4)
        dx, dy = custom_offsets.get(name, (8, 8))
        ax2.annotate(
            f"{name}\n({pm:.1f}M, {acc:.2f}%)",
            (pm, acc),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8.2,
            arrowprops={"arrowstyle": "-", "color": "#8b949e", "lw": 0.7} if abs(dx) > 15 or abs(dy) > 15 else None,
        )
    ax2.set_xlabel("Trainable Parameters (Millions)")
    ax2.set_ylabel("PublicTest Validation Accuracy (%)")
    ax2.set_title("Accuracy vs. Parameter Efficiency")
    ax2.set_xlim(0, max(params_m) * 1.18)
    ax2.set_ylim(min(val_accs) - 2.0, max(val_accs) + 1.8)
    _save_figure(fig, figures_dir / "exp1_arch_comparison.png")

    # 2) Validation Learning Curves across EXP-01..EXP-06
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(13.2, 5.0), layout="constrained")
    for idx, row in enumerate(summary_rows):
        hist_path = Path(row["history_path"])
        if not hist_path.is_file():
            hist_path = logs_dir / f"history_{row['exp_id']}.csv"
        if not hist_path.is_file():
            continue
        hist = _load_csv_rows(hist_path)
        epochs = [int(h["epoch"]) for h in hist]
        v_acc = [float(h["val_acc"]) * 100.0 for h in hist]
        v_loss = [float(h["val_loss"]) for h in hist]
        color = palette[idx % len(palette)]
        ls = "--" if row["model_name"] == "VGG16Transfer" else "-"
        lw = 2.3 if idx == best_idx else 1.7
        label = f"{row['exp_id']} {row['model_name']} ({float(row['val_acc'])*100:.2f}%)"
        ax_acc.plot(epochs, v_acc, label=label, color=color, linestyle=ls, linewidth=lw)
        ax_loss.plot(epochs, v_loss, label=label, color=color, linestyle=ls, linewidth=lw)

    ax_acc.set_title("Validation Accuracy over Epochs (PublicTest)")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Validation Accuracy (%)")
    ax_acc.legend(loc="lower right", framealpha=0.92)

    ax_loss.set_title("Validation Loss over Epochs (PublicTest)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.legend(loc="upper right", framealpha=0.92)
    _save_figure(fig, figures_dir / "exp1_learning_curves.png")

    # 3) Dedicated Fail Case plot for EXP-06 (VGG16Transfer)
    vgg_row = next((r for r in summary_rows if r["exp_id"] == "EXP-06"), None)
    if vgg_row is not None:
        hist_path = Path(vgg_row["history_path"])
        if not hist_path.is_file():
            hist_path = logs_dir / "history_EXP-06.csv"
        if hist_path.is_file():
            hist = _load_csv_rows(hist_path)
            epochs = [int(h["epoch"]) for h in hist]
            tr_acc = [float(h["train_acc"]) * 100.0 for h in hist]
            va_acc = [float(h["val_acc"]) * 100.0 for h in hist]
            tr_loss = [float(h["train_loss"]) for h in hist]
            va_loss = [float(h["val_loss"]) for h in hist]
            best_ep = int(vgg_row["best_epoch"])
            best_va_loss = float(vgg_row["val_loss"])
            best_va_acc = float(vgg_row["val_acc"]) * 100.0

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8), layout="constrained")
            fig.suptitle(
                "Fail Case (EXP-06): Fine-Tuning VGG16 on 48×48 FER2013 Images",
                fontsize=12.5,
                fontweight="bold",
            )

            ax1.plot(epochs, tr_acc, label="Train Accuracy", color="#1f6feb", linewidth=2.0)
            ax1.plot(epochs, va_acc, label="Validation Accuracy", color="#cf222e", linewidth=2.0)
            ax1.axvline(best_ep, color="#6e7781", linestyle=":", linewidth=1.2, label=f"Best epoch ({best_ep})")
            ax1.fill_between(epochs, va_acc, tr_acc, color="#cf222e", alpha=0.10, label="Generalization gap")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Accuracy (%)")
            ax1.set_title(f"Accuracy (Best Val = {best_va_acc:.2f}%, Final Train = {tr_acc[-1]:.2f}%)")
            ax1.legend(loc="lower right")

            ax2.plot(epochs, tr_loss, label="Train Loss", color="#1f6feb", linewidth=2.0)
            ax2.plot(epochs, va_loss, label="Validation Loss", color="#cf222e", linewidth=2.0)
            ax2.axvline(best_ep, color="#6e7781", linestyle=":", linewidth=1.2, label=f"Best val_loss ({best_va_loss:.3f})")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Cross-Entropy Loss")
            ax2.set_title("Loss Divergence & Overfitting Behavior")
            ax2.legend(loc="upper left")

            _save_figure(fig, figures_dir / "exp1_failcase_vgg16.png")


def run_group1(
    data_path: str,
    selected_ids: set[str] | None = None,
    max_epochs: int = 60,
    patience: int = 12,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    device: str = "auto",
    logs_dir: Path = Path("logs"),
    checkpoints_dir: Path = Path("checkpoints"),
    figures_dir: Path = Path("figures"),
    plot_only: bool = False,
) -> list[dict[str, str]]:
    """Run Group 1 architecture experiments and export summary CSV + figures."""
    runs_csv = logs_dir / "experiment_runs.csv"
    summary_csv = logs_dir / "exp1_architectures_summary.csv"

    if not plot_only:
        seed_everything(seed)
        loaders = get_dataloaders(
            data_path=data_path,
            aug_mode="standard",
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        )
        # Strict Zero Data Leakage: NEVER touch loaders["test"] (PrivateTest) in Group 1.
        train_loader = loaders["train"]
        val_loader = loaders["val"]

        best_val_acc = -1.0
        best_per_class = None

        for spec in EXP1_SPECS:
            if selected_ids and spec["exp_id"] not in selected_ids:
                continue

            seed_everything(seed)
            model = build_model(
                name=spec["model_name"],
                num_classes=7,
                activation="relu",
                use_bn=spec["use_bn"],
                dropout=spec["dropout"],
                pretrained=spec["pretrained"],
            )
            config = {
                "exp_id": spec["exp_id"],
                "owner": "TV2 - Chau Quoc Bao (3123411024)",
                "group_name": "Group1_Architectures",
                "model_name": spec["model_name"],
                "num_classes": 7,
                "activation": "relu",
                "use_bn": spec["use_bn"],
                "dropout": spec["dropout"],
                "optimizer": "adam",
                "lr": spec["lr"],
                "weight_decay": 0.0,
                "scheduler": "plateau",
                "scheduler_factor": 0.5,
                "scheduler_patience": 4,
                "min_lr": 1e-6,
                "loss_type": "ce",
                "aug_mode": "standard",
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "patience": patience,
                "seed": seed,
                "device": device,
                "logs_dir": str(logs_dir),
                "checkpoint_dir": str(checkpoints_dir),
                "runs_csv": str(runs_csv),
                "notes": spec["notes"],
            }
            result = train_model(model, train_loader, val_loader, config)
            if result["best_val_acc"] > best_val_acc:
                best_val_acc = result["best_val_acc"]
                best_per_class = result["best_per_class"]
            del model

        if best_per_class is not None:
            save_per_class_metrics_csv(best_per_class, logs_dir / "exp1_best_val_per_class.csv")

    summary_rows = export_summary_csv(runs_csv, summary_csv)
    plot_exp1_figures(summary_rows, logs_dir, figures_dir)
    print(f"\nExported Group 1 summary: {summary_csv}")
    print(f"Exported Group 1 figures into: {figures_dir}")
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/fer2013.csv", help="FER2013 CSV or split directory")
    parser.add_argument(
        "--exp-ids",
        nargs="*",
        default=None,
        help="Optional subset of experiment IDs (e.g. EXP-01 EXP-05)",
    )
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--plot-only", action="store_true", help="Regenerate summary CSV and plots only")
    args = parser.parse_args()

    selected = set(args.exp_ids) if args.exp_ids else None
    run_group1(
        data_path=args.data_path,
        selected_ids=selected,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        logs_dir=args.logs_dir,
        checkpoints_dir=args.checkpoints_dir,
        figures_dir=args.figures_dir,
        plot_only=args.plot_only,
    )


if __name__ == "__main__":
    main()

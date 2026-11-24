"""Shared 300-DPI plotting utilities for the FER2013 Deep Learning project.

Author: TV4 - Vo Huynh Minh Sang (MSSV: 3123411256)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = (
    "#0969DA",
    "#CF222E",
    "#1A7F37",
    "#9A6700",
    "#8250DF",
    "#0A3069",
    "#BF3989",
)


def _save_figure(fig: plt.Figure, output_path: str | Path, dpi: int = 300) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def load_history_csv(history_path: str | Path) -> dict[str, np.ndarray]:
    """Load epoch-level training history from a CSV exported by src.trainer."""
    epochs: list[int] = []
    train_loss: list[float] = []
    train_acc: list[float] = []
    val_loss: list[float] = []
    val_acc: list[float] = []
    val_macro_f1: list[float] = []
    lr: list[float] = []

    with Path(history_path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            epochs.append(int(float(row["epoch"])))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_acc"]))
            val_macro_f1.append(float(row.get("val_macro_f1", 0.0)))
            lr.append(float(row.get("lr", 0.0)))

    return {
        "epoch": np.asarray(epochs, dtype=np.int64),
        "train_loss": np.asarray(train_loss, dtype=np.float64),
        "train_acc": np.asarray(train_acc, dtype=np.float64),
        "val_loss": np.asarray(val_loss, dtype=np.float64),
        "val_acc": np.asarray(val_acc, dtype=np.float64),
        "val_macro_f1": np.asarray(val_macro_f1, dtype=np.float64),
        "lr": np.asarray(lr, dtype=np.float64),
    }


def plot_learning_curves(
    history: Mapping[str, Sequence[float]] | str | Path,
    output_path: str | Path,
    title: str = "Learning Curves (Loss & Accuracy)",
    best_epoch: int | None = None,
    dpi: int = 300,
) -> Path:
    """Plot Loss and Accuracy learning curves side-by-side at 300 DPI."""
    data = load_history_csv(history) if isinstance(history, (str, Path)) else history
    epochs = np.asarray(data["epoch"], dtype=np.int64)
    train_loss = np.asarray(data["train_loss"], dtype=np.float64)
    val_loss = np.asarray(data["val_loss"], dtype=np.float64)
    train_acc = np.asarray(data["train_acc"], dtype=np.float64) * 100.0
    val_acc = np.asarray(data["val_acc"], dtype=np.float64) * 100.0

    if best_epoch is None and val_loss.size > 0:
        best_epoch = int(epochs[int(np.argmin(val_loss))])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), layout="constrained")
    fig.suptitle(title, fontsize=13, fontweight="bold", color="#1F2328")

    # Left: Loss
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train Loss", color="#0969DA", linewidth=2.0)
    ax.plot(epochs, val_loss, label="PublicTest (Val) Loss", color="#CF222E", linewidth=2.0)
    if best_epoch is not None:
        ax.axvline(best_epoch, color="#57606A", linestyle="--", linewidth=1.2, label=f"Best Epoch ({best_epoch})")
    ax.set_title("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)

    # Right: Accuracy
    ax = axes[1]
    ax.plot(epochs, train_acc, label="Train Accuracy", color="#0969DA", linewidth=2.0)
    ax.plot(epochs, val_acc, label="PublicTest (Val) Accuracy", color="#1A7F37", linewidth=2.0)
    if "val_macro_f1" in data and np.any(np.asarray(data["val_macro_f1"]) > 0):
        val_f1 = np.asarray(data["val_macro_f1"], dtype=np.float64) * 100.0
        ax.plot(epochs, val_f1, label="PublicTest Macro-F1", color="#8250DF", linewidth=1.8, linestyle="-.")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="#57606A", linestyle="--", linewidth=1.2, label=f"Best Epoch ({best_epoch})")
    ax.set_title("Accuracy & Macro-F1 (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (%)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)

    return _save_figure(fig, output_path, dpi=dpi)


def plot_experiment_comparison(
    labels: Sequence[str],
    val_accs: Sequence[float],
    val_macro_f1s: Sequence[float] | None,
    output_path: str | Path,
    title: str = "Experiment Comparison",
    ylabel: str = "Score (%)",
    dpi: int = 300,
) -> Path:
    """Plot grouped bar chart comparing Accuracy and Macro-F1 across experiments."""
    x = np.arange(len(labels))
    acc_vals = np.asarray(val_accs, dtype=np.float64)
    if acc_vals.max() <= 1.0:
        acc_vals = acc_vals * 100.0

    fig, ax = plt.subplots(figsize=(max(8.5, 1.35 * len(labels)), 4.8), layout="constrained")
    if val_macro_f1s is not None:
        f1_vals = np.asarray(val_macro_f1s, dtype=np.float64)
        if f1_vals.max() <= 1.0:
            f1_vals = f1_vals * 100.0
        width = 0.36
        bars1 = ax.bar(x - width / 2, acc_vals, width, label="Accuracy (%)", color="#0969DA")
        bars2 = ax.bar(x + width / 2, f1_vals, width, label="Macro-F1 (%)", color="#1A7F37")
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8.5)
    else:
        bars1 = ax.bar(x, acc_vals, width=0.52, color="#0969DA", label="Accuracy (%)")
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12.5, fontweight="bold", color="#1F2328")
    ax.set_ylim(0, min(100.0, max(acc_vals.max() + 8.0, 75.0)))
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(frameon=True, fontsize=9.5)

    return _save_figure(fig, output_path, dpi=dpi)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path,
    title: str = "Confusion Matrix on PrivateTest",
    normalize: bool = True,
    dpi: int = 300,
) -> Path:
    """Plot a dual-annotated (count + row percentage) confusion matrix at 300 DPI."""
    cm_arr = np.asarray(cm, dtype=np.int64)
    row_sums = np.maximum(cm_arr.sum(axis=1, keepdims=True), 1)
    cm_norm = cm_arr / row_sums * 100.0

    fig, ax = plt.subplots(figsize=(8.6, 7.2), layout="constrained")
    display_matrix = cm_norm if normalize else cm_arr.astype(np.float64)
    im = ax.imshow(display_matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=100 if normalize else None)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (%)" if normalize else "Sample Count", fontsize=10)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12.5, fontweight="bold", color="#1F2328", pad=12)

    threshold = display_matrix.max() * 0.55
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            count = cm_arr[i, j]
            pct = cm_norm[i, j]
            text = f"{count}\n({pct:.1f}%)"
            color = "white" if display_matrix[i, j] > threshold else "#1F2328"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8.5, fontweight=weight)

    return _save_figure(fig, output_path, dpi=dpi)


def plot_prediction_grid(
    images: Sequence[np.ndarray],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    confidences: Sequence[float] | None,
    class_names: Sequence[str],
    output_path: str | Path,
    title: str = "Prediction Samples",
    nrows: int = 4,
    ncols: int = 8,
    dpi: int = 300,
) -> Path:
    """Plot a grid (default 4x8 = 32 images) of True or Wrong predictions."""
    total_slots = nrows * ncols
    count = min(total_slots, len(images))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.55, nrows * 1.85), layout="constrained")
    fig.suptitle(title, fontsize=13.5, fontweight="bold", color="#1F2328")

    flat_axes = np.asarray(axes).reshape(-1)
    for idx in range(total_slots):
        ax = flat_axes[idx]
        ax.axis("off")
        if idx >= count:
            continue
        img = np.asarray(images[idx])
        if img.ndim == 3:
            img = img.squeeze()
        ax.imshow(img, cmap="gray", vmin=0, vmax=255 if img.max() > 1.5 else 1.0)
        true_name = class_names[int(y_true[idx])]
        pred_name = class_names[int(y_pred[idx])]
        is_correct = int(y_true[idx]) == int(y_pred[idx])
        color = "#1A7F37" if is_correct else "#CF222E"
        if confidences is not None:
            conf_str = f" ({confidences[idx] * 100:.0f}%)"
        else:
            conf_str = ""
        if is_correct:
            label_text = f"{pred_name}{conf_str}"
        else:
            label_text = f"P: {pred_name}{conf_str}\nT: {true_name}"
        ax.set_title(label_text, fontsize=8.0, color=color, fontweight="bold" if not is_correct else "normal", pad=3)

    return _save_figure(fig, output_path, dpi=dpi)

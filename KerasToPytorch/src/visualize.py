"""Shared 300-DPI plotting utilities for the KerasToPytorch project.

Author: TV4 - Vo Huynh Minh Sang (MSSV: 3123411256)
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


SEG_PALETTE = ListedColormap([
    "#0D1117",  # 0: Background
    "#F2CC60",  # 1: Palm
    "#FF7B72",  # 2: Thumb
    "#79C0FF",  # 3: Index
    "#56D364",  # 4: Middle
    "#D2A8FF",  # 5: Ring
    "#FFA657",  # 6: Pinky
])


def _save_fig(fig: plt.Figure, output_path: str | Path, dpi: int = 300) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_learning_curves(
    history: Sequence[Mapping[str, float]],
    output_path: str | Path,
    metric_key: str = "pixel_accuracy",
    metric_label: str = "Pixel Accuracy (%)",
    title: str = "Training & Validation Learning Curves",
    scale_percent: bool = True,
    dpi: int = 300,
) -> Path:
    """Plot Train/Val Loss and Metric curves side-by-side at 300 DPI."""
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    factor = 100.0 if scale_percent else 1.0
    train_m = [float(row[f"train_{metric_key}"]) * factor for row in history]
    val_m = [float(row[f"val_{metric_key}"]) * factor for row in history]
    best_idx = int(np.argmin(val_loss))
    best_ep = epochs[best_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), layout="constrained")
    fig.suptitle(title, fontsize=13, fontweight="bold", color="#1F2328")

    ax0 = axes[0]
    ax0.plot(epochs, train_loss, label="Train Loss", color="#0969DA", linewidth=2.0)
    ax0.plot(epochs, val_loss, label="Validation Loss", color="#CF222E", linewidth=2.0)
    ax0.axvline(best_ep, color="#57606A", linestyle="--", linewidth=1.2, label=f"Best Epoch ({best_ep})")
    ax0.set_title("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")
    ax0.grid(True, linestyle=":", alpha=0.5)
    ax0.legend(frameon=True, fontsize=9)

    ax1 = axes[1]
    ax1.plot(epochs, train_m, label=f"Train {metric_label}", color="#0969DA", linewidth=2.0)
    ax1.plot(epochs, val_m, label=f"Val {metric_label}", color="#1A7F37", linewidth=2.0)
    ax1.axvline(best_ep, color="#57606A", linestyle="--", linewidth=1.2, label=f"Best Epoch ({best_ep})")
    ax1.set_title(metric_label, fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(metric_label)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(frameon=True, fontsize=9)

    return _save_fig(fig, output_path, dpi=dpi)


def plot_before_after_comparison(
    labels: Sequence[str],
    before_vals: Sequence[float],
    after_vals: Sequence[float],
    output_path: str | Path,
    title: str = "Before vs After Comparison",
    ylabel: str = "Score (%)",
    before_label: str = "Baseline",
    after_label: str = "Improved",
    dpi: int = 300,
) -> Path:
    """Plot grouped bar chart comparing Baseline vs Improved configurations."""
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(7.5, 1.5 * len(labels)), 4.6), layout="constrained")
    b1 = ax.bar(x - width / 2, before_vals, width, label=before_label, color="#8C959F")
    b2 = ax.bar(x + width / 2, after_vals, width, label=after_label, color="#0969DA")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12.5, fontweight="bold", color="#1F2328")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(frameon=True, fontsize=9.5)
    return _save_fig(fig, output_path, dpi=dpi)


def plot_segmentation_results(
    depth_images: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    pred_masks: Sequence[np.ndarray],
    class_names: Sequence[str],
    output_path: str | Path,
    title: str = "SegNet Hand Depth Segmentation — Input | Ground Truth | Prediction",
    dpi: int = 300,
) -> Path:
    """Plot a 3-column visual comparison grid: Input Depth | Ground Truth Mask | Predicted Mask."""
    n = len(depth_images)
    num_classes = len(class_names)
    fig, axes = plt.subplots(n, 3, figsize=(8.8, n * 2.15 + 0.9), layout="constrained")
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(title, fontsize=13, fontweight="bold", color="#1F2328")

    col_titles = ["Input Depth (96×96)", "Ground Truth Mask", "Predicted Mask (SegNet)"]
    for row in range(n):
        depth = np.asarray(depth_images[row]).squeeze()
        gt = np.asarray(gt_masks[row])
        pred = np.asarray(pred_masks[row])
        pix_acc = float((pred == gt).mean()) * 100.0

        axes[row, 0].imshow(depth, cmap="magma")
        axes[row, 1].imshow(gt, cmap=SEG_PALETTE, vmin=0, vmax=num_classes - 1, interpolation="nearest")
        axes[row, 2].imshow(pred, cmap=SEG_PALETTE, vmin=0, vmax=num_classes - 1, interpolation="nearest")

        for col in range(3):
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=10.5, fontweight="bold", color="#0F3460")
        axes[row, 2].text(
            0.03, 0.05, f"PixAcc: {pix_acc:.1f}%",
            transform=axes[row, 2].transAxes,
            fontsize=8.2, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#0F3460", alpha=0.82),
        )

    # Legend strip at bottom
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=f"{i}: {name}",
                   markerfacecolor=SEG_PALETTE(i), markersize=9)
        for i, name in enumerate(class_names)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(7, num_classes), frameon=True, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    return _save_fig(fig, output_path, dpi=dpi)

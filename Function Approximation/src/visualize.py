"""
Module: visualize.py
Các hàm vẽ biểu đồ trực quan: 3D Scatter, mặt lý thuyết 3D, đường cong học tập,
xu hướng sai số theo tham số thực nghiệm, Parity/Residual plot và đối sánh bề mặt 3D.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List, Union, Dict
import pandas as pd


def plot_3d_scatter(
    X: np.ndarray,
    y: np.ndarray,
    title: str = "3D Scatter of Generated Dataset",
    save_path: Optional[str] = None
) -> None:
    """Vẽ biểu đồ phân tán 3D các điểm dữ liệu (x1, x2, y)."""
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    if X.ndim != 2 or X.shape[1] != 2 or len(X) != len(y) or len(X) == 0:
        raise ValueError("X phải có shape (N, 2), y có N giá trị và N > 0.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Dữ liệu phải chứa các số hữu hạn.")
    fig = plt.figure(figsize=(9, 7))
    try:
        ax = fig.add_subplot(111, projection="3d")
        points = ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap="viridis", s=24)
        ax.set(xlabel="x1", ylabel="x2", zlabel="Mật độ y", title=title)
        fig.colorbar(points, ax=ax, shrink=0.65, pad=0.12, label="Mật độ y")
        fig.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
    finally:
        plt.close(fig)


def plot_ground_truth_surface(
    pdf_func,
    x1_range=(-2.0, 2.0),
    x2_range=(-4.0, 4.0),
    grid_size: int = 100,
    save_path: Optional[str] = None
) -> None:
    """Vẽ bề mặt 3D hàm Gaussian lý thuyết trên lưới điểm đều (Meshgrid)."""
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("grid_size phải là số nguyên từ 2 trở lên.")
    for bounds in (x1_range, x2_range):
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError("Khoảng vẽ phải có cận dưới nhỏ hơn cận trên và hữu hạn.")
    X1, X2 = np.meshgrid(np.linspace(*x1_range, grid_size), np.linspace(*x2_range, grid_size))
    points = np.column_stack((X1.ravel(), X2.ravel()))
    values = np.asarray(pdf_func(points))
    if values.size != len(points) or not np.isfinite(values).all():
        raise ValueError("pdf_func phải trả về một giá trị hữu hạn cho mỗi điểm.")
    Y = values.reshape(X1.shape)
    fig = plt.figure(figsize=(9, 7))
    try:
        ax = fig.add_subplot(111, projection="3d")
        surface = ax.plot_surface(X1, X2, Y, cmap="coolwarm", linewidth=0, antialiased=True)
        ax.set(xlabel="x1", ylabel="x2", zlabel="Mật độ y", title="Bề mặt Gaussian lý thuyết")
        fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.12, label="Mật độ y")
        fig.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
    finally:
        plt.close(fig)


def plot_learning_curve(
    history_df: Union[pd.DataFrame, Dict[str, List[float]]],
    title: str = "Learning Curve (Loss vs Epochs)",
    save_path: Optional[str] = None
) -> None:
    """
    Vẽ đường cong Train Loss và Validation Loss theo Epoch.
    Hỗ trợ DataFrame có các cột ['epoch', 'train_loss', 'val_loss']
    hoặc Dict chứa danh sách loss tương ứng.
    (TV4 hoàn thành từ Giai đoạn 1 để TV3 và TV4 sử dụng).
    """
    if isinstance(history_df, dict):
        df = pd.DataFrame(history_df)
    else:
        df = history_df.copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        epochs = df["epoch"] if "epoch" in df.columns else np.arange(1, len(df) + 1)
        if "train_loss" in df.columns:
            ax.plot(epochs, df["train_loss"], label="Train Loss (MSE)", color="tab:blue", linewidth=2)
        if "val_loss" in df.columns:
            ax.plot(epochs, df["val_loss"], label="Val Loss (MSE)", color="tab:orange", linewidth=2)
            best_idx = int(df["val_loss"].idxmin())
            best_ep = epochs.iloc[best_idx] if hasattr(epochs, "iloc") else epochs[best_idx]
            best_val = df["val_loss"].iloc[best_idx] if hasattr(df["val_loss"], "iloc") else df["val_loss"][best_idx]
            ax.scatter([best_ep], [best_val], color="red", s=60, zorder=5, label=f"Best Val ({best_val:.5f} @ ep {best_ep})")

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss (MSE)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=10)
        fig.tight_layout()

        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
    finally:
        plt.close(fig)


def plot_experiment_trend(
    x_vals: List[Union[int, float]],
    train_errors: List[float],
    val_errors: Optional[List[float]] = None,
    test_errors: Optional[List[float]] = None,
    x_label: str = "Biến độc lập",
    y_label: str = "MSE",
    title: str = "Kết Quả Thực Nghiệm",
    std_test: Optional[List[float]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Vẽ đồ thị sai số theo biến độc lập (hỗ trợ vẽ các đường Train/Val/Test và dải sai số ± std).
    (TV4 hoàn thành từ Giai đoạn 1 để bàn giao cho TV2 và TV3).
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    try:
        x_arr = np.array(x_vals)
        ax.plot(x_arr, train_errors, marker='o', linewidth=2, color='tab:blue', label='Train MSE')
        if val_errors is not None:
            ax.plot(x_arr, val_errors, marker='s', linewidth=2, color='tab:orange', label='Validation MSE')
        if test_errors is not None:
            ax.plot(x_arr, test_errors, marker='^', linewidth=2, color='tab:green', label='Test MSE (Mean)')
            if std_test is not None:
                std_arr = np.array(std_test)
                test_arr = np.array(test_errors)
                ax.fill_between(x_arr, test_arr - std_arr, test_arr + std_arr, color='tab:green', alpha=0.2, label='Test MSE (±1 Std)')

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10)
        fig.tight_layout()

        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    finally:
        plt.close(fig)


def plot_3d_model_comparison(
    pdf_func,
    model,
    device,
    x1_range=(-2.0, 2.0),
    x2_range=(-4.0, 4.0),
    grid_size: int = 60,
    save_path: Optional[str] = None
) -> None:
    """
    Vẽ so sánh song song 2 bề mặt 3D: Hàm thật vs Mạng MLP dự đoán.
    Cả hai bề mặt dùng chung thang màu (vmin/vmax) để đối sánh trực quan công bằng.
    (TV4 hoàn thành Giai đoạn 3).
    """
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("grid_size phải là số nguyên từ 2 trở lên.")
    for bounds in (x1_range, x2_range):
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError("Khoảng vẽ phải có cận dưới nhỏ hơn cận trên và hữu hạn.")

    X1, X2 = np.meshgrid(np.linspace(*x1_range, grid_size), np.linspace(*x2_range, grid_size))
    points = np.column_stack((X1.ravel(), X2.ravel()))

    Y_true = np.asarray(pdf_func(points)).reshape(X1.shape)
    if Y_true.shape != X1.shape or not np.isfinite(Y_true).all():
        raise ValueError("pdf_func phải trả về một giá trị hữu hạn cho mỗi điểm lưới.")

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(points, dtype=torch.float32).to(device))
    Y_pred = preds.detach().cpu().numpy().astype(np.float64).reshape(X1.shape)

    vmin = float(min(Y_true.min(), Y_pred.min()))
    vmax = float(max(Y_true.max(), Y_pred.max()))

    fig = plt.figure(figsize=(14, 6))
    try:
        ax_true = fig.add_subplot(121, projection="3d")
        surf_true = ax_true.plot_surface(
            X1, X2, Y_true, cmap="coolwarm", vmin=vmin, vmax=vmax,
            linewidth=0, antialiased=True
        )
        ax_true.set(xlabel="x1", ylabel="x2", zlabel="Mật độ y", title="Bề mặt Gaussian lý thuyết")

        ax_pred = fig.add_subplot(122, projection="3d")
        surf_pred = ax_pred.plot_surface(
            X1, X2, Y_pred, cmap="coolwarm", vmin=vmin, vmax=vmax,
            linewidth=0, antialiased=True
        )
        ax_pred.set(xlabel="x1", ylabel="x2", zlabel="Mật độ y", title="Bề mặt MLP dự đoán")

        fig.colorbar(surf_pred, ax=[ax_true, ax_pred], shrink=0.6, pad=0.08, label="Mật độ y")
        fig.suptitle("Đối sánh bề mặt 3D: Hàm thật vs Mô hình MLP", fontsize=13, fontweight="bold", y=0.98)
        fig.subplots_adjust(top=0.85, left=0.03, right=0.95, bottom=0.05, wspace=0.05)
        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
    finally:
        plt.close(fig)


def plot_parity_and_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title_prefix: str = "Tập Test",
    save_path: Optional[str] = None
) -> None:
    """
    Vẽ Parity Plot (y vs ŷ kèm đường 45 độ y = ŷ) và biểu đồ phần dư (y - ŷ vs ŷ).
    (TV4 hoàn thành Giai đoạn 3).
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("y_true và y_pred phải cùng độ dài và khác rỗng.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("y_true và y_pred phải chứa các số hữu hạn.")
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    try:
        # Panel trái: Parity Plot y vs ŷ kèm đường chuẩn 45 độ
        ax_parity = axes[0]
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        margin = 0.05 * (hi - lo) if hi > lo else 0.05
        lo -= margin
        hi += margin
        ax_parity.scatter(
            y_true, y_pred, c="tab:blue", s=34, alpha=0.75,
            edgecolors="white", linewidths=0.5, label="Mẫu dự đoán"
        )
        ax_parity.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1.8,
                       label="Đường chuẩn $y = \\hat{y}$")
        ax_parity.set_xlim(lo, hi)
        ax_parity.set_ylim(lo, hi)
        ax_parity.set_aspect("equal", adjustable="box")
        ax_parity.set_xlabel("Giá trị thật $y$", fontsize=11)
        ax_parity.set_ylabel("Giá trị dự đoán $\\hat{y}$", fontsize=11)
        ax_parity.set_title(f"Parity Plot ({title_prefix})", fontsize=12, fontweight="bold")
        ax_parity.grid(True, linestyle="--", alpha=0.6)
        ax_parity.legend(fontsize=10)

        # Panel phải: Residual Plot phần dư y - ŷ theo ŷ
        ax_res = axes[1]
        ax_res.scatter(
            y_pred, residuals, c="tab:orange", s=34, alpha=0.75,
            edgecolors="white", linewidths=0.5, label="Phần dư từng mẫu"
        )
        ax_res.axhline(0.0, color="red", linestyle="--", linewidth=1.8, label="Phần dư = 0")
        ax_res.set_xlabel("Giá trị dự đoán $\\hat{y}$", fontsize=11)
        ax_res.set_ylabel("Phần dư $y - \\hat{y}$", fontsize=11)
        ax_res.set_title(f"Residual Plot ({title_prefix})", fontsize=12, fontweight="bold")
        ax_res.grid(True, linestyle="--", alpha=0.6)
        ax_res.legend(fontsize=10)

        fig.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
    finally:
        plt.close(fig)


if __name__ == "__main__":
    from src.data_generator import gaussian_2d_pdf, load_dataset_from_csv

    X, y = load_dataset_from_csv("data/test_dataset_100.csv")
    plot_3d_scatter(X, y, title="100 mẫu test", save_path="figures/test_3d_scatter.png")
    plot_ground_truth_surface(gaussian_2d_pdf, save_path="figures/ground_truth_surface.png")
    print("Đã tạo hai hình 3D trong thư mục figures.")

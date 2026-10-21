"""
================================================================================
KHUNG SƯỜN MÃ NGUỒN (SKELETON TEMPLATE)
MODULE: visualize.py
PHỤ TRÁCH: THÀNH VIÊN 1 & THÀNH VIÊN 4
MÔ TẢ NHIỆM VỤ:
    1. Cài đặt hàm vẽ phân tán 3D Scatter (TV1) -> figures/test_3d_scatter.png.
    2. Cài đặt hàm vẽ bề mặt 3D lý thuyết (TV1) -> figures/ground_truth_surface.png.
    3. Cài đặt hàm vẽ đường cong học tập (TV4 - hoàn thành từ Giai đoạn 1):
       Dùng cho Thực nghiệm C (figures/exp_C_loss_convergence.png) và Best Model (figures/best_model_learning_curve.png).
    4. Cài đặt hàm vẽ đồ thị xu hướng sai số theo biến độc lập (TV4 - hoàn thành từ Giai đoạn 1):
       Bàn giao ngay để TV2 vẽ exp_A_samples_vs_error.png và TV3 vẽ exp_B_width_vs_error.png, exp_C_depth_vs_error.png.
    5. Cài đặt hàm vẽ đối sánh 3D song song: Bề mặt thật vs MLP dự đoán (TV4) -> figures/best_model_3d_comparison.png.
    6. Cài đặt hàm vẽ Parity Plot và Residuals (TV4) -> figures/best_model_parity_and_residuals.png.
================================================================================
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
    # TODO (Thành viên 1):
    # ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap='viridis'). Lưu hình nếu có save_path.
    raise NotImplementedError("TODO (Thành viên 1): Hãy triển khai hàm plot_3d_scatter.")


def plot_ground_truth_surface(
    pdf_func,
    x1_range=(-2.0, 2.0),
    x2_range=(-4.0, 4.0),
    grid_size: int = 100,
    save_path: Optional[str] = None
) -> None:
    """Vẽ bề mặt 3D hàm Gaussian lý thuyết trên lưới điểm đều (Meshgrid)."""
    # TODO (Thành viên 1):
    # np.meshgrid -> pdf_func -> ax.plot_surface(X1, X2, Y, cmap='coolwarm').
    raise NotImplementedError("TODO (Thành viên 1): Hãy triển khai hàm plot_ground_truth_surface.")


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
    # TODO (Thành viên 4):
    # Vẽ Train Loss và Val Loss qua các epoch; hiển thị nhãn, chú giải, điểm epoch tốt nhất.
    raise NotImplementedError("TODO (Thành viên 4): Hãy triển khai hàm plot_learning_curve.")


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
    # TODO (Thành viên 4):
    # Vẽ các đường sai số theo x_vals; dùng ax.fill_between nếu có std_test; lưu hình nếu có save_path.
    raise NotImplementedError("TODO (Thành viên 4): Hãy triển khai hàm plot_experiment_trend.")


def plot_3d_model_comparison(
    pdf_func,
    model,
    device,
    x1_range=(-2.0, 2.0),
    x2_range=(-4.0, 4.0),
    grid_size: int = 60,
    save_path: Optional[str] = None
) -> None:
    """Vẽ so sánh song song 2 bề mặt 3D: Hàm thật vs Mạng MLP dự đoán."""
    # TODO (Thành viên 4):
    # Tạo 2 subplot 3D: bên trái hàm thật Y_true, bên phải hàm dự đoán Y_pred từ model.
    raise NotImplementedError("TODO (Thành viên 4): Hãy triển khai hàm plot_3d_model_comparison.")


def plot_parity_and_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title_prefix: str = "Tập Test",
    save_path: Optional[str] = None
) -> None:
    """Vẽ Parity Plot (y vs ŷ kèm đường 45 độ y = ŷ) và biểu đồ phần dư (y - ŷ)."""
    # TODO (Thành viên 4):
    # Vẽ 2 subplot: Parity plot bên trái và Residuals plot bên phải.
    raise NotImplementedError("TODO (Thành viên 4): Hãy triển khai hàm plot_parity_and_residuals.")


if __name__ == "__main__":
    print("--- KIỂM TRA MODULE VISUALIZE (THÀNH VIÊN 1 & 4) ---")

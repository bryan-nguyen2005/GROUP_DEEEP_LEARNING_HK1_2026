"""
================================================================================
KHUNG SƯỜN MÃ NGUỒN (SKELETON TEMPLATE)
MODULE: data_generator.py
PHỤ TRÁCH: THÀNH VIÊN 1 (Data & Architecture Lead)
MÔ TẢ NHIỆM VỤ:
    1. Cài đặt hàm tính mật độ xác suất Gaussian 2D theo đúng công thức giải tích.
    2. Cài đặt hàm sinh ngẫu nhiên mẫu dữ liệu (X, y) trong miền [-2, 2] x [-4, 4].
    3. Cài đặt các hàm lưu và đọc dữ liệu ra/vào file CSV.
    4. Cài đặt hàm chuẩn bị 2 tập dữ liệu chuẩn:
        - Tập Test độc lập cố định gồm đúng 100 mẫu (lưu vào data/test_dataset_100.csv, seed=42).
        - Tập Data Pool gồm 5000 mẫu (lưu vào data/train_val_pool.csv, seed=2024).
    5. Cài đặt hàm tiện ích get_train_val_split thực hiện phân chia lát cắt chỉ số cố định:
        - Tập Val: 400 mẫu đầu tiên (iloc[0:400]).
        - Tập Train: lấy từ Train Pool (iloc[400:400+n_train]), bảo đảm Zero Data Leakage.
================================================================================
"""

import os
import sys

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi mã hóa
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def gaussian_2d_pdf(
    x: np.ndarray,
    mu: Optional[np.ndarray] = None,
    cov: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Tính giá trị mật độ xác suất của phân bố Gaussian hai chiều:
    f(x) = 1 / (2*pi*sqrt(det(C))) * exp(-0.5 * (x - mu)^T * C^-1 * (x - mu))

    Tham số:
        x: Mảng numpy kích thước (N, 2) chứa các tọa độ điểm (x1, x2).
        mu: Vector kỳ vọng (2,), mặc định [0.0, 0.0].
        cov: Ma trận hiệp phương sai C (2, 2), mặc định [[0.3, -0.5], [-0.5, 2.0]].

    Trả về:
        y: Mảng numpy kích thước (N, 1) chứa giá trị hàm mật độ xác suất tại các điểm.
    """
    if mu is None:
        mu = np.array([0.0, 0.0], dtype=np.float64)
    if cov is None:
        cov = np.array([[0.3, -0.5], [-0.5, 2.0]], dtype=np.float64)

    # TODO (Thành viên 1):
    # Bước 1: Kiểm tra chiều của x (phải là N x 2).
    # Bước 2: Tính định thức det(C) và ma trận nghịch đảo C^-1 bằng np.linalg.det và np.linalg.inv.
    # Bước 3: Tính hệ số chuẩn hóa 1 / (2 * pi * sqrt(det(C))).
    # Bước 4: Tính khoảng cách Mahalanobis (x - mu)^T * C^-1 * (x - mu) cho từng mẫu.
    # Bước 5: Nhân với exp(-0.5 * khoảng cách) và trả về mảng shape (N, 1).
    raise NotImplementedError("TODO (Thành viên 1): Hãy triển khai hàm gaussian_2d_pdf theo công thức.")


def generate_samples(
    n_samples: int,
    x1_range: Tuple[float, float] = (-2.0, 2.0),
    x2_range: Tuple[float, float] = (-4.0, 4.0),
    mu: Optional[np.ndarray] = None,
    cov: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sinh ngẫu nhiên n_samples điểm phân bố đều trong miền [x1_range] x [x2_range]
    và tính nhãn mục tiêu y = gaussian_2d_pdf(x).
    """
    # TODO (Thành viên 1):
    # Bước 1: Nếu seed is not None, gọi np.random.seed(seed).
    # Bước 2: Dùng np.random.uniform để sinh x1 trong x1_range và x2 trong x2_range.
    # Bước 3: Ghép lại thành X có kích thước (n_samples, 2).
    # Bước 4: Gọi hàm gaussian_2d_pdf(X, mu, cov) để tính y.
    # Bước 5: Trả về (X, y).
    raise NotImplementedError("TODO (Thành viên 1): Hãy triển khai hàm generate_samples.")


def save_dataset_to_csv(X: np.ndarray, y: np.ndarray, file_path: str) -> None:
    """
    Lưu dữ liệu X (2 cột) và y (1 cột) vào file CSV với header [x1, x2, y].
    """
    # TODO (Thành viên 1):
    # Tạo DataFrame từ X và y, đảm bảo thư mục cha tồn tại (os.makedirs), lưu ra file_path.
    raise NotImplementedError("TODO (Thành viên 1): Hãy triển khai hàm save_dataset_to_csv.")


def load_dataset_from_csv(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Đọc dữ liệu từ file CSV, trả về X có shape (N, 2) và y có shape (N, 1).
    """
    # TODO (Thành viên 1):
    # Đọc CSV bằng pandas, lấy cột 'x1', 'x2' làm X (float32) và cột 'y' làm y (float32).
    raise NotImplementedError("TODO (Thành viên 1): Hãy triển khai hàm load_dataset_from_csv.")


def setup_standard_datasets(
    data_dir: str = "data",
    test_seed: int = 42,
    pool_seed: int = 2024,
    n_test: int = 100,
    n_pool: int = 5000
) -> None:
    """
    Tạo sẵn 2 file dữ liệu chuẩn cho cả nhóm:
    1. data/test_dataset_100.csv (đúng 100 mẫu, seed_data_gen cố định 42).
    2. data/train_val_pool.csv (5000 mẫu, seed_data_gen cố định 2024) để các thực nghiệm trích xuất.
    """
    # TODO (Thành viên 1):
    # Sinh và lưu test_dataset_100.csv và train_val_pool.csv.
    raise NotImplementedError("TODO (Thành viên 1): Hãy triển khai hàm setup_standard_datasets.")


def get_train_val_split(
    pool_X: np.ndarray,
    pool_y: np.ndarray,
    n_train: int,
    n_val: int = 400
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Hàm tiện ích chuẩn hóa phân chia tập dữ liệu từ train_val_pool theo quy tắc chỉ số cố định:
    - Tập Validation (n_val mẫu đầu tiên): pool_X[0:n_val], pool_y[0:n_val]
    - Tập Train (n_train mẫu tiếp theo): pool_X[n_val : n_val + n_train], pool_y[n_val : n_val + n_train]
    Bảo đảm:
      1. Train và Validation tuyệt đối không có mẫu trùng lặp (Zero Data Leakage).
      2. Với Thực nghiệm A: n_train tăng dần tạo thành tập con lồng nhau nghiêm ngặt:
         D_200 ⊂ D_400 ⊂ ... ⊂ D_2000.
      3. Với Thực nghiệm B & C: cố định n_train = 1500 (lấy từ chỉ số 400 đến 1900).
    """
    # TODO (Thành viên 1):
    # Trích xuất và trả về (X_train, y_train, X_val, y_val).
    if n_val + n_train > len(pool_X):
        raise ValueError(f"Tổng mẫu yêu cầu ({n_val} val + {n_train} train) vượt quá kích thước pool ({len(pool_X)}).")
    X_val = pool_X[:n_val]
    y_val = pool_y[:n_val]
    X_train = pool_X[n_val : n_val + n_train]
    y_train = pool_y[n_val : n_val + n_train]
    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    print("--- KIỂM TRA MODULE DATA GENERATOR (THÀNH VIÊN 1) ---")

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

    x = np.asarray(x, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
        raise ValueError("x phải có kích thước (N, 2) và chứa các số hữu hạn.")
    if mu.shape != (2,) or not np.isfinite(mu).all():
        raise ValueError("mu phải có kích thước (2,) và chứa các số hữu hạn.")
    if cov.shape != (2, 2) or not np.isfinite(cov).all() or not np.allclose(cov, cov.T):
        raise ValueError("cov phải là ma trận đối xứng (2, 2) chứa các số hữu hạn.")
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("cov phải là ma trận xác định dương.") from exc
    delta = x - mu
    normalized = np.linalg.solve(chol, delta.T)
    squared_distance = np.sum(normalized ** 2, axis=0)
    coefficient = 1.0 / (2.0 * np.pi * np.prod(np.diag(chol)))
    return (coefficient * np.exp(-0.5 * squared_distance)).reshape(-1, 1)


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
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError("n_samples phải là số nguyên dương.")
    for bounds in (x1_range, x2_range):
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError("Mỗi khoảng phải gồm hai số hữu hạn với cận dưới nhỏ hơn cận trên.")
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(*x1_range, size=n_samples)
    x2 = rng.uniform(*x2_range, size=n_samples)
    X = np.column_stack((x1, x2))
    return X, gaussian_2d_pdf(X, mu, cov)


def save_dataset_to_csv(X: np.ndarray, y: np.ndarray, file_path: str) -> None:
    """
    Lưu dữ liệu X (2 cột) và y (1 cột) vào file CSV với header [x1, x2, y].
    """
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or X.shape[1] != 2 or y.shape != (len(X), 1):
        raise ValueError("X phải có shape (N, 2) và y phải có shape (N, 1).")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Dữ liệu phải chứa các số hữu hạn.")
    parent = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(parent, exist_ok=True)
    pd.DataFrame(np.column_stack((X, y)), columns=["x1", "x2", "y"]).to_csv(file_path, index=False)


def load_dataset_from_csv(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Đọc dữ liệu từ file CSV, trả về X có shape (N, 2) và y có shape (N, 1).
    """
    df = pd.read_csv(file_path)
    if list(df.columns) != ["x1", "x2", "y"]:
        raise ValueError("File CSV phải có các cột x1, x2, y theo đúng thứ tự.")
    values = df.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Dữ liệu CSV phải chứa các số hữu hạn.")
    return values[:, :2], values[:, 2:3]


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
    X_test, y_test = generate_samples(n_test, seed=test_seed)
    X_pool, y_pool = generate_samples(n_pool, seed=pool_seed)
    save_dataset_to_csv(X_test, y_test, os.path.join(data_dir, "test_dataset_100.csv"))
    save_dataset_to_csv(X_pool, y_pool, os.path.join(data_dir, "train_val_pool.csv"))


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
    pool_X, pool_y = np.asarray(pool_X), np.asarray(pool_y)
    if pool_X.ndim != 2 or pool_X.shape[1] != 2 or pool_y.shape != (len(pool_X), 1):
        raise ValueError("pool_X phải có shape (N, 2) và pool_y phải có shape (N, 1).")
    for name, count in (("n_train", n_train), ("n_val", n_val)):
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count <= 0:
            raise ValueError(f"{name} phải là số nguyên dương.")
    if n_val + n_train > len(pool_X):
        raise ValueError(f"Tổng mẫu yêu cầu ({n_val} val + {n_train} train) vượt quá kích thước pool ({len(pool_X)}).")
    X_val = pool_X[:n_val]
    y_val = pool_y[:n_val]
    X_train = pool_X[n_val : n_val + n_train]
    y_train = pool_y[n_val : n_val + n_train]
    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    setup_standard_datasets()
    print("Đã tạo data/test_dataset_100.csv và data/train_val_pool.csv.")

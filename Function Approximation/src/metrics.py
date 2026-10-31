"""
================================================================================
MODULE: metrics.py
PHỤ TRÁCH: THÀNH VIÊN 2 (Training Pipeline & Exp A)
MÔ TẢ NHIỆM VỤ:
    1. Cài đặt các hàm đo lường sai số hồi quy:
        - MSE (Mean Squared Error) - chỉ số chính theo yêu cầu đề bài.
        - MAE (Mean Absolute Error) - chỉ số bổ sung bắt buộc.
        - RMSE (Root Mean Squared Error) - chỉ số bổ sung bắt buộc.
        - R2 Score (Hệ số xác định).
    2. Hỗ trợ nhận đầu vào linh hoạt là numpy.ndarray hoặc torch.Tensor.
================================================================================
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
from typing import Union, Dict


def _to_numpy(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Chuyển đổi input về dạng mảng numpy 1 chiều (flatten) để tính toán thống nhất."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr, dtype=np.float64).flatten()


def mean_squared_error(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """Tính sai số toàn phương trung bình: MSE = (1/N) * sum((y_true - y_pred)^2)"""
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"Độ dài y_true ({len(yt)}) và y_pred ({len(yp)}) không khớp.")
    return float(np.mean((yt - yp) ** 2))


def mean_absolute_error(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """Tính sai số tuyệt đối trung bình: MAE = (1/N) * sum(|y_true - y_pred|)"""
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"Độ dài y_true ({len(yt)}) và y_pred ({len(yp)}) không khớp.")
    return float(np.mean(np.abs(yt - yp)))


def root_mean_squared_error(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """Tính căn bậc hai của sai số toàn phương trung bình: RMSE = sqrt(MSE)"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """Tính hệ số xác định: R2 = 1 - (SS_res / SS_tot)"""
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"Độ dài y_true ({len(yt)}) và y_pred ({len(yp)}) không khớp.")
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - (ss_res / ss_tot))


def compute_all_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> Dict[str, float]:
    """Tính toàn bộ các chỉ số: {'mse': ..., 'mae': ..., 'rmse': ..., 'r2': ...}"""
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }


if __name__ == "__main__":
    print("--- KIỂM TRA MODULE METRICS (THÀNH VIÊN 2) ---")
    y_t = np.array([1.0, 2.0, 3.0, 4.0])
    y_p = np.array([1.1, 1.9, 3.2, 3.8])
    metrics = compute_all_metrics(y_t, y_p)
    print("Kết quả tính thử metrics:", metrics)
    assert abs(metrics["mse"] - 0.025) < 1e-6, "MSE tính chưa chính xác!"
    print("✅ Kiểm tra module metrics thành công!")

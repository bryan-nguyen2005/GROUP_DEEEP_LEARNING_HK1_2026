"""
================================================================================
KHUNG SƯỜN MÃ NGUỒN (SKELETON TEMPLATE)
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
    # TODO (Thành viên 2):
    # Dùng _to_numpy và tính np.mean((yt - yp) ** 2).
    raise NotImplementedError("TODO (Thành viên 2): Hãy triển khai hàm mean_squared_error.")


def mean_absolute_error(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """Tính sai số tuyệt đối trung bình: MAE = (1/N) * sum(|y_true - y_pred|)"""
    # TODO (Thành viên 2):
    # Dùng _to_numpy và tính np.mean(np.abs(yt - yp)).
    raise NotImplementedError("TODO (Thành viên 2): Hãy triển khai hàm mean_absolute_error.")


def root_mean_squared_error(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """Tính căn bậc hai của sai số toàn phương trung bình: RMSE = sqrt(MSE)"""
    # TODO (Thành viên 2):
    # Lấy np.sqrt của hàm mean_squared_error.
    raise NotImplementedError("TODO (Thành viên 2): Hãy triển khai hàm root_mean_squared_error.")


def r2_score(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """Tính hệ số xác định: R2 = 1 - (SS_res / SS_tot)"""
    # TODO (Thành viên 2):
    # Tính SS_res và SS_tot, trả về 1 - SS_res / SS_tot.
    raise NotImplementedError("TODO (Thành viên 2): Hãy triển khai hàm r2_score.")


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

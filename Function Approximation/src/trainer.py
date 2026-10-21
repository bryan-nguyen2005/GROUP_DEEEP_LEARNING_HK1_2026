"""
================================================================================
KHUNG SƯỜN MÃ NGUỒN (SKELETON TEMPLATE)
MODULE: trainer.py
PHỤ TRÁCH: THÀNH VIÊN 2 (Training Pipeline & Exp A)
MÔ TẢ NHIỆM VỤ:
    1. Cài đặt hàm đóng gói numpy array thành PyTorch DataLoader (dùng TensorDataset).
    2. Cài đặt vòng lặp huấn luyện 1 epoch: forward -> loss -> zero_grad -> backward -> step.
    3. Cài đặt hàm đánh giá mô hình với model.eval() và torch.no_grad().
    4. Cài đặt lớp EarlyStopping để tự động lưu checkpoint trọng số model tốt nhất theo val loss.
    5. Cài đặt pipeline huấn luyện hoàn chỉnh ghi nhận log theo từng epoch ra DataFrame/CSV:
       - Trả về Dict chuẩn: model, best_epoch, best_val_loss, total_epochs, history, checkpoint_path.
       - Hỗ trợ tham số history_csv_path để lưu lịch sử loss theo epoch cho TẤT CẢ các runs.
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

import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Any, Optional

from src.metrics import compute_all_metrics


class EarlyStopping:
    """
    Cơ chế dừng sớm (Early Stopping) khi validation loss không cải thiện sau 'patience' epochs.
    Đồng thời tự động lưu checkpoint trọng số mô hình tốt nhất (.pth).
    """
    def __init__(self, patience: int = 15, min_delta: float = 1e-6, checkpoint_path: str = "checkpoints/best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_epoch = 0
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """
        Kiểm tra val_loss của epoch hiện tại.
        Nếu tốt hơn kỷ lục cũ -> lưu checkpoint mô hình qua torch.save(model.state_dict(), path).
        Nếu không cải thiện -> tăng counter; nếu counter >= patience -> đặt self.early_stop = True.
        """
        # TODO (Thành viên 2):
        # Triển khai logic so sánh val_loss với self.best_loss - self.min_delta.
        # Lưu checkpoint khi đạt loss tốt nhất và reset counter = 0.
        raise NotImplementedError("TODO (Thành viên 2): Hãy hoàn thiện lớp EarlyStopping.")


def create_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """Chuyển đổi dữ liệu X, y từ numpy array thành PyTorch DataLoader."""
    # TODO (Thành viên 2):
    # TensorDataset(torch.tensor(X, float32), torch.tensor(y, float32)) -> DataLoader
    raise NotImplementedError("TODO (Thành viên 2): Hãy triển khai hàm create_dataloader.")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """
    Huấn luyện mô hình trong 1 epoch:
    model.train() -> zero_grad() -> forward -> loss -> backward -> optimizer.step().
    """
    # TODO (Thành viên 2):
    # Triển khai vòng lặp huấn luyện chuẩn PyTorch.
    raise NotImplementedError("TODO (Thành viên 2): Hãy triển khai hàm train_one_epoch.")


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Đánh giá mô hình trên tập validation hoặc test với model.eval() và with torch.no_grad().
    Trả về: (avg_loss, y_true_all, y_pred_all)
    """
    # TODO (Thành viên 2):
    # Triển khai vòng lặp đánh giá không cập nhật gradient.
    raise NotImplementedError("TODO (Thành viên 2): Hãy triển khai hàm evaluate_model.")


def train_mlp_pipeline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 150,
    lr: float = 0.005,
    optimizer_name: str = "adam",
    checkpoint_path: str = "checkpoints/best_model.pth",
    history_csv_path: Optional[str] = None,
    patience: int = 15,
    device: Optional[torch.device] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Pipeline huấn luyện hoàn chỉnh một mô hình MLP:
    - Optimizer Adam (lr=0.005), Criterion nn.MSELoss().
    - Vòng lặp epochs: train_one_epoch, evaluate_model trên val_loader, EarlyStopping, checkpoint.
    - Ghi nhận lịch sử từng epoch: ['epoch', 'train_loss', 'val_loss', 'epoch_time_sec'].
    - Nếu history_csv_path is not None: lưu DataFrame history ra CSV để phục vụ phân tích/hội tụ.
    - Nạp lại trọng số tốt nhất từ checkpoint_path trước khi trả về.

    Trả về Dict[str, Any] gồm các trường chuẩn hóa:
        - "model": mô hình nạp lại trọng số tốt nhất
        - "best_epoch": số epoch đạt val_loss thấp nhất
        - "best_val_loss": giá trị val_loss thấp nhất
        - "total_epochs": tổng số epoch đã chạy trước khi dừng
        - "history": pd.DataFrame lịch sử loss qua các epoch
        - "checkpoint_path": đường dẫn checkpoint đã lưu
    """
    # TODO (Thành viên 2):
    # Triển khai pipeline huấn luyện theo các bước trên.
    raise NotImplementedError("TODO (Thành viên 2): Hãy triển khai hàm train_mlp_pipeline.")


if __name__ == "__main__":
    print("--- KIỂM TRA MODULE TRAINER (THÀNH VIÊN 2) ---")

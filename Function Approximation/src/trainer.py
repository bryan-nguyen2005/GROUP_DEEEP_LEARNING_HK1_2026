"""
================================================================================
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
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def create_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """Chuyển đổi dữ liệu X, y từ numpy array thành PyTorch DataLoader."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
    model.train()
    running_loss = 0.0
    total_samples = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


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
    model.eval()
    running_loss = 0.0
    total_samples = 0
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            y_true_list.append(targets.cpu().numpy())
            y_pred_list.append(outputs.cpu().numpy())

    avg_loss = running_loss / max(total_samples, 1)
    y_true_all = np.vstack(y_true_list) if y_true_list else np.empty((0, 1))
    y_pred_all = np.vstack(y_pred_list) if y_pred_list else np.empty((0, 1))
    return avg_loss, y_true_all, y_pred_all


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
    if device is None:
        device = torch.device("cpu")
    model.to(device)

    criterion = nn.MSELoss()
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Optimizer {optimizer_name} không được hỗ trợ.")

    early_stopping = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path)

    history_records = []
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate_model(model, val_loader, criterion, device)
        epoch_time = time.perf_counter() - t0

        history_records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time_sec": epoch_time
        })

        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            print(f"Epoch [{epoch:03d}/{epochs:03d}] - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Time: {epoch_time:.3f}s")

        if early_stopping(val_loss, model, epoch):
            if verbose:
                print(f"Early stopping kích hoạt tại epoch {epoch}. Kỷ lục tốt nhất tại epoch {early_stopping.best_epoch}.")
            break

    history_df = pd.DataFrame(history_records)

    if history_csv_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(history_csv_path)), exist_ok=True)
        history_df.to_csv(history_csv_path, index=False)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    return {
        "model": model,
        "best_epoch": early_stopping.best_epoch,
        "best_val_loss": early_stopping.best_loss,
        "total_epochs": len(history_records),
        "history": history_df,
        "checkpoint_path": checkpoint_path
    }


if __name__ == "__main__":
    print("--- KIỂM TRA MODULE TRAINER (THÀNH VIÊN 2) ---")
    from src.models import GaussianMLP
    dummy_x = np.random.randn(100, 2).astype(np.float32)
    dummy_y = np.random.randn(100, 1).astype(np.float32)

    tr_loader = create_dataloader(dummy_x[:70], dummy_y[:70], batch_size=16)
    va_loader = create_dataloader(dummy_x[70:], dummy_y[70:], batch_size=16, shuffle=False)

    test_model = GaussianMLP(input_dim=2, output_dim=1, num_hidden_layers=1, hidden_units=16)
    ckpt_test = "checkpoints/test_checkpoint.pth"
    hist_test = "logs/test_history.csv"

    res = train_mlp_pipeline(
        test_model, tr_loader, va_loader, epochs=5, lr=0.01,
        checkpoint_path=ckpt_test, history_csv_path=hist_test, verbose=True
    )
    print("Kết quả train thử nghiệm:")
    print(f"- Best epoch: {res['best_epoch']}, Best val loss: {res['best_val_loss']:.6f}")
    assert os.path.exists(ckpt_test), "Checkpoint chưa được tạo!"
    assert os.path.exists(hist_test), "History CSV chưa được tạo!"
    # Dọn dẹp file test
    if os.path.exists(ckpt_test): os.remove(ckpt_test)
    if os.path.exists(hist_test): os.remove(hist_test)
    print("✅ Kiểm tra module trainer thành công!")

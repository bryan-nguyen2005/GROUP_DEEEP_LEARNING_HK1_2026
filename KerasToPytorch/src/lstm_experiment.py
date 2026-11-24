"""Thực nghiệm Bài 5: CNN + TimeDistributed + LSTM tính tổng chuỗi ảnh số MNIST.

Tác giả: TV3 - Nguyen Hoang Long (MSSV: 3123411179)
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import time
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms.functional import convert_image_dtype

from .models.mnist_lstm import MNISTSequenceLSTM
from .pytorch_core import EarlyStopping, evaluate, fit, get_device, seed_everything


KERAS_REFERENCE_MAE = 0.6502


class MNISTSequenceDataset(Dataset):
    """Tập dữ liệu chuỗi T ảnh MNIST và nhãn là tổng giá trị các chữ số."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, num_sequences: int, seq_len: int = 7, seed: int = 42):
        self.seq_len = seq_len
        self.num_sequences = num_sequences
        rng = np.random.default_rng(seed)
        n_total = len(images)

        # Tạo ma trận chỉ số (num_sequences, seq_len)
        indices = rng.integers(0, n_total, size=(num_sequences, seq_len))
        self.sequences = images[indices]  # (num_sequences, seq_len, 1, 28, 28)
        self.targets = labels[indices].sum(dim=1, keepdim=True).float()  # (num_sequences, 1)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


def make_sequence_loaders(
    data_root: str = "data",
    num_train: int = 15000,
    num_val: int = 2500,
    num_test: int = 10000,
    seq_len: int = 7,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Tải MNIST và đóng gói thành chuỗi 7 ảnh cho Train, Val, và Test."""
    root = Path(data_root)
    raw_train = MNIST(root, train=True, download=True)
    raw_test = MNIST(root, train=False, download=True)

    train_imgs = convert_image_dtype(raw_train.data.unsqueeze(1), torch.float32)
    train_lbls = raw_train.targets.long()

    test_imgs = convert_image_dtype(raw_test.data.unsqueeze(1), torch.float32)
    test_lbls = raw_test.targets.long()

    # Phân chia 50.000 cho train pool, 10.000 cho val pool
    train_pool_imgs = train_imgs[:50000]
    train_pool_lbls = train_lbls[:50000]

    val_pool_imgs = train_imgs[50000:]
    val_pool_lbls = train_lbls[50000:]

    train_ds = MNISTSequenceDataset(train_pool_imgs, train_pool_lbls, num_train, seq_len, seed=seed)
    val_ds = MNISTSequenceDataset(val_pool_imgs, val_pool_lbls, num_val, seq_len, seed=seed + 1)
    # Test dataset chuẩn 10.000 chuỗi độc lập như Slide 50
    test_ds = MNISTSequenceDataset(test_imgs, test_lbls, num_test, seq_len, seed=seed + 2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def plot_lstm_results(history: list[dict], test_samples: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], figures_dir: Path):
    """Vẽ đường cong học tập MAE và lưới dự đoán mẫu."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

    # 1. Đường cong MAE
    epochs = [h["epoch"] for h in history]
    tr_mae = [h["train_mae"] for h in history]
    va_mae = [h["val_mae"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
    ax1.plot(epochs, tr_mae, label="Train MAE", color="#1f6feb", linewidth=2.0)
    ax1.plot(epochs, va_mae, label="Val MAE", color="#2ea043", linewidth=2.0)
    ax1.axhline(KERAS_REFERENCE_MAE, color="#cf222e", linestyle="--", linewidth=1.5, label=f"Keras Test MAE ({KERAS_REFERENCE_MAE})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean Absolute Error (MAE)")
    ax1.set_title("CNN + LSTM: MAE Convergence Curve (MNIST Summation)")
    ax1.legend()

    tr_loss = [h["train_loss"] for h in history]
    va_loss = [h["val_loss"] for h in history]
    ax2.plot(epochs, tr_loss, label="Train L1 Loss", color="#1f6feb", linewidth=2.0)
    ax2.plot(epochs, va_loss, label="Val L1 Loss", color="#2ea043", linewidth=2.0)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("L1 Loss")
    ax2.set_title("L1 Loss Trajectory")
    ax2.legend()

    curve_path = figures_dir / "05_mnist_lstm_mae_curve.png"
    fig.savefig(curve_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 2. Lưới dự đoán 8 chuỗi mẫu
    sample_seqs, sample_true, sample_pred = test_samples
    fig_samples, axes = plt.subplots(8, 7, figsize=(11.5, 11.0))
    fig_samples.suptitle(
        "Bài 5: Dự đoán tổng chuỗi 7 ảnh MNIST bằng TimeDistributed(CNN) + LSTM\n"
        "(Mỗi hàng là một chuỗi 7 chữ số độc lập)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )

    for i in range(8):
        t_val = sample_true[i].item()
        p_val = sample_pred[i].item()
        err = abs(p_val - t_val)
        color = "#1a7f37" if err < 1.0 else "#cf222e"
        for t in range(7):
            ax = axes[i, t]
            img = sample_seqs[i, t, 0].cpu().numpy()
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            if t == 0:
                ax.set_title(f"Row {i+1}: True={t_val:.0f} | Pred={p_val:.1f} (Δ={err:.1f})", loc="left", fontsize=9.2, color=color, fontweight="bold")

    samples_path = figures_dir / "05_mnist_lstm_samples.png"
    fig_samples.savefig(samples_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_samples)


def run_experiment(
    data_root: str = "data",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    num_train: int = 15000,
    seed: int = 42,
    device_name: str = "auto",
):
    """Chạy trọn gói huấn luyện, đánh giá và lưu trữ cho Bài 5."""
    seed_everything(seed)
    device = torch.device("cuda" if (device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available())) else "cpu")
    print(f"Device: {device} | Seed: {seed} | Epochs: {epochs}")

    train_loader, val_loader, test_loader = make_sequence_loaders(
        data_root=data_root,
        num_train=num_train,
        num_val=2500,
        num_test=10000,
        seq_len=7,
        batch_size=batch_size,
        seed=seed,
    )

    model = MNISTSequenceLSTM(time_steps=7, lstm_units=128).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    criterion = nn.L1Loss()  # MAE Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=4, min_delta=0.01)

    t0 = time.time()
    history = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs=epochs,
        task="regression",
        early_stopping=early_stopping,
        scheduler=scheduler,
        csv_path="logs/history_05_mnist_lstm.csv",
        verbose=True,
    )
    fit_time = time.time() - t0

    # Đánh giá trên 10.000 chuỗi test
    test_metrics = evaluate(model, test_loader, criterion, device, task="regression")
    test_mae = test_metrics["mae"]
    test_loss = test_metrics["loss"]

    delta_mae = test_mae - KERAS_REFERENCE_MAE
    print(f"\n--- KẾT QUẢ KIỂM THỬ BÀI 5 (10.000 chuỗi test) ---")
    print(f"PyTorch Test MAE : {test_mae:.4f}")
    print(f"Keras Test MAE   : {KERAS_REFERENCE_MAE:.4f}")
    print(f"Chênh lệch (Δ)   : {delta_mae:+.4f} ({'Tốt hơn' if delta_mae < 0 else 'Tương đương'} Keras)")
    print(f"Thời gian fit    : {fit_time:.2f}s")

    # Lưu checkpoint
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "05_mnist_lstm.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "test_mae": test_mae,
            "test_loss": test_loss,
            "epochs": len(history),
            "parameters": total_params,
        },
        ckpt_path,
    )

    # Lưu result CSV
    result = {
        "task": "mnist_lstm_sum",
        "epochs": len(history),
        "parameters": total_params,
        "test_loss": test_loss,
        "test_mae": test_mae,
        "keras_reference_mae": KERAS_REFERENCE_MAE,
        "mae_delta": delta_mae,
        "fit_seconds": fit_time,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    with Path("logs/result_05_mnist_lstm.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)

    # Lấy mẫu để vẽ hình demo
    model.eval()
    with torch.no_grad():
        sample_batch, sample_targets = next(iter(test_loader))
        sample_preds = model(sample_batch.to(device)).cpu()
    plot_lstm_results(history, (sample_batch[:8], sample_targets[:8], sample_preds[:8]), Path("figures"))

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm Bài 5 CNN + LSTM")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-train", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_experiment(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_train=args.num_train,
        seed=args.seed,
        device_name=args.device,
    )

"""
================================================================================
KHUNG SƯỜN MÃ NGUỒN (SKELETON TEMPLATE)
SCRIPT: run_experiment_B.py
PHỤ TRÁCH: THÀNH VIÊN 3 (Experiments B & C Lead)
MÔ TẢ NHIỆM VỤ:
    - Thực hiện Thực nghiệm B: Khảo sát ảnh hưởng của Độ rộng tầng ẩn (B - Hidden Units).
    - Cấu hình mạng: MLP 1 tầng ẩn, số nơ-ron B in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64].
    - NGUYÊN TẮC CÔNG BẰNG & CHỐNG RÒ RỈ DỮ LIỆU (Zero Data Leakage):
        + Cố định tập Train (N = 1500 mẫu, chỉ số iloc[400:1900] của train_val_pool.csv).
        + Cố định tập Validation (N = 400 mẫu, chỉ số iloc[0:400] của train_val_pool.csv).
        + Tập Test: 100 mẫu độc lập từ data/test_dataset_100.csv (chỉ dùng đo ngoại kiểm sau huấn luyện).
        + Cố định mọi siêu tham số baseline khác (Adam lr=0.005, batch 32, epochs 150, activation='relu').
        + Biến duy nhất thay đổi là B.
    - Giao thức Đa-seed:
        + Chạy 3 seeds cố định: [42, 100, 2024] cho khởi tạo trọng số và DataLoader.
    - Xuất dữ liệu chuẩn Schema đầy đủ tham số độc lập (Self-contained):
        + logs/raw_runs_exp_B.csv: [run_id, exp_name, seed, param_name, param_value, num_layers, hidden_units,
          activation, lr, batch_size, max_epochs, patience, n_train, n_val, total_epochs, best_epoch,
          train_mse, val_mse, test_mse, train_time_sec, checkpoint_file, history_file]
        + logs/exp_B_results.csv: [param_value, train_mse_mean, train_mse_std, val_mse_mean, val_mse_std,
          test_mse_mean, test_mse_std, best_epoch_mean, train_time_mean]
        + Lưu file history từng run: logs/history_exp_B_width_{B}_seed{seed}.csv
        + figures/exp_B_width_vs_error.png (gọi hàm plot_experiment_trend do TV4 bàn giao).
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
import numpy as np
import pandas as pd
import torch

from src.data_generator import load_dataset_from_csv, get_train_val_split
from src.models import GaussianMLP
from src.trainer import create_dataloader, train_mlp_pipeline, evaluate_model
from src.metrics import compute_all_metrics
from src.visualize import plot_experiment_trend


def run_experiment_B(
    data_dir: str = "data",
    logs_dir: str = "logs",
    figures_dir: str = "figures",
    checkpoints_dir: str = "checkpoints",
    hidden_units_list: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64],
    seeds: list = [42, 100, 2024],
    fixed_train_samples: int = 1500,
    device: torch.device = torch.device("cpu")
):
    """
    Quy trình thực hiện Thực nghiệm B chuẩn hóa.
    """
    print("================================================================================")
    print("BẮT ĐẦU THỰC NGHIỆM B: KHẢO SÁT ĐỘ RỘNG TẦNG ẨN (B)")
    print(f"Danh sách độ rộng B: {hidden_units_list}")
    print(f"Mẫu train cố định: {fixed_train_samples} | Seeds: {seeds} | Thiết bị: {device}")
    print("================================================================================")

    # TODO (Thành viên 3):
    # Bước 1: Nạp tập Test 100 mẫu từ data/test_dataset_100.csv.
    #         Nạp tập Pool từ data/train_val_pool.csv và dùng get_train_val_split
    #         để lấy đúng X_train (1500 mẫu), y_train, X_val (400 mẫu), y_val.
    # Bước 2: Duyệt qua từng số nơ-ron B trong hidden_units_list:
    #         - Với mỗi seed trong seeds:
    #             + torch.manual_seed(seed), np.random.seed(seed).
    #             + Khởi tạo GaussianMLP(num_hidden_layers=1, hidden_units=B, activation='relu').
    #             + checkpoint_path = checkpoints/exp_B_width_{B}_seed{seed}.pth.
    #             + history_csv_path = logs/history_exp_B_width_{B}_seed{seed}.csv.
    #             + Huấn luyện bằng train_mlp_pipeline (sử dụng EarlyStopping theo Validation Loss).
    #             + Đánh giá Train MSE, Validation MSE, và Test MSE (unseen evaluation).
    #             + Ghi nhận đầy đủ siêu tham số và kết quả vào danh sách raw runs.
    #         - Tính Mean +- Std của Train/Val/Test metrics cho từng giá trị B.
    # Bước 3: Lưu nhật ký chi tiết từng run vào logs/raw_runs_exp_B.csv theo chuẩn schema tự chứa.
    #         Lưu bảng tổng hợp ra logs/exp_B_results.csv theo chuẩn schema.
    # Bước 4: Gọi plot_experiment_trend vẽ biểu đồ sai số theo B (lưu vào figures/exp_B_width_vs_error.png).
    raise NotImplementedError("TODO (Thành viên 3): Hãy hoàn thiện script Thực nghiệm B theo các bước trên.")


if __name__ == "__main__":
    run_experiment_B()

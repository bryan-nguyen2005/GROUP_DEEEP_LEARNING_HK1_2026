"""
================================================================================
KHUNG SƯỜN MÃ NGUỒN (SKELETON TEMPLATE)
SCRIPT: run_experiment_A.py
PHỤ TRÁCH: THÀNH VIÊN 2 (Training Pipeline & Exp A)
MÔ TẢ NHIỆM VỤ:
    - Thực hiện Thực nghiệm A: Khảo sát ảnh hưởng của Quy mô dữ liệu huấn luyện (A).
    - Các mốc giá trị của A: [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000].
    - NGUYÊN TẮC CÔNG BẰNG & CHỐNG RÒ RỈ DỮ LIỆU (Zero Data Leakage):
        + Tập Validation: Đúng 400 mẫu đầu tiên từ data/train_val_pool.csv (iloc[0:400]) cố định.
        + Tập con lồng nhau (Nested Subsets): Trích xuất theo chỉ số iloc[400 : 400+A] từ train_val_pool.csv
          bảo đảm D_200 ⊂ D_400 ⊂ ... ⊂ D_2000 và tuyệt đối không trùng với tập Validation.
        + Tập Test: 100 mẫu độc lập từ data/test_dataset_100.csv chỉ dùng đo ngoại kiểm sau khi đóng băng trọng số.
    - Cấu hình Baseline chuẩn cố định:
        + Kiến trúc: MLP 1 tầng ẩn, 64 nơ-ron, activation='relu'.
        + Optimizer: Adam, Learning Rate = 0.005, Batch Size = 32.
        + Số Epochs = 150, Early Stopping patience = 15 theo Validation Loss.
        + Thiết bị: CPU (torch.device('cpu')).
    - Giao thức Đa-seed:
        + Chạy 3 seeds cố định: [42, 100, 2024] cho khởi tạo trọng số và DataLoader.
    - Xuất dữ liệu chuẩn Schema đầy đủ tham số độc lập (Self-contained):
        + logs/raw_runs_exp_A.csv: [run_id, exp_name, seed, param_name, param_value, num_layers, hidden_units,
          activation, lr, batch_size, max_epochs, patience, n_train, n_val, total_epochs, best_epoch,
          train_mse, val_mse, test_mse, train_time_sec, checkpoint_file, history_file]
        + logs/exp_A_results.csv: [param_value, train_mse_mean, train_mse_std, val_mse_mean, val_mse_std,
          test_mse_mean, test_mse_std, best_epoch_mean, train_time_mean]
        + Lưu file history từng run: logs/history_exp_A_samples_{A}_seed{seed}.csv
        + figures/exp_A_samples_vs_error.png (gọi hàm plot_experiment_trend do TV4 bàn giao).
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


def run_experiment_A(
    data_dir: str = "data",
    logs_dir: str = "logs",
    figures_dir: str = "figures",
    checkpoints_dir: str = "checkpoints",
    sample_sizes: list = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    seeds: list = [42, 100, 2024],
    device: torch.device = torch.device("cpu")
):
    """
    Quy trình thực hiện Thực nghiệm A chuẩn hóa.
    """
    print("================================================================================")
    print("BẮT ĐẦU THỰC NGHIỆM A: KHẢO SÁT QUY MÔ DỮ LIỆU HUẤN LUYỆN (A)")
    print(f"Mốc số mẫu A: {sample_sizes}")
    print(f"Seeds: {seeds} | Thiết bị: {device}")
    print("================================================================================")

    # TODO (Thành viên 2):
    # Bước 1: Nạp tập Test 100 mẫu từ data/test_dataset_100.csv.
    #         Nạp tập Pool từ data/train_val_pool.csv.
    #         Dùng hàm get_train_val_split(pool_X, pool_y, n_train=max(sample_sizes), n_val=400)
    #         để có tập Validation cố định (400 mẫu đầu) và toàn bộ Train Pool (từ chỉ số 400).
    # Bước 2: Với mỗi mốc A trong sample_sizes:
    #         Lấy tập con lồng nhau: X_train_A = X_train_pool[:A], y_train_A = y_train_pool[:A].
    #         Với mỗi seed trong seeds:
    #             - torch.manual_seed(seed), np.random.seed(seed).
    #             - Tạo dataloaders cho train_loader (batch 32, shuffle=True), val_loader, test_loader.
    #             - Khởi tạo GaussianMLP(num_hidden_layers=1, hidden_units=64, activation='relu').
    #             - Định nghĩa checkpoint_path: checkpoints/exp_A_samples_{A}_seed{seed}.pth.
    #             - Định nghĩa history_csv_path: logs/history_exp_A_samples_{A}_seed{seed}.csv.
    #             - Huấn luyện qua train_mlp_pipeline (đo thời gian, lưu checkpoint và history).
    #             - Đánh giá train_mse, val_mse, test_mse.
    #             - Ghi nhận đầy đủ thông số siêu tham số và kết quả vào danh sách raw runs.
    # Bước 3: Lưu nhật ký chi tiết vào logs/raw_runs_exp_A.csv theo chuẩn schema tự chứa (self-contained).
    #         Tính Mean +- Std qua 3 seeds và lưu bảng tổng hợp vào logs/exp_A_results.csv.
    # Bước 4: Gọi hàm plot_experiment_trend để xuất đồ thị figures/exp_A_samples_vs_error.png.
    raise NotImplementedError("TODO (Thành viên 2): Hãy hoàn thiện script Thực nghiệm A theo các bước trên.")


if __name__ == "__main__":
    run_experiment_A()

"""
================================================================================
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
        + logs/raw_runs_exp_A.csv
        + logs/exp_A_results.csv
        + logs/history_exp_A_samples_{A}_seed{seed}.csv
        + figures/exp_A_samples_vs_error.png
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
import torch.nn as nn

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

    # Đảm bảo các thư mục đích tồn tại
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    test_path = os.path.join(data_dir, "test_dataset_100.csv")
    pool_path = os.path.join(data_dir, "train_val_pool.csv")

    if not os.path.exists(test_path) or not os.path.exists(pool_path):
        raise FileNotFoundError(f"Không tìm thấy tập dữ liệu trong '{data_dir}'. Vui lòng kiểm tra!")

    # Bước 1: Nạp tập Test 100 mẫu và tập Pool 5000 mẫu
    X_test, y_test = load_dataset_from_csv(test_path)
    pool_X, pool_y = load_dataset_from_csv(pool_path)

    # Phân chia: 400 mẫu validation cố định và train pool 4600 mẫu
    max_a = max(sample_sizes)
    X_train_pool_all, y_train_pool_all, X_val, y_val = get_train_val_split(
        pool_X, pool_y, n_train=max_a, n_val=400
    )

    # Tham số Baseline chuẩn
    num_layers = 1
    hidden_units = 64
    activation = "relu"
    lr = 0.005
    batch_size = 32
    max_epochs = 150
    patience = 15
    n_val = len(X_val)
    criterion = nn.MSELoss()

    raw_runs = []
    run_id = 1
    total_runs = len(sample_sizes) * len(seeds)

    # Bước 2: Lặp qua từng mốc A và từng seed
    for A in sample_sizes:
        # Tập con lồng nhau nghiêm ngặt: D_200 ⊂ D_400 ⊂ ... ⊂ D_2000
        X_train_A = X_train_pool_all[:A]
        y_train_A = y_train_pool_all[:A]

        for seed in seeds:
            print(f"\n[Run {run_id:02d}/{total_runs:02d}] Cấu hình A={A:4d} | Seed={seed} ...")

            # Thiết lập random seed
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Tạo DataLoader
            train_loader = create_dataloader(X_train_A, y_train_A, batch_size=batch_size, shuffle=True)
            val_loader = create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)
            test_loader = create_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

            # Khởi tạo mô hình MLP Baseline
            model = GaussianMLP(
                input_dim=2,
                output_dim=1,
                num_hidden_layers=num_layers,
                hidden_units=hidden_units,
                activation=activation
            )

            ckpt_filename = f"exp_A_samples_{A}_seed{seed}.pth"
            ckpt_path = os.path.join(checkpoints_dir, ckpt_filename)
            hist_filename = f"history_exp_A_samples_{A}_seed{seed}.csv"
            hist_path = os.path.join(logs_dir, hist_filename)

            # Huấn luyện mô hình
            t0 = time.perf_counter()
            train_res = train_mlp_pipeline(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=max_epochs,
                lr=lr,
                optimizer_name="adam",
                checkpoint_path=ckpt_path,
                history_csv_path=hist_path,
                patience=patience,
                device=device,
                verbose=False
            )
            train_time = time.perf_counter() - t0

            best_model = train_res["model"]
            best_epoch = train_res["best_epoch"]
            total_epochs = train_res["total_epochs"]

            # Đo lường MSE trên các tập
            train_mse, _, _ = evaluate_model(best_model, train_loader, criterion, device)
            val_mse, _, _ = evaluate_model(best_model, val_loader, criterion, device)
            test_mse, _, _ = evaluate_model(best_model, test_loader, criterion, device)

            print(f" -> Xong: Epoch dừng={total_epochs:2d} (Best={best_epoch:2d}) | Train MSE={train_mse:.6f} | Val MSE={val_mse:.6f} | Test MSE={test_mse:.6f} | Time={train_time:.2f}s")

            # Ghi nhận nhật ký đầy đủ tham số (self-contained)
            raw_runs.append({
                "run_id": run_id,
                "exp_name": "exp_A",
                "seed": seed,
                "param_name": "n_train",
                "param_value": A,
                "num_layers": num_layers,
                "hidden_units": hidden_units,
                "activation": activation,
                "lr": lr,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "patience": patience,
                "n_train": A,
                "n_val": n_val,
                "total_epochs": total_epochs,
                "best_epoch": best_epoch,
                "train_mse": float(train_mse),
                "val_mse": float(val_mse),
                "test_mse": float(test_mse),
                "train_time_sec": float(train_time),
                "checkpoint_file": ckpt_filename,
                "history_file": hist_filename
            })
            run_id += 1

    # Bước 3: Lưu nhật ký chi tiết và tính bảng tổng hợp
    raw_runs_df = pd.DataFrame(raw_runs)
    raw_runs_csv = os.path.join(logs_dir, "raw_runs_exp_A.csv")
    raw_runs_df.to_csv(raw_runs_csv, index=False)
    print(f"\n✅ Đã lưu nhật ký chi tiết {len(raw_runs_df)} lần chạy vào: {raw_runs_csv}")

    # Tính bảng tổng hợp Mean +- Std theo mốc A
    summary_rows = []
    for A in sample_sizes:
        sub = raw_runs_df[raw_runs_df["param_value"] == A]
        summary_rows.append({
            "param_value": A,
            "train_mse_mean": float(sub["train_mse"].mean()),
            "train_mse_std": float(sub["train_mse"].std(ddof=1) if len(sub) > 1 else 0.0),
            "val_mse_mean": float(sub["val_mse"].mean()),
            "val_mse_std": float(sub["val_mse"].std(ddof=1) if len(sub) > 1 else 0.0),
            "test_mse_mean": float(sub["test_mse"].mean()),
            "test_mse_std": float(sub["test_mse"].std(ddof=1) if len(sub) > 1 else 0.0),
            "best_epoch_mean": float(sub["best_epoch"].mean()),
            "train_time_mean": float(sub["train_time_sec"].mean())
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(logs_dir, "exp_A_results.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"✅ Đã lưu bảng kết quả tổng hợp vào: {summary_csv}")

    # In bảng tổng hợp ra console
    print("\n" + "=" * 80)
    print("BẢNG KẾT QUẢ TỔNG HỢP THỰC NGHIỆM A (MEAN ± STD QUA 3 SEEDS):")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Bước 4: Vẽ và lưu biểu đồ xu hướng
    fig_path = os.path.join(figures_dir, "exp_A_samples_vs_error.png")
    plot_experiment_trend(
        x_vals=summary_df["param_value"].tolist(),
        train_errors=summary_df["train_mse_mean"].tolist(),
        val_errors=summary_df["val_mse_mean"].tolist(),
        test_errors=summary_df["test_mse_mean"].tolist(),
        x_label="Số lượng mẫu huấn luyện A (Sample Size)",
        y_label="Sai số toàn phương trung bình (MSE)",
        title="Thực Nghiệm A: Ảnh Hưởng Của Quy Mô Dữ Liệu Huấn Luyện (Nested Subsets)",
        std_test=summary_df["test_mse_std"].tolist(),
        save_path=fig_path
    )
    print(f"✅ Đã tạo đồ thị sai số xuất bản vào: {fig_path}")
    print("================================================================================")
    print("HOÀN THÀNH TOÀN DIỆN THỰC NGHIỆM A!")
    print("================================================================================")


if __name__ == "__main__":
    run_experiment_A()

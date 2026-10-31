"""
================================================================================
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
        + logs/raw_runs_exp_B.csv
        + logs/exp_B_results.csv
        + logs/history_exp_B_width_{B}_seed{seed}.csv
        + figures/exp_B_width_vs_error.png
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

    # Phân chia: đúng 400 mẫu validation cố định (iloc[0:400]) và 1500 mẫu train cố định (iloc[400:1900])
    X_train, y_train, X_val, y_val = get_train_val_split(
        pool_X, pool_y, n_train=fixed_train_samples, n_val=400
    )

    # Tham số Baseline chuẩn
    num_layers = 1
    activation = "relu"
    lr = 0.005
    batch_size = 32
    max_epochs = 150
    patience = 15
    n_val = len(X_val)
    criterion = nn.MSELoss()

    raw_runs = []
    run_id = 1
    total_runs = len(hidden_units_list) * len(seeds)

    # Bước 2: Lặp qua từng số nơ-ron B và từng seed
    for B in hidden_units_list:
        for seed in seeds:
            print(f"\n[Run {run_id:02d}/{total_runs:02d}] Cấu hình B={B:2d} | Seed={seed} ...")

            # Thiết lập random seed
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Tạo DataLoader
            train_loader = create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
            val_loader = create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)
            test_loader = create_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

            # Khởi tạo mô hình MLP 1 tầng ẩn với B nơ-ron
            model = GaussianMLP(
                input_dim=2,
                output_dim=1,
                num_hidden_layers=num_layers,
                hidden_units=B,
                activation=activation
            )

            ckpt_filename = f"exp_B_width_{B}_seed{seed}.pth"
            ckpt_path = os.path.join(checkpoints_dir, ckpt_filename)
            hist_filename = f"history_exp_B_width_{B}_seed{seed}.csv"
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
                "exp_name": "exp_B",
                "seed": seed,
                "param_name": "hidden_units",
                "param_value": B,
                "num_layers": num_layers,
                "hidden_units": B,
                "activation": activation,
                "lr": lr,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "patience": patience,
                "n_train": fixed_train_samples,
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
    raw_runs_csv = os.path.join(logs_dir, "raw_runs_exp_B.csv")
    raw_runs_df.to_csv(raw_runs_csv, index=False)
    print(f"\n✅ Đã lưu nhật ký chi tiết {len(raw_runs_df)} lần chạy vào: {raw_runs_csv}")

    # Tính bảng tổng hợp Mean +- Std theo mốc B
    summary_rows = []
    for B in hidden_units_list:
        sub = raw_runs_df[raw_runs_df["param_value"] == B]
        summary_rows.append({
            "param_value": B,
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
    summary_csv = os.path.join(logs_dir, "exp_B_results.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"✅ Đã lưu bảng kết quả tổng hợp vào: {summary_csv}")

    # In bảng tổng hợp ra console
    print("\n" + "=" * 80)
    print("BẢNG KẾT QUẢ TỔNG HỢP THỰC NGHIỆM B (MEAN ± STD QUA 3 SEEDS):")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Bước 4: Vẽ và lưu biểu đồ xu hướng sai số theo B
    fig_path = os.path.join(figures_dir, "exp_B_width_vs_error.png")
    plot_experiment_trend(
        x_vals=summary_df["param_value"].tolist(),
        train_errors=summary_df["train_mse_mean"].tolist(),
        val_errors=summary_df["val_mse_mean"].tolist(),
        test_errors=summary_df["test_mse_mean"].tolist(),
        x_label="Độ rộng tầng ẩn B (Hidden Units)",
        y_label="Sai số toàn phương trung bình (MSE)",
        title="Thực Nghiệm B: Ảnh Hưởng Của Độ Rộng Tầng Ẩn (Hidden Units B)",
        std_test=summary_df["test_mse_std"].tolist(),
        save_path=fig_path
    )
    print(f"✅ Đã tạo đồ thị sai số xuất bản vào: {fig_path}")
    print("================================================================================")
    print("HOÀN THÀNH TOÀN DIỆN THỰC NGHIỆM B!")
    print("================================================================================")


if __name__ == "__main__":
    run_experiment_B()

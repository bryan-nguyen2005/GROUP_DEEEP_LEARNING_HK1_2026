"""
================================================================================
SCRIPT: run_experiment_C.py
PHỤ TRÁCH: THÀNH VIÊN 3 (Experiments B & C Lead)
MÔ TẢ NHIỆM VỤ:
    - Thực hiện Thực nghiệm C: Khảo sát ảnh hưởng của Độ sâu mạng MLP (C - Hidden Layers).
    - Cấu hình mạng: Số tầng ẩn C in [1, 2, 3, 4, 5], CỐ ĐỊNH 32 NƠ-RON TRÊN MỖI TẦNG ẨN.
    - NGUYÊN TẮC CÔNG BẰNG & CHỐNG RÒ RỈ DỮ LIỆU (Zero Data Leakage):
        + Cố định tập Train (N = 1500 mẫu, chỉ số iloc[400:1900] của train_val_pool.csv).
        + Cố định tập Validation (N = 400 mẫu, chỉ số iloc[0:400] của train_val_pool.csv).
        + Tập Test: 100 mẫu độc lập từ data/test_dataset_100.csv (chỉ dùng đo ngoại kiểm sau huấn luyện).
        + Cố định toàn bộ siêu tham số baseline khác (Adam lr=0.005, batch 32, epochs 150, ReLU).
        + Biến duy nhất thay đổi là số tầng ẩn C.
    - ĐẦU RA BẮT BUỘC THEO ĐỀ BÀI:
        + Bảng/đồ thị test error (MSE, MAE, RMSE).
        + Loss hội tụ và epoch tối ưu.
        + Thời gian huấn luyện (Training Time tính bằng giây).
    - Giao thức Đa-seed:
        + Chạy 3 seeds cố định: [42, 100, 2024] cho khởi tạo trọng số và DataLoader.
    - Xuất dữ liệu chuẩn Schema đầy đủ tham số độc lập (Self-contained):
        + logs/raw_runs_exp_C.csv: [run_id, exp_name, seed, param_name, param_value, num_layers, hidden_units,
          activation, lr, batch_size, max_epochs, patience, n_train, n_val, total_epochs, best_epoch,
          train_mse, val_mse, test_mse, train_time_sec, checkpoint_file, history_file]
        + logs/exp_C_results.csv: [param_value, train_mse_mean, train_mse_std, val_mse_mean, val_mse_std,
          test_mse_mean, test_mse_std, best_epoch_mean, train_time_mean]
        + Lưu file history từng run: logs/history_exp_C_depth_{C}_seed{seed}.csv
        + figures/exp_C_depth_vs_error.png: đồ thị sai số theo C (plot_experiment_trend).
        + figures/exp_C_depth_vs_time.png: biểu đồ cột thời gian huấn luyện theo C.
        + figures/exp_C_loss_convergence.png: đồ thị đường cong hội tụ loss theo epoch.
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
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.data_generator import load_dataset_from_csv, get_train_val_split
from src.models import GaussianMLP
from src.trainer import create_dataloader, train_mlp_pipeline, evaluate_model
from src.visualize import plot_experiment_trend


def run_experiment_C(
    data_dir: str = "data",
    logs_dir: str = "logs",
    figures_dir: str = "figures",
    checkpoints_dir: str = "checkpoints",
    depth_list: list = [1, 2, 3, 4, 5],
    units_per_layer: int = 32,
    seeds: list = [42, 100, 2024],
    fixed_train_samples: int = 1500,
    device: torch.device = torch.device("cpu")
):
    """
    Quy trình thực hiện Thực nghiệm C chuẩn hóa.
    """
    print("================================================================================")
    print("BẮT ĐẦU THỰC NGHIỆM C: KHẢO SÁT ĐỘ SÂU MẠNG (C - HIDDEN LAYERS)")
    print(f"Danh sách độ sâu C: {depth_list} | Cố định: {units_per_layer} nơ-ron/tầng")
    print(f"Mẫu train: {fixed_train_samples} | Seeds: {seeds} | Thiết bị: {device}")
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
    activation = "relu"
    lr = 0.005
    batch_size = 32
    max_epochs = 150
    patience = 15
    n_val = len(X_val)
    criterion = nn.MSELoss()

    raw_runs = []
    run_id = 1
    total_runs = len(depth_list) * len(seeds)

    # Bước 2: Lặp qua từng số tầng ẩn C và từng seed
    for C in depth_list:
        for seed in seeds:
            print(f"\n[Run {run_id:02d}/{total_runs:02d}] Cấu hình C={C} tầng ({units_per_layer} units/tầng) | Seed={seed} ...")

            # Thiết lập random seed
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Tạo DataLoader
            train_loader = create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
            val_loader = create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)
            test_loader = create_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

            # Khởi tạo mô hình MLP C tầng ẩn, mỗi tầng 32 nơ-ron
            model = GaussianMLP(
                input_dim=2,
                output_dim=1,
                num_hidden_layers=C,
                hidden_units=units_per_layer,
                activation=activation
            )

            ckpt_filename = f"exp_C_depth_{C}_seed{seed}.pth"
            ckpt_path = os.path.join(checkpoints_dir, ckpt_filename)
            hist_filename = f"history_exp_C_depth_{C}_seed{seed}.csv"
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
                "exp_name": "exp_C",
                "seed": seed,
                "param_name": "num_layers",
                "param_value": C,
                "num_layers": C,
                "hidden_units": units_per_layer,
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

    # Bước 3: Lưu nhật ký chi tiết và tính bảng tổng kết
    raw_runs_df = pd.DataFrame(raw_runs)
    raw_runs_csv = os.path.join(logs_dir, "raw_runs_exp_C.csv")
    raw_runs_df.to_csv(raw_runs_csv, index=False)
    print(f"\n✅ Đã lưu nhật ký chi tiết {len(raw_runs_df)} lần chạy vào: {raw_runs_csv}")

    # Tính bảng tổng hợp Mean +- Std theo mốc C
    summary_rows = []
    for C in depth_list:
        sub = raw_runs_df[raw_runs_df["param_value"] == C]
        summary_rows.append({
            "param_value": C,
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
    summary_csv = os.path.join(logs_dir, "exp_C_results.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"✅ Đã lưu bảng kết quả tổng hợp vào: {summary_csv}")

    # In bảng tổng hợp ra console
    print("\n" + "=" * 80)
    print("BẢNG KẾT QUẢ TỔNG HỢP THỰC NGHIỆM C (MEAN ± STD QUA 3 SEEDS):")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Bước 4: Vẽ biểu đồ xu hướng sai số theo C
    error_fig_path = os.path.join(figures_dir, "exp_C_depth_vs_error.png")
    plot_experiment_trend(
        x_vals=summary_df["param_value"].tolist(),
        train_errors=summary_df["train_mse_mean"].tolist(),
        val_errors=summary_df["val_mse_mean"].tolist(),
        test_errors=summary_df["test_mse_mean"].tolist(),
        x_label="Độ sâu mạng C (Số tầng ẩn - 32 units/tầng)",
        y_label="Sai số toàn phương trung bình (MSE)",
        title="Thực Nghiệm C: Ảnh Hưởng Của Độ Sâu Mạng MLP (Depth C)",
        std_test=summary_df["test_mse_std"].tolist(),
        save_path=error_fig_path
    )
    print(f"✅ Đã tạo đồ thị sai số theo độ sâu vào: {error_fig_path}")

    # Bước 5: Vẽ biểu đồ cột thời gian huấn luyện theo C
    time_fig_path = os.path.join(figures_dir, "exp_C_depth_vs_time.png")
    time_stds = [
        raw_runs_df[raw_runs_df["param_value"] == C]["train_time_sec"].std(ddof=1)
        if len(raw_runs_df[raw_runs_df["param_value"] == C]) > 1 else 0.0
        for C in depth_list
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        bars = ax.bar(
            summary_df["param_value"],
            summary_df["train_time_mean"],
            yerr=time_stds,
            capsize=5,
            color="tab:purple",
            alpha=0.85,
            edgecolor="black",
            linewidth=1.2
        )
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}s",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

        ax.set_xlabel("Độ sâu mạng C (Số tầng ẩn, 32 nơ-ron/tầng)", fontsize=11)
        ax.set_ylabel("Thời gian huấn luyện trung bình (giây)", fontsize=11)
        ax.set_title("Thực Nghiệm C: Thời Gian Huấn Luyện Theo Độ Sâu Mạng (CPU)", fontsize=12, fontweight="bold")
        ax.set_xticks(summary_df["param_value"])
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        fig.tight_layout()
        fig.savefig(time_fig_path, dpi=300, bbox_inches="tight")
        print(f"✅ Đã tạo biểu đồ thời gian huấn luyện vào: {time_fig_path}")
    finally:
        plt.close(fig)

    # Bước 6: Vẽ biểu đồ đường cong hội tụ loss theo epoch cho từng độ sâu C (Seed đại diện 42)
    loss_fig_path = os.path.join(figures_dir, "exp_C_loss_convergence.png")
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    try:
        for i, C in enumerate(depth_list):
            hist_file = os.path.join(logs_dir, f"history_exp_C_depth_{C}_seed42.csv")
            if os.path.exists(hist_file):
                h_df = pd.read_csv(hist_file)
                color = colors[i % len(colors)]
                ax.plot(
                    h_df["epoch"],
                    h_df["val_loss"],
                    label=f"C = {C} (Best Val: {h_df['val_loss'].min():.6f})",
                    color=color,
                    linewidth=2
                )
                best_idx = h_df["val_loss"].idxmin()
                best_ep = h_df.loc[best_idx, "epoch"]
                best_val = h_df.loc[best_idx, "val_loss"]
                ax.scatter([best_ep], [best_val], color=color, s=50, zorder=5)

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Validation Loss (MSE)", fontsize=11)
        ax.set_title("Thực Nghiệm C: Đường Cong Hội Tụ Loss Theo Độ Sâu Mạng (Seed 42)", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(loss_fig_path, dpi=300, bbox_inches="tight")
        print(f"✅ Đã tạo biểu đồ đường cong hội tụ loss vào: {loss_fig_path}")
    finally:
        plt.close(fig)

    print("================================================================================")
    print("HOÀN THÀNH TOÀN DIỆN THỰC NGHIỆM C!")
    print("================================================================================")


if __name__ == "__main__":
    run_experiment_C()

"""
Script: evaluate_best_model.py
Thẩm định cấu hình ứng viên trên tập Validation qua 3 seed [42, 100, 2024],
chọn checkpoint đại diện và đánh giá trên tập Test độc lập 100 mẫu.
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

import shutil
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List

from src.data_generator import load_dataset_from_csv, get_train_val_split, gaussian_2d_pdf
from src.models import GaussianMLP
from src.trainer import create_dataloader, train_mlp_pipeline, evaluate_model
from src.metrics import compute_all_metrics
from src.visualize import plot_3d_model_comparison, plot_parity_and_residuals, plot_learning_curve


def describe_config(config: dict) -> str:
    """Diễn giải cấu hình dưới dạng chuỗi dễ đọc cho console/báo cáo."""
    return (
        f"{config['num_hidden_layers']} tầng ẩn x {config['hidden_units']} nơ-ron/tầng "
        f"({config['activation']}, lr={config['lr']}, batch={config['batch_size']}, "
        f"epochs<={config['epochs']}, patience={config['patience']}, N_train={config['train_samples']})"
    )


def derive_benchmark_from_logs(logs_dir: str):
    """
    Đọc logs/exp_{A,B,C}_results.csv, trả về (benchmark_best_val_mse, fallback_config, source_exp, source_param_value).
    - Nếu thiếu bất kỳ file log nào -> raise FileNotFoundError (dừng, yêu cầu chạy xong Thực nghiệm A/B/C).
    - fallback_config được tái dựng từ dòng có val_mse_mean thấp nhất theo đúng thiết lập từng thực nghiệm:
        exp A: param_value = số mẫu train (mạng 1 tầng 64 nơ-ron).
        exp B: param_value = số nơ-ron tầng ẩn (mạng 1 tầng, 1500 mẫu train).
        exp C: param_value = số tầng ẩn (32 nơ-ron/tầng, 1500 mẫu train).
    """
    exp_files = {exp: os.path.join(logs_dir, f"exp_{exp}_results.csv") for exp in ("A", "B", "C")}
    missing = [path for path in exp_files.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Thiếu file kết quả thực nghiệm: " + ", ".join(missing)
            + ". Vui lòng chạy xong Thực nghiệm A/B/C (run_experiment_A/B/C.py) trước khi thẩm định Best Model."
        )
    frames = []
    for exp, path in exp_files.items():
        df = pd.read_csv(path)
        df["exp"] = exp
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    best_row = all_df.loc[all_df["val_mse_mean"].idxmin()]
    exp = str(best_row["exp"])
    param_value = int(best_row["param_value"])
    benchmark_val_mse = float(best_row["val_mse_mean"])

    base = {"activation": "relu", "lr": 0.005, "batch_size": 32, "epochs": 150, "patience": 15}
    if exp == "A":
        fallback_config = {"num_hidden_layers": 1, "hidden_units": 64, "train_samples": param_value, **base}
    elif exp == "B":
        fallback_config = {"num_hidden_layers": 1, "hidden_units": param_value, "train_samples": 1500, **base}
    else:  # exp C
        fallback_config = {"num_hidden_layers": param_value, "hidden_units": 32, "train_samples": 1500, **base}
    return benchmark_val_mse, fallback_config, exp, param_value


def train_config_on_seeds(
    config: dict,
    seeds: list,
    pool_X: np.ndarray,
    pool_y: np.ndarray,
    logs_dir: str,
    checkpoints_dir: str,
    checkpoint_template: str,
    history_template: str,
    device: torch.device
) -> List[Dict[str, Any]]:
    """
    Huấn luyện một cấu hình trên nhiều seed theo đúng giao thức thực nghiệm A/B/C:
    đặt seed -> tạo DataLoader (shuffle train) -> khởi tạo GaussianMLP -> train_mlp_pipeline (EarlyStopping)
    -> đo Validation MSE bằng evaluate_model trên đúng tập val 400 mẫu cố định.
    Trả về danh sách kết quả từng seed: {seed, val_mse, best_epoch, total_epochs, checkpoint_path, history_path}.
    """
    X_train, y_train, X_val, y_val = get_train_val_split(
        pool_X, pool_y, n_train=config["train_samples"], n_val=400
    )
    criterion = nn.MSELoss()
    results = []
    for seed in seeds:
        print(f"  -> Huấn luyện seed={seed} ...")
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader = create_dataloader(X_train, y_train, batch_size=config["batch_size"], shuffle=True)
        val_loader = create_dataloader(X_val, y_val, batch_size=config["batch_size"], shuffle=False)

        model = GaussianMLP(
            input_dim=2,
            output_dim=1,
            num_hidden_layers=config["num_hidden_layers"],
            hidden_units=config["hidden_units"],
            activation=config["activation"]
        )
        ckpt_path = os.path.join(checkpoints_dir, checkpoint_template.format(seed=seed))
        hist_path = os.path.join(logs_dir, history_template.format(seed=seed))

        train_res = train_mlp_pipeline(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config["epochs"],
            lr=config["lr"],
            optimizer_name="adam",
            checkpoint_path=ckpt_path,
            history_csv_path=hist_path,
            patience=config["patience"],
            device=device,
            verbose=False
        )
        val_mse, _, _ = evaluate_model(train_res["model"], val_loader, criterion, device)
        print(f"     Seed {seed}: Val MSE={val_mse:.8f} | Best epoch={train_res['best_epoch']} | Dừng tại epoch {train_res['total_epochs']}")
        results.append({
            "seed": int(seed),
            "val_mse": float(val_mse),
            "best_epoch": int(train_res["best_epoch"]),
            "total_epochs": int(train_res["total_epochs"]),
            "checkpoint_path": ckpt_path,
            "history_path": hist_path
        })
    return results


def collect_fallback_runs_from_logs(
    config: dict,
    source_exp: str,
    source_param_value: int,
    seeds: list,
    logs_dir: str,
    checkpoints_dir: str
) -> Optional[List[Dict[str, Any]]]:
    """
    Khi fallback được kích hoạt, thu thập kết quả từng seed của cấu hình benchmark
    từ nhật ký logs/raw_runs_exp_{exp}.csv và tái sử dụng checkpoint/history đã có.
    Trả về None nếu nhật ký hoặc file checkpoint/history không đầy đủ (khi đó cần huấn luyện lại).
    """
    raw_path = os.path.join(logs_dir, f"raw_runs_exp_{source_exp}.csv")
    if not os.path.exists(raw_path):
        return None
    raw_df = pd.read_csv(raw_path)
    sub = raw_df[raw_df["param_value"] == source_param_value]
    results = []
    for seed in seeds:
        rows = sub[sub["seed"] == seed]
        if len(rows) != 1:
            return None
        row = rows.iloc[0]
        ckpt_path = os.path.join(checkpoints_dir, str(row["checkpoint_file"]))
        hist_path = os.path.join(logs_dir, str(row["history_file"]))
        if not (os.path.exists(ckpt_path) and os.path.exists(hist_path)):
            return None
        results.append({
            "seed": int(seed),
            "val_mse": float(row["val_mse"]),
            "best_epoch": int(row["best_epoch"]),
            "total_epochs": int(row["total_epochs"]),
            "checkpoint_path": ckpt_path,
            "history_path": hist_path
        })
    return results


def summarize_runs(runs: List[Dict[str, Any]]):
    """Tính (mean, std, seed đại diện) theo quy tắc: gần mean nhất, hòa chọn seed nhỏ nhất."""
    val_by_seed = {run["seed"]: run["val_mse"] for run in runs}
    seeds_sorted = sorted(val_by_seed.keys())
    mean_val = float(np.mean([val_by_seed[s] for s in seeds_sorted]))
    std_val = float(np.std([val_by_seed[s] for s in seeds_sorted], ddof=1)) if len(seeds_sorted) > 1 else 0.0
    rep_seed = min(seeds_sorted, key=lambda s: (abs(val_by_seed[s] - mean_val), s))
    return mean_val, std_val, rep_seed


def evaluate_best_model(
    data_dir: str = "data",
    logs_dir: str = "logs",
    figures_dir: str = "figures",
    checkpoints_dir: str = "checkpoints",
    candidate_config: dict = {
        "num_hidden_layers": 2,
        "hidden_units": 64,
        "activation": "relu",
        "lr": 0.005,
        "batch_size": 32,
        "epochs": 150,
        "patience": 15,
        "train_samples": 2000
    },
    benchmark_best_val_mse: Optional[float] = None,
    fallback_config: Optional[dict] = None,
    seeds: list = [42, 100, 2024],
    device: torch.device = torch.device("cpu")
):
    """
    Quy trình thẩm định và đánh giá mô hình tối ưu chuẩn hóa:
    1. Đọc benchmark từ logs/exp_{A,B,C}_results.csv nếu benchmark_best_val_mse là None (dừng nếu thiếu logs).
    2. Huấn luyện ứng viên trên 3 seeds [42, 100, 2024], tính Mean +- Std Validation MSE.
    3. Đối sánh với benchmark; kích hoạt Fallback nếu ứng viên không tốt hơn.
    4. Chọn checkpoint đại diện theo quy tắc duy nhất: gần mean nhất (hòa chọn seed nhỏ nhất).
    5. Đánh giá ngoại kiểm trên tập Test 100 mẫu độc lập và xuất biểu đồ, metrics.
    """
    print("================================================================================")
    print("THẨM ĐỊNH VÀ ĐÁNH GIÁ MÔ HÌNH TỐI ƯU TOÀN DIỆN (BEST MODEL EVALUATION)")
    print(f"Cấu hình ứng viên khảo sát: {candidate_config}")
    print(f"Seeds thẩm định: {seeds} | Thiết bị: {device}")
    print("================================================================================")

    for directory in (logs_dir, figures_dir, checkpoints_dir):
        os.makedirs(directory, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Bước 1: Kiểm tra dữ liệu chuẩn; nạp Pool chia train/val cố định.
    #         Tập Test CHỈ được nạp ở Bước 4, sau khi mô hình đã được chốt.
    # ------------------------------------------------------------------ #
    test_path = os.path.join(data_dir, "test_dataset_100.csv")
    pool_path = os.path.join(data_dir, "train_val_pool.csv")
    if not os.path.exists(test_path) or not os.path.exists(pool_path):
        raise FileNotFoundError(
            "Không tìm thấy tập dữ liệu chuẩn trong thư mục 'data'. "
            "Vui lòng chạy 'python src/data_generator.py' để sinh data/test_dataset_100.csv và data/train_val_pool.csv."
        )
    pool_X, pool_y = load_dataset_from_csv(pool_path)
    _, _, X_val, y_val = get_train_val_split(pool_X, pool_y, n_train=candidate_config["train_samples"], n_val=400)
    print(f"\n[Bước 1] Đã nạp Pool={len(pool_X)} mẫu | Val cố định={len(X_val)} mẫu | Test={os.path.basename(test_path)} (nạp ở Bước 4)")
    print("         (Validation dùng để chọn mô hình; script này đánh giá checkpoint đã chốt trên Test một lần mỗi lượt chạy)")

    # ------------------------------------------------------------------ #
    # Bước 2: Xác định Benchmark thực tế từ logs A/B/C
    # ------------------------------------------------------------------ #
    fallback_src_exp = "external"
    fallback_src_param = -1
    if benchmark_best_val_mse is None or fallback_config is None:
        bench_val, derived_fallback, src_exp, src_param = derive_benchmark_from_logs(logs_dir)
        fallback_src_exp, fallback_src_param = src_exp, src_param
        if benchmark_best_val_mse is None:
            benchmark_best_val_mse = bench_val
        if fallback_config is None:
            fallback_config = derived_fallback
    print(f"\n[Bước 2] Benchmark tốt nhất từ logs A/B/C: exp_{fallback_src_exp} (param_value={fallback_src_param})")
    print(f"         Cấu hình benchmark (fallback): {describe_config(fallback_config)}")
    print(f"         benchmark_best_val_mse = {benchmark_best_val_mse:.8f}")

    # ------------------------------------------------------------------ #
    # Bước 3: Huấn luyện ứng viên trên 3 seeds và đối sánh với benchmark
    # ------------------------------------------------------------------ #
    print(f"\n[Bước 3] Huấn luyện ứng viên ({describe_config(candidate_config)}) trên {len(seeds)} seeds ...")
    candidate_runs = train_config_on_seeds(
        config=candidate_config,
        seeds=seeds,
        pool_X=pool_X,
        pool_y=pool_y,
        logs_dir=logs_dir,
        checkpoints_dir=checkpoints_dir,
        checkpoint_template="candidate_best_model_seed{seed}.pth",
        history_template="history_candidate_seed{seed}.csv",
        device=device
    )
    candidate_mean, candidate_std, _ = summarize_runs(candidate_runs)
    print(f"\n  Ứng viên tổ hợp: Val MSE = {candidate_mean:.8f} ± {candidate_std:.8f}")
    print(f"  Benchmark A/B/C: Val MSE = {benchmark_best_val_mse:.8f}")

    if candidate_mean < benchmark_best_val_mse:
        selection_mode = "candidate"
        source_exp = "candidate_combo"
        source_param = -1
        selected_config = dict(candidate_config)
        selected_runs = candidate_runs
        print("\n  >> Ứng viên tổ hợp VƯỢT TRỘI hơn benchmark. Chốt dùng cấu hình ứng viên!")
    else:
        selection_mode = "fallback"
        print("\n  >> Ứng viên tổ hợp không vượt trội hơn benchmark; Kích hoạt fallback!")
        source_exp = fallback_src_exp
        source_param = fallback_src_param
        selected_config = dict(fallback_config)
        selected_runs = collect_fallback_runs_from_logs(
            config=selected_config,
            source_exp=str(source_exp),
            source_param_value=int(source_param),
            seeds=seeds,
            logs_dir=logs_dir,
            checkpoints_dir=checkpoints_dir
        ) if source_exp in ("A", "B", "C") and source_param != -1 else None
        if selected_runs is None:
            print("  Nhật ký/checkpoint benchmark không đầy đủ -> huấn luyện lại cấu hình benchmark trên 3 seeds ...")
            selected_runs = train_config_on_seeds(
                config=selected_config,
                seeds=seeds,
                pool_X=pool_X,
                pool_y=pool_y,
                logs_dir=logs_dir,
                checkpoints_dir=checkpoints_dir,
                checkpoint_template="fallback_best_model_seed{seed}.pth",
                history_template="history_fallback_best_model_seed{seed}.csv",
                device=device
            )
        else:
            print(f"  Tái sử dụng {len(selected_runs)} checkpoint của benchmark từ logs (không huấn luyện lại).")

    # Chọn checkpoint đại diện theo quy tắc duy nhất (gần mean nhất, hòa chọn seed nhỏ nhất)
    selected_mean, selected_std, rep_seed = summarize_runs(selected_runs)
    rep_run = next(run for run in selected_runs if run["seed"] == rep_seed)
    print(f"\n  Checkpoint đại diện: seed={rep_seed} "
          f"(Val MSE={rep_run['val_mse']:.8f}, gần mean {selected_mean:.8f} nhất; hòa chọn seed nhỏ nhất)")

    best_ckpt_path = os.path.join(checkpoints_dir, "best_model.pth")
    best_hist_path = os.path.join(logs_dir, "history_best_model.csv")
    shutil.copyfile(rep_run["checkpoint_path"], best_ckpt_path)
    shutil.copyfile(rep_run["history_path"], best_hist_path)
    print(f"  Đã sao chép checkpoint -> {best_ckpt_path}")
    print(f"  Đã sao chép history   -> {best_hist_path}")

    # ------------------------------------------------------------------ #
    # Bước 4: Đánh giá ngoại kết checkpoint đã chốt trên tập Test 100 mẫu
    #         (tập Test được nạp tại đây, sau khi mọi quyết định đã khóa)
    # ------------------------------------------------------------------ #
    print("\n[Bước 4] Nạp tập Test độc lập 100 mẫu và đánh giá checkpoint đã chốt (trọng số đóng băng) ...")
    X_test, y_test = load_dataset_from_csv(test_path)
    final_model = GaussianMLP(
        input_dim=2,
        output_dim=1,
        num_hidden_layers=selected_config["num_hidden_layers"],
        hidden_units=selected_config["hidden_units"],
        activation=selected_config["activation"]
    )
    final_model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    final_model.to(device)

    test_loader = create_dataloader(X_test, y_test, batch_size=selected_config["batch_size"], shuffle=False)
    criterion = nn.MSELoss()
    t_eval = time.perf_counter()
    test_mse, y_true_test, y_pred_test = evaluate_model(final_model, test_loader, criterion, device)
    eval_time_sec = time.perf_counter() - t_eval
    metrics = compute_all_metrics(y_true_test, y_pred_test)
    n_negative = int(np.sum(y_pred_test < 0))
    print(f"  Test MSE={metrics['mse']:.8f} | MAE={metrics['mae']:.8f} | RMSE={metrics['rmse']:.8f} | R2={metrics['r2']:.6f} | Âm: {n_negative}/{len(y_pred_test)}")

    model_name = (f"MLP_{selected_config['num_hidden_layers']}x{selected_config['hidden_units']}_"
                  f"{selected_config['activation']}_A{selected_config['train_samples']}")
    # Tổng hợp chỉ số đánh giá và thông tin cấu hình của Best Model
    metrics_row = {
        "model_name": model_name,
        "num_layers": selected_config["num_hidden_layers"],
        "hidden_units": selected_config["hidden_units"],
        "activation": selected_config["activation"],
        "train_samples": selected_config["train_samples"],
        "mse": metrics["mse"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "eval_time_sec": float(eval_time_sec),
        "selection_mode": selection_mode,
        "source_experiment": source_exp,
        "source_param_value": source_param,
        "representative_seed": rep_seed,
        "lr": selected_config["lr"],
        "batch_size": selected_config["batch_size"],
        "max_epochs": selected_config["epochs"],
        "patience": selected_config["patience"],
        "n_val": 400,
        "n_test": len(X_test),
        "best_epoch": rep_run["best_epoch"],
        "total_epochs": rep_run["total_epochs"],
        "benchmark_val_mse": float(benchmark_best_val_mse),
        "candidate_val_mse_mean": float(candidate_mean),
        "candidate_val_mse_std": float(candidate_std),
        "selected_val_mse_mean": float(selected_mean),
        "selected_val_mse_std": float(selected_std),
        "selected_val_mse_representative": float(rep_run["val_mse"]),
        "test_negative_predictions": n_negative,
        "checkpoint_file": "best_model.pth"
    }
    metrics_csv = os.path.join(logs_dir, "best_model_test_metrics.csv")
    pd.DataFrame([metrics_row]).to_csv(metrics_csv, index=False)
    print(f"  Đã lưu bảng metrics test -> {metrics_csv}")

    # ------------------------------------------------------------------ #
    # Bước 5: Trực quan hóa kết quả Best Model
    # ------------------------------------------------------------------ #
    print("\n[Bước 5] Xuất các biểu đồ trực quan Best Model ...")
    history_df = pd.read_csv(best_hist_path)

    lc_path = os.path.join(figures_dir, "best_model_learning_curve.png")
    plot_learning_curve(
        history_df,
        title=f"Đường cong học tập Best Model (seed {rep_seed}, "
              f"{selected_config['num_hidden_layers']}x{selected_config['hidden_units']})",
        save_path=lc_path
    )
    print(f"  Đã tạo {lc_path}")

    parity_path = os.path.join(figures_dir, "best_model_parity_and_residuals.png")
    plot_parity_and_residuals(
        y_true_test,
        y_pred_test,
        title_prefix="Tập Test 100 mẫu độc lập",
        save_path=parity_path
    )
    print(f"  Đã tạo {parity_path}")

    cmp_path = os.path.join(figures_dir, "best_model_3d_comparison.png")
    plot_3d_model_comparison(
        gaussian_2d_pdf,
        final_model,
        device,
        save_path=cmp_path
    )
    print(f"  Đã tạo {cmp_path}")

    print("\n================================================================================")
    print("TỔNG KẾT BEST MODEL:")
    print(f"  - Chế độ chọn    : {selection_mode.upper()}" + (f" (tái sử dụng exp_{source_exp}, param={source_param})" if selection_mode == 'fallback' else ""))
    print(f"  - Mô hình        : {model_name}")
    print(f"  - Seed đại diện  : {rep_seed} (checkpoint: best_model.pth)")
    print(f"  - Val MSE (3 seed): {selected_mean:.8f} ± {selected_std:.8f}")
    print(f"  - TEST (100 mẫu) : MSE={metrics['mse']:.8f} | MAE={metrics['mae']:.8f} | "
          f"RMSE={metrics['rmse']:.8f} | R2={metrics['r2']:.6f} | dự đoán âm: {n_negative}/{len(y_pred_test)}")
    print("  (Lưu ý: tập test cũng đã được dùng để báo cáo Test MSE của từng cấu hình trong")
    print("   Thực nghiệm A/B/C; trong script này test chỉ được đánh giá sau khi chốt mô hình.)")
    print("================================================================================")


if __name__ == "__main__":
    evaluate_best_model()

"""Ablation study E0–E5 và đánh giá từng feature — Lab 03, Thành viên 4.

Chạy hai thực nghiệm chính theo plan mục 8:

1. **Ablation E0–E5**: so sánh Original vs các tổ hợp feature mới với cùng
   hai mô hình sklearn (Ridge, GradientBoosting) qua 5-Fold CV và MLP qua
   holdout validation, dùng đúng siêu tham số đã tinh chỉnh ở mục 4/5 plan.
2. **Đánh giá từng feature**: thêm từng feature một vào tập Original, đo mức
   thay đổi RMSE so với baseline E0 để biết feature nào thực sự giúp model.

Kết quả xuất ra:

- ``outputs/results/ablation_results.csv``
- ``outputs/results/feature_evaluation.csv``
- ``outputs/results/ablation_summary.json``
- ``outputs/figures/ablation_comparison.png``
- ``outputs/figures/feature_evaluation.png``
- ``outputs/figures/before_after_feature_engineering.png``

Chạy từ thư mục gốc project::

    python src/run_ablation.py

Lưu ý metric: Ridge/GradientBoosting dùng 5-Fold CV RMSE trên
``log1p(SalePrice)``; MLP dùng validation RMSE trên cùng một holdout split
(seed 42, 20%) như Thành viên 3 — hai con số khác kiểu đánh giá nên được đặt
trong hai cột riêng, không trộn lẫn.
"""

from dataclasses import replace
import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    load_data,
    add_target_log,
    split_feature_target,
    select_column_groups,
    build_preprocessor,
    RANDOM_STATE,
)
from src.features import (
    ABLATION_EXPERIMENTS,
    ENGINEERED_FEATURES,
    add_features,
)
from src.mlp_model import MLPConfig
from src import train_mlp as mlp
from src.train_sklearn import create_pipeline, evaluate_pipeline_cv

RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
ABLATION_CKPT_DIR = PROJECT_ROOT / "models" / "ablation"

# Siêu tham số dùng chung cho mọi experiment — lấy từ bảng kết quả Thành viên 2
# (sklearn_results.csv) để mọi feature set được đánh giá bằng cùng một model.
RIDGE_PARAMS = {"alpha": 30.0}
GB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.8,
}

# Cấu hình MLP tốt nhất từ Thành viên 3 (mlp_results.csv: lr=3e-4, 256-128-64).
MLP_BASE_CONFIG = MLPConfig(
    name="MLP_best",
    hidden_dims=(256, 128, 64),
    dropout=0.2,
    batch_norm=False,
    learning_rate=3e-4,
    weight_decay=1e-4,
    batch_size=64,
)

# Bảng màu đã validate (dataviz reference palette, light mode):
# blue / orange / aqua — 3 slot đầu pass all-pairs CVD checks.
SERIES_COLORS = {
    "Ridge": "#2a78d6",
    "GradientBoosting": "#eb6834",
    "MLP": "#1baf7a",
}
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
DIVERGE_BAD = "#e34948"   # RMSE tăng lên sau khi thêm feature
DIVERGE_GOOD = "#2a78d6"  # RMSE giảm (cải thiện) sau khi thêm feature


def load_train(project_dir=PROJECT_ROOT):
    """Đọc train.csv và chuẩn bị X (features) + y (log1p SalePrice)."""
    train_df, _ = load_data(Path(project_dir) / "data")
    train_df = add_target_log(train_df)
    X, y = split_feature_target(train_df)
    return X, y


def make_sklearn_pipelines(numeric_cols, categorical_cols):
    """Tạo pipeline Ridge và GradientBoosting với siêu tham số cố định."""
    ridge = create_pipeline(
        Ridge(**RIDGE_PARAMS, random_state=RANDOM_STATE),
        numeric_cols, categorical_cols, scale_numeric=True,
    )
    gb = create_pipeline(
        GradientBoostingRegressor(**GB_PARAMS, random_state=RANDOM_STATE),
        numeric_cols, categorical_cols, scale_numeric=False,
    )
    return {"Ridge": ridge, "GradientBoosting": gb}


def evaluate_sklearn(X_exp, y, cv):
    """5-Fold CV RMSE cho Ridge và GradientBoosting trên một feature set."""
    numeric_cols, categorical_cols = select_column_groups(X_exp)
    pipelines = make_sklearn_pipelines(numeric_cols, categorical_cols)
    scores = {}
    for name, pipeline in pipelines.items():
        result = evaluate_pipeline_cv(pipeline, X_exp, y, cv=cv)
        scores[name] = result
    return scores


def evaluate_mlp(X_exp, y_log, run_name, project_dir=PROJECT_ROOT,
                 seed=RANDOM_STATE, device="cpu", threads=2,
                 max_epochs=400, patience=40):
    """Huấn luyện MLP (cấu hình tốt nhất) trên holdout split cho một feature set.

    Dùng đúng cách chia 80/20 với seed=42 như ``src/train_mlp.py`` để số liệu
    validation RMSE của các feature set so sánh được với nhau và với bảng của
    Thành viên 3.
    """
    import torch

    project_dir = Path(project_dir)
    torch.set_num_threads(threads)

    train_idx, valid_idx = train_test_split(
        np.arange(len(X_exp)), test_size=0.2, random_state=seed, shuffle=True)
    X_train_df, X_valid_df = X_exp.iloc[train_idx], X_exp.iloc[valid_idx]
    y_train, y_valid = y_log[train_idx], y_log[valid_idx]

    numeric_cols, categorical_cols = select_column_groups(X_train_df)
    preprocessor = build_preprocessor(
        numeric_cols, categorical_cols, scale_numeric=True, sparse_threshold=0)
    X_train = mlp.dense_float32(preprocessor.fit_transform(X_train_df))
    X_valid = mlp.dense_float32(preprocessor.transform(X_valid_df))
    statistics = mlp.target_statistics(y_train)

    ABLATION_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor_rel = f"models/ablation/{run_name}_preprocessor.joblib"
    preprocessor_path = project_dir / preprocessor_rel
    import joblib
    joblib.dump(preprocessor, preprocessor_path)

    feature_set = ("engineered"
                   if any(c in X_exp.columns for c in ENGINEERED_FEATURES)
                   else "original")
    config = replace(MLP_BASE_CONFIG, name=f"MLP_{run_name}")
    mlp.set_seed(seed)
    result, _history = mlp.fit_configuration(
        X_train, X_valid, y_train, y_valid, config, statistics,
        ABLATION_CKPT_DIR / f"{run_name}.pt", X_exp.columns, feature_set,
        seed, device, preprocessor_rel, max_epochs, patience)
    result["checkpoint"] = Path(result["checkpoint"]).name
    return result


def run_ablation(project_dir=PROJECT_ROOT, with_mlp=True, seed=RANDOM_STATE,
                 verbose=True):
    """Chạy ablation E0–E5 (plan mục 8) và lưu bảng kết quả."""
    project_dir = Path(project_dir)
    X, y = load_train(project_dir)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    y_log = y.to_numpy(dtype=np.float64)

    rows = []
    for exp_id, (label, features) in ABLATION_EXPERIMENTS.items():
        started = time.perf_counter()
        X_exp = add_features(X, features=features) if features else X.copy()
        scores = evaluate_sklearn(X_exp, y, cv)

        row = {
            "experiment": exp_id,
            "feature_set": label,
            "features": "+".join(features) if features else "Original",
            "n_raw_features": X_exp.shape[1],
            "ridge_cv_rmse": round(scores["Ridge"]["rmse_mean"], 5),
            "ridge_cv_std": round(scores["Ridge"]["rmse_std"], 5),
            "gb_cv_rmse": round(scores["GradientBoosting"]["rmse_mean"], 5),
            "gb_cv_std": round(scores["GradientBoosting"]["rmse_std"], 5),
        }
        if with_mlp:
            mlp_result = evaluate_mlp(X_exp, y_log, exp_id, project_dir, seed)
            row["mlp_val_rmse"] = round(mlp_result["val_rmse"], 5)
            row["mlp_best_epoch"] = mlp_result["best_epoch"]
        row["seconds"] = round(time.perf_counter() - started, 1)
        rows.append(row)
        if verbose:
            print(f"[{exp_id}] {label}: Ridge={row['ridge_cv_rmse']}, "
                  f"GB={row['gb_cv_rmse']}"
                  + (f", MLP={row.get('mlp_val_rmse')}" if with_mlp else ""),
                  flush=True)

    results = pd.DataFrame(rows)

    # Mức cải thiện so với E0 (số âm = RMSE thấp hơn = tốt hơn).
    baseline = results.loc[results["experiment"] == "E0"].iloc[0]
    for col, out in [("ridge_cv_rmse", "ridge_delta_vs_e0"),
                     ("gb_cv_rmse", "gb_delta_vs_e0"),
                     ("mlp_val_rmse", "mlp_delta_vs_e0")]:
        if col in results.columns:
            results[out] = (results[col] - baseline[col]).round(5)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)

    summary = {
        "owner": "Thanh vien 4 — Feature Engineering",
        "metric": "RMSE on log1p(SalePrice)",
        "sklearn_evaluation": "5-Fold KFold(shuffle=True, random_state=42)",
        "mlp_evaluation": "holdout 80/20, random_state=42 (giong thanh vien 3)",
        "ridge_params": RIDGE_PARAMS,
        "gradient_boosting_params": GB_PARAMS,
        "mlp_config": {
            "hidden_dims": list(MLP_BASE_CONFIG.hidden_dims),
            "dropout": MLP_BASE_CONFIG.dropout,
            "learning_rate": MLP_BASE_CONFIG.learning_rate,
            "batch_size": MLP_BASE_CONFIG.batch_size,
        },
        "seed": seed,
        "experiments": {k: {"label": v[0], "features": list(v[1])}
                        for k, v in ABLATION_EXPERIMENTS.items()},
        "kaggle_score": None,
    }
    (RESULTS_DIR / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def run_feature_evaluation(project_dir=PROJECT_ROOT, seed=RANDOM_STATE,
                           verbose=True):
    """Thêm từng feature một vào Original và đo delta RMSE so với E0.

    Trả về DataFrame với delta âm nghĩa là feature đó giúp giảm RMSE.
    """
    X, y = load_train(project_dir)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    base = evaluate_sklearn(X, y, cv)
    base_ridge = base["Ridge"]["rmse_mean"]
    base_gb = base["GradientBoosting"]["rmse_mean"]
    if verbose:
        print(f"Baseline E0: Ridge={base_ridge:.5f}, GB={base_gb:.5f}",
              flush=True)

    rows = []
    for feature in ENGINEERED_FEATURES:
        X_f = add_features(X, features=[feature])
        scores = evaluate_sklearn(X_f, y, cv)
        ridge_rmse = scores["Ridge"]["rmse_mean"]
        gb_rmse = scores["GradientBoosting"]["rmse_mean"]
        rows.append({
            "feature": feature,
            "ridge_cv_rmse": round(ridge_rmse, 5),
            "ridge_delta_vs_e0": round(ridge_rmse - base_ridge, 5),
            "gb_cv_rmse": round(gb_rmse, 5),
            "gb_delta_vs_e0": round(gb_rmse - base_gb, 5),
            "gb_improves": bool(gb_rmse < base_gb),
        })
        if verbose:
            print(f"  +{feature:<15} Ridge={ridge_rmse:.5f} "
                  f"({ridge_rmse - base_ridge:+.5f}) | GB={gb_rmse:.5f} "
                  f"({gb_rmse - base_gb:+.5f})", flush=True)

    results = pd.DataFrame(rows).sort_values("gb_delta_vs_e0").reset_index(drop=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / "feature_evaluation.csv", index=False)
    return results, round(base_ridge, 5), round(base_gb, 5)


# ---------------------------------------------------------------------------
# Biểu đồ
# ---------------------------------------------------------------------------


def _style_axis(axis):
    axis.set_facecolor(SURFACE)
    axis.grid(axis="y", color=GRIDLINE, linewidth=0.8, alpha=1.0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRIDLINE)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.xaxis.label.set_color(INK_SECONDARY)
    axis.yaxis.label.set_color(INK_SECONDARY)
    axis.title.set_color(INK)


def plot_ablation(results, figures_dir=FIGURES_DIR):
    """Grouped bar: RMSE E0–E5 của 3 model (Ridge, GradientBoosting, MLP)."""
    series = [("Ridge", "ridge_cv_rmse"),
              ("GradientBoosting", "gb_cv_rmse"),
              ("MLP", "mlp_val_rmse")]
    series = [(name, col) for name, col in series if col in results.columns]

    x = np.arange(len(results))
    width = 0.26
    fig, axis = plt.subplots(figsize=(10, 5.2))
    fig.patch.set_facecolor(SURFACE)

    for i, (name, col) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        bars = axis.bar(x + offset, results[col], width * 0.92,
                        label=name, color=SERIES_COLORS[name],
                        edgecolor=SURFACE, linewidth=1.2)
        for bar in bars:
            axis.annotate(f"{bar.get_height():.4f}",
                          (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          textcoords="offset points", xytext=(0, 3),
                          ha="center", fontsize=7.5, color=INK_SECONDARY)

    axis.set_xticks(x)
    axis.set_xticklabels(
        [f"{e}\n{label}" for e, label in
         zip(results["experiment"], results["feature_set"])], fontsize=8.5)
    axis.set_ylabel("RMSE trên log1p(SalePrice)")
    axis.set_title("Ablation study E0–E5: ảnh hưởng của feature engineering",
                   fontsize=12, pad=12)
    axis.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    # Bar chart phải có trục y bắt đầu từ 0 (không cắt cụt để phóng đại khác biệt);
    # số liệu chính xác đã được in trên đầu mỗi thanh và trong bảng CSV.
    upper = max(results[col].max() for _, col in series)
    axis.set_ylim(0, upper * 1.14)
    _style_axis(axis)
    fig.tight_layout()
    path = Path(figures_dir) / "ablation_comparison.png"
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_feature_evaluation(feature_results, base_ridge, base_gb,
                            figures_dir=FIGURES_DIR):
    """Hai panel ngang: delta RMSE từng feature với Ridge và GradientBoosting.

    Xanh = RMSE giảm (feature hữu ích), đỏ = RMSE tăng (feature gây hại).
    """
    # Sắp xếp theo delta của GradientBoosting: feature cải thiện nhất ở trên.
    data = feature_results.sort_values("gb_delta_vs_e0").reset_index(drop=True)
    y = np.arange(len(data))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    panels = [
        ("Ridge", "ridge_delta_vs_e0", base_ridge),
        ("GradientBoosting", "gb_delta_vs_e0", base_gb),
    ]
    for axis, (name, col, base) in zip(axes, panels):
        values = data[col]
        colors = [DIVERGE_GOOD if v < 0 else DIVERGE_BAD for v in values]
        bars = axis.barh(y, values, height=0.62, color=colors,
                         edgecolor=SURFACE, linewidth=1.0)
        axis.set_title(f"{name} (baseline E0 = {base:.5f})", fontsize=10.5,
                       pad=10)
        axis.set_xlabel("ΔRMSE so với E0 (âm = cải thiện)")
        axis.axvline(0, color=MUTED, linewidth=1.0)
        limit = max(abs(values.min()), abs(values.max())) * 1.35
        axis.set_xlim(-limit, limit)
        for bar, value in zip(bars, values):
            axis.annotate(f"{value:+.5f}",
                          (value, bar.get_y() + bar.get_height() / 2),
                          xytext=(4 if value >= 0 else -4, 0),
                          textcoords="offset points",
                          va="center",
                          ha="left" if value >= 0 else "right",
                          fontsize=8, color=INK_SECONDARY)
        axis.set_facecolor(SURFACE)
        axis.grid(axis="x", color=GRIDLINE, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRIDLINE)
        axis.tick_params(colors=MUTED, labelsize=9)
        axis.xaxis.label.set_color(INK_SECONDARY)
        axis.title.set_color(INK)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(data["feature"], fontsize=9.5)
    axes[0].tick_params(axis="y", labelcolor=INK)
    fig.suptitle("Đánh giá từng feature (thêm lần lượt vào tập Original)",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = Path(figures_dir) / "feature_evaluation.png"
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_before_after(results, figures_dir=FIGURES_DIR):
    """So sánh trực tiếp E0 (Original) vs E5 (All engineered)."""
    rows = results[results["experiment"].isin(["E0", "E5"])].set_index(
        "experiment")
    series = [("Ridge", "ridge_cv_rmse"),
              ("GradientBoosting", "gb_cv_rmse"),
              ("MLP", "mlp_val_rmse")]
    series = [(name, col) for name, col in series if col in rows.columns]

    x = np.arange(len(series))
    width = 0.34
    fig, axis = plt.subplots(figsize=(8.4, 5))
    fig.patch.set_facecolor(SURFACE)

    values_e0 = [rows.loc["E0", col] for _, col in series]
    values_e5 = [rows.loc["E5", col] for _, col in series]

    bars_e0 = axis.bar(x - width / 2, values_e0, width * 0.94,
                       label="E0 — Original", color=MUTED,
                       edgecolor=SURFACE, linewidth=1.2)
    bars_e5 = axis.bar(x + width / 2, values_e5, width * 0.94,
                       label="E5 — All engineered",
                       color=[SERIES_COLORS[name] for name, _ in series],
                       edgecolor=SURFACE, linewidth=1.2)

    for bars in (bars_e0, bars_e5):
        for bar in bars:
            axis.annotate(f"{bar.get_height():.4f}",
                          (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          textcoords="offset points", xytext=(0, 3),
                          ha="center", fontsize=8.5, color=INK_SECONDARY)

    # Nhãn model theo từng cặp bar (không dùng màu để nhận diện model).
    axis.set_xticks(x)
    axis.set_xticklabels([name for name, _ in series], fontsize=10)

    # Trục y bắt đầu từ 0; khác biệt nhỏ được ghi rõ bằng số trên đầu thanh.
    upper = max(values_e0 + values_e5)
    axis.set_ylim(0, upper * 1.16)
    axis.set_ylabel("RMSE trên log1p(SalePrice)")
    axis.set_title("Trước / sau feature engineering (E0 vs E5)",
                   fontsize=12, pad=12)
    axis.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    _style_axis(axis)
    fig.tight_layout()
    path = Path(figures_dir) / "before_after_feature_engineering.png"
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return path


def write_final_table(project_dir=PROJECT_ROOT):
    """Tổng hợp bảng kết quả cuối (plan mục 17) -> experiment_results.csv.

    Gộp bảng của Thành viên 2 (sklearn_results.csv), Thành viên 3
    (mlp_results.csv) và ablation E0/E5 của Thành viên 4. Điểm Kaggle để
    trống cho tới khi nhóm nộp file lên Kaggle (không điền số giả định).
    """
    project_dir = Path(project_dir)
    results_dir = project_dir / "outputs" / "results"

    rows = []
    sklearn_csv = results_dir / "sklearn_results.csv"
    if sklearn_csv.exists():
        df_sk = pd.read_csv(sklearn_csv)
        for _, r in df_sk.iterrows():
            if r["model"] in ("Ridge", "RandomForest", "GradientBoosting (Tuned)"):
                rows.append({
                    "Model": r["model"],
                    "Feature Set": "Original",
                    "CV/Val RMSE": r["cv_rmse_mean"],
                    "Std": r["cv_rmse_std"],
                    "Evaluation": "5-Fold CV",
                    "Source": "outputs/results/sklearn_results.csv",
                    "Kaggle Score": "",
                })

    mlp_csv = results_dir / "mlp_results.csv"
    if mlp_csv.exists():
        df_mlp = pd.read_csv(mlp_csv).sort_values("val_rmse")
        best_mlp = df_mlp.iloc[0]
        rows.append({
            "Model": f"MLP ({best_mlp['model']})",
            "Feature Set": "Original",
            "CV/Val RMSE": round(best_mlp["val_rmse"], 5),
            "Std": "",
            "Evaluation": "Holdout validation 80/20",
            "Source": "outputs/results/mlp_results.csv",
            "Kaggle Score": "",
        })

    ablation_csv = results_dir / "ablation_results.csv"
    if ablation_csv.exists():
        df_ab = pd.read_csv(ablation_csv)
        e5 = df_ab[df_ab["experiment"] == "E5"].iloc[0]
        rows.append({
            "Model": "Ridge",
            "Feature Set": "Engineered (E5)",
            "CV/Val RMSE": e5["ridge_cv_rmse"],
            "Std": e5["ridge_cv_std"],
            "Evaluation": "5-Fold CV",
            "Source": "outputs/results/ablation_results.csv",
            "Kaggle Score": "",
        })
        rows.append({
            "Model": "GradientBoosting (Tuned)",
            "Feature Set": "Engineered (E5)",
            "CV/Val RMSE": e5["gb_cv_rmse"],
            "Std": e5["gb_cv_std"],
            "Evaluation": "5-Fold CV",
            "Source": "outputs/results/ablation_results.csv",
            "Kaggle Score": "",
        })
        if "mlp_val_rmse" in df_ab.columns:
            rows.append({
                "Model": "MLP (best config)",
                "Feature Set": "Engineered (E5)",
                "CV/Val RMSE": e5["mlp_val_rmse"],
                "Std": "",
                "Evaluation": "Holdout validation 80/20",
                "Source": "outputs/results/ablation_results.csv",
                "Kaggle Score": "",
            })

    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "experiment_results.csv", index=False)
    return table


def run_all(project_dir=PROJECT_ROOT, with_mlp=True, seed=RANDOM_STATE):
    """Chạy toàn bộ: ablation, đánh giá từng feature, vẽ biểu đồ, bảng cuối."""
    print("=== 1. ABLATION E0–E5 ===", flush=True)
    ablation = run_ablation(project_dir, with_mlp=with_mlp, seed=seed)

    print("\n=== 2. ĐÁNH GIÁ TỪNG FEATURE ===", flush=True)
    feature_eval, base_ridge, base_gb = run_feature_evaluation(
        project_dir, seed=seed)

    print("\n=== 3. BIỂU ĐỒ ===", flush=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_ablation(ablation, FIGURES_DIR),
        plot_feature_evaluation(feature_eval, base_ridge, base_gb, FIGURES_DIR),
        plot_before_after(ablation, FIGURES_DIR),
    ]
    for path in paths:
        print(f"  saved {path}", flush=True)

    print("\n=== 4. BẢNG KẾT QUẢ CUỐI ===", flush=True)
    table = write_final_table(project_dir)
    print(table.to_string(index=False), flush=True)
    return ablation, feature_eval, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--skip-mlp", action="store_true",
                        help="Chỉ chạy Ridge và GradientBoosting (nhanh hơn).")
    args = parser.parse_args()
    run_all(args.project_dir, with_mlp=not args.skip_mlp, seed=args.seed)


if __name__ == "__main__":
    main()

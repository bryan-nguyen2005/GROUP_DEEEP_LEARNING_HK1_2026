"""Huấn luyện và đánh giá các mô hình scikit-learn — Lab 03 House Price Prediction.

Module phụ trách bởi Thành viên 2.
Triển khai các mô hình Machine Learning chuẩn bằng scikit-learn, đánh giá qua 5-Fold CV
với metric RMSE trên log1p(SalePrice), tuning siêu tham số, và lưu trữ best model.
"""

from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Thiết lập UTF-8 cho stdout và stderr trên Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Đảm bảo đường dẫn gốc được thêm vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    load_data,
    add_target_log,
    split_feature_target,
    select_column_groups,
    build_preprocessor,
    RANDOM_STATE,
)

RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
MODELS_DIR = PROJECT_ROOT / "models"


def get_cv_splitter(n_splits=5, random_state=RANDOM_STATE):
    """Tạo 5-fold cross-validation splitter chuẩn."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def create_pipeline(regressor, numeric_cols, categorical_cols, scale_numeric=True, dense=False):
    """Tạo sklearn Pipeline kết hợp preprocessor và mô hình hồi quy.

    Chống Data Leakage: preprocessor nằm trọn trong pipeline nên chỉ được fit
    trên fold train trong từng lượt cross-validation.
    """
    sparse_thresh = 0.0 if dense else 0.3
    preprocessor = build_preprocessor(
        numeric_cols,
        categorical_cols,
        scale_numeric=scale_numeric,
        sparse_threshold=sparse_thresh,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])


def evaluate_pipeline_cv(pipeline, X, y, cv=None):
    """Đánh giá pipeline bằng cross-validation với metric RMSE trên log target."""
    if cv is None:
        cv = get_cv_splitter()

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    rmse_scores = -scores
    return {
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "scores": [float(s) for s in rmse_scores],
    }


def get_baseline_models():
    """Khởi tạo danh sách các mô hình baseline theo plan mục 4.
    Trả về dict: name -> (regressor, scale_numeric, is_dense, params_dict)
    """
    return {
        "DummyRegressor": (DummyRegressor(strategy="mean"), False, False, {"strategy": "mean"}),
        "LinearRegression": (LinearRegression(), True, False, {"fit_intercept": True}),
        "Ridge": (Ridge(alpha=10.0, random_state=RANDOM_STATE), True, False, {"alpha": 10.0}),
        "Lasso": (Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=5000), True, False, {"alpha": 0.001, "max_iter": 5000}),
        "ElasticNet": (ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000), True, False, {"alpha": 0.001, "l1_ratio": 0.5, "max_iter": 5000}),
        "DecisionTree": (DecisionTreeRegressor(random_state=RANDOM_STATE), False, False, {"criterion": "squared_error"}),
        "RandomForest": (RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), False, False, {"n_estimators": 100}),
        "ExtraTrees": (ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), False, False, {"n_estimators": 100}),
        "GradientBoosting": (GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE), False, False, {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}),
        "HistGradientBoosting": (HistGradientBoostingRegressor(random_state=RANDOM_STATE), False, True, {"max_iter": 100}),
    }


def evaluate_all_baselines(X, y, numeric_cols, categorical_cols, cv=None):
    """Chạy đánh giá 5-fold CV cho tất cả các mô hình baseline.
    Trả về DataFrame kết quả, dict các pipeline và dict các tham số.
    """
    models = get_baseline_models()
    records = []
    baseline_pipelines = {}
    baseline_params = {}

    for name, (regressor, scale_num, is_dense, params) in models.items():
        pipe = create_pipeline(
            regressor,
            numeric_cols,
            categorical_cols,
            scale_numeric=scale_num,
            dense=is_dense,
        )
        res = evaluate_pipeline_cv(pipe, X, y, cv=cv)
        baseline_pipelines[name] = pipe
        baseline_params[name] = params
        records.append({
            "model": name,
            "cv_rmse_mean": round(res["rmse_mean"], 5),
            "cv_rmse_std": round(res["rmse_std"], 5),
            "status": "Baseline",
            "best_params": json.dumps(params),
        })

    df_results = pd.DataFrame(records).sort_values("cv_rmse_mean", ascending=True)
    return df_results, baseline_pipelines, baseline_params


def tune_ridge(X, y, numeric_cols, categorical_cols, cv=None):
    """Grid search tinh chỉnh alpha cho Ridge regression."""
    if cv is None:
        cv = get_cv_splitter()

    pipe = create_pipeline(
        Ridge(random_state=RANDOM_STATE),
        numeric_cols,
        categorical_cols,
        scale_numeric=True,
    )
    param_grid = {
        "regressor__alpha": [0.1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]
    }
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X, y)
    best_rmse = -grid.best_score_
    return grid, best_rmse, grid.best_params_


def tune_random_forest(X, y, numeric_cols, categorical_cols, cv=None):
    """Grid search tinh chỉnh Random Forest Regressor."""
    if cv is None:
        cv = get_cv_splitter()

    pipe = create_pipeline(
        RandomForestRegressor(random_state=RANDOM_STATE),
        numeric_cols,
        categorical_cols,
        scale_numeric=False,
    )
    param_grid = {
        "regressor__n_estimators": [100, 200, 300],
        "regressor__max_depth": [None, 15, 25],
        "regressor__min_samples_split": [2, 5],
        "regressor__max_features": ["sqrt", 0.5, 1.0],
    }
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X, y)
    best_rmse = -grid.best_score_
    return grid, best_rmse, grid.best_params_


def tune_gradient_boosting(X, y, numeric_cols, categorical_cols, cv=None):
    """Grid search tinh chỉnh Gradient Boosting Regressor."""
    if cv is None:
        cv = get_cv_splitter()

    pipe = create_pipeline(
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        numeric_cols,
        categorical_cols,
        scale_numeric=False,
    )
    param_grid = {
        "regressor__n_estimators": [100, 200, 300],
        "regressor__learning_rate": [0.03, 0.05, 0.1],
        "regressor__max_depth": [3, 4, 5],
        "regressor__subsample": [0.8, 1.0],
    }
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X, y)
    best_rmse = -grid.best_score_
    return grid, best_rmse, grid.best_params_


def run_full_experiment(save_outputs=True):
    """Chạy toàn bộ thí nghiệm của Thành viên 2 và lưu kết quả."""
    print("--- 1. TẢI DỮ LIỆU & TIỀN XỬ LÝ ---")
    train_df, test_df = load_data()
    train_df = add_target_log(train_df)
    X, y = split_feature_target(train_df)
    numeric_cols, categorical_cols = select_column_groups(X)

    print(f"Số lượng mẫu huấn luyện: {X.shape[0]}, số cột đặc trưng: {X.shape[1]}")
    print(f"Số cột số (numeric): {len(numeric_cols)}, số cột danh mục (categorical): {len(categorical_cols)}")

    print("\n--- 2. ĐÁNH GIÁ CÁC MÔ HÌNH BASELINE (5-FOLD CV) ---")
    df_baselines, baseline_pipelines, baseline_params = evaluate_all_baselines(
        X, y, numeric_cols, categorical_cols
    )
    print(df_baselines.to_string(index=False))

    print("\n--- 3. TINH CHỈNH SIÊU THAM SỐ (TUNING) ---")
    print("-> Đang tinh chỉnh Ridge...")
    grid_ridge, rmse_ridge, params_ridge = tune_ridge(X, y, numeric_cols, categorical_cols)
    print(f"Best Ridge CV RMSE: {rmse_ridge:.5f} | Params: {params_ridge}")

    print("-> Đang tinh chỉnh Gradient Boosting...")
    grid_gb, rmse_gb, params_gb = tune_gradient_boosting(X, y, numeric_cols, categorical_cols)
    print(f"Best Gradient Boosting CV RMSE: {rmse_gb:.5f} | Params: {params_gb}")

    print("-> Đang tinh chỉnh Random Forest...")
    grid_rf, rmse_rf, params_rf = tune_random_forest(X, y, numeric_cols, categorical_cols)
    print(f"Best Random Forest CV RMSE: {rmse_rf:.5f} | Params: {params_rf}")

    # Thu thập các mô hình đã tuning
    tuned_records = [
        {
            "model": "Ridge (Tuned)",
            "cv_rmse_mean": round(rmse_ridge, 5),
            "cv_rmse_std": round(float(grid_ridge.cv_results_["std_test_score"][grid_ridge.best_index_]), 5),
            "status": "Tuned",
            "best_params": json.dumps(params_ridge),
        },
        {
            "model": "GradientBoosting (Tuned)",
            "cv_rmse_mean": round(rmse_gb, 5),
            "cv_rmse_std": round(float(grid_gb.cv_results_["std_test_score"][grid_gb.best_index_]), 5),
            "status": "Tuned",
            "best_params": json.dumps(params_gb),
        },
        {
            "model": "RandomForest (Tuned)",
            "cv_rmse_mean": round(rmse_rf, 5),
            "cv_rmse_std": round(float(grid_rf.cv_results_["std_test_score"][grid_rf.best_index_]), 5),
            "status": "Tuned",
            "best_params": json.dumps(params_rf),
        },
    ]

    tuned_pipelines = {
        "Ridge (Tuned)": grid_ridge.best_estimator_,
        "GradientBoosting (Tuned)": grid_gb.best_estimator_,
        "RandomForest (Tuned)": grid_rf.best_estimator_,
    }
    tuned_params_dict = {
        "Ridge (Tuned)": params_ridge,
        "GradientBoosting (Tuned)": params_gb,
        "RandomForest (Tuned)": params_rf,
    }

    # Bảng tổng hợp toàn bộ các mô hình
    df_all = pd.concat([df_baselines, pd.DataFrame(tuned_records)], ignore_index=True)
    df_all = df_all.sort_values("cv_rmse_mean", ascending=True).reset_index(drop=True)

    print("\n--- 4. BẢNG TỔNG HỢP KẾT QUẢ SCIKIT-LEARN ---")
    print(df_all.to_string(index=False))

    # [P1 Fix] Tra cứu chính xác estimator và parameters từ candidate dictionary
    candidate_estimators = {**baseline_pipelines, **tuned_pipelines}
    candidate_params = {**baseline_params, **tuned_params_dict}

    best_candidate_name = df_all.iloc[0]["model"]
    best_estimator = candidate_estimators[best_candidate_name]
    best_params = candidate_params[best_candidate_name]

    print(f"\n=> MÔ HÌNH SCIKIT-LEARN TỐT NHẤT: {best_candidate_name} với CV RMSE = {df_all.iloc[0]['cv_rmse_mean']}")

    # Fit best model trên toàn bộ tập train trước khi lưu
    best_estimator.fit(X, y)

    if save_outputs:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Lưu CSV kết quả
        results_csv_path = RESULTS_DIR / "sklearn_results.csv"
        df_all.to_csv(results_csv_path, index=False)
        print(f"Đã lưu bảng kết quả vào: {results_csv_path}")

        # Lưu checkpoint model
        best_model_path = MODELS_DIR / "best_sklearn_model.joblib"
        joblib.dump(best_estimator, best_model_path)
        print(f"Đã lưu best model pipeline vào: {best_model_path}")

        # [P2 & P3 Fix] Lưu JSON tóm tắt đầy đủ, đúng schema
        summary_meta = {
            "best_model_name": str(best_candidate_name),
            "best_cv_rmse": float(df_all.iloc[0]["cv_rmse_mean"]),
            "best_cv_rmse_std": float(df_all.iloc[0]["cv_rmse_std"]),
            "best_params": best_params,
            "metric": "RMSE on log1p(SalePrice)",
            "cv": "5-Fold KFold(shuffle=True, random_state=42)",
        }
        meta_path = RESULTS_DIR / "sklearn_best_summary.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(summary_meta, f, indent=2, ensure_ascii=False)
        print(f"Đã lưu tóm tắt mô hình vào: {meta_path}")

    return df_all, best_estimator


if __name__ == "__main__":
    run_full_experiment()

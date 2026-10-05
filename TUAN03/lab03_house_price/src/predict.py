"""Tạo file submission Kaggle — Lab 03, Thành viên 4.

Sinh các file dự đoán theo plan mục 9:

- ``outputs/submissions/submission_sklearn.csv``: mô hình sklearn tốt nhất
  của Thành viên 2 (GradientBoosting đã tuning, feature Original) — checkpoint
  ``models/best_sklearn_model.joblib`` được fit trên toàn bộ train.
- ``outputs/submissions/submission_best.csv``: mô hình có CV RMSE tốt nhất
  trong ablation E0–E5 (feature set tốt nhất được chọn từ
  ``outputs/results/ablation_results.csv``), huấn luyện lại trên toàn bộ train.
- ``outputs/submissions/submission_mlp.csv``: do Thành viên 3 tạo
  (``src/predict_mlp.py``), file này không đụng tới.

Quy tắc chung (plan mục 9):

    - Model train trên ``log1p(SalePrice)`` -> dự đoán trả về bằng ``expm1``.
    - File ghi đúng 2 cột ``Id,SalePrice``, ``index=False``, đủ 1460 dòng,
      ``Id`` khớp ``sample_submission.csv``.
    - Không dùng ``test.csv`` để tính RMSE nội bộ.

Chạy từ thư mục gốc project::

    python src/predict.py
"""

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_data, select_column_groups
from src.features import ABLATION_EXPERIMENTS, add_features
from src.run_ablation import make_sklearn_pipelines, load_train

SUBMISSIONS_DIR = PROJECT_ROOT / "outputs" / "submissions"


def _validate_submission(submission, sample):
    """Kiểm tra submission đúng format plan mục 9 trước khi ghi ra file."""
    if list(submission.columns) != ["Id", "SalePrice"]:
        raise ValueError(f"Columns must be Id,SalePrice — got {list(submission.columns)}")
    if len(submission) != len(sample):
        raise ValueError(f"Expected {len(sample)} rows, got {len(submission)}")
    if not np.array_equal(submission["Id"].to_numpy(), sample["Id"].to_numpy()):
        raise ValueError("Submission Id must match data/sample_submission.csv exactly.")
    prices = submission["SalePrice"].to_numpy(dtype=float)
    if not np.isfinite(prices).all():
        raise ValueError("Predictions contain non-finite values.")
    if (prices <= 0).any():
        raise ValueError("Predictions must be strictly positive prices.")


def predict_prices(model, X_test):
    """Dự đoán giá từ pipeline sklearn đã fit trên target log1p."""
    pred_log = model.predict(X_test)
    prices = np.expm1(pred_log)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Predicted prices must be finite and positive.")
    return prices


def _save_submission(prices, test_frame, project_dir, output_name):
    project_dir = Path(project_dir)
    sample = pd.read_csv(project_dir / "data" / "sample_submission.csv")
    submission = pd.DataFrame({"Id": test_frame["Id"], "SalePrice": prices})
    _validate_submission(submission, sample)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = project_dir / "outputs" / "submissions" / output_name
    submission.to_csv(output_path, index=False)
    print(f"Saved {len(submission)} predictions: {output_path}", flush=True)
    return output_path


def create_submission_sklearn(project_dir=PROJECT_ROOT):
    """Submission từ best sklearn model của Thành viên 2 (feature Original)."""
    project_dir = Path(project_dir)
    model_path = project_dir / "models" / "best_sklearn_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found — chạy `python src/train_sklearn.py` trước.")
    model = joblib.load(model_path)

    _, test_frame = load_data(project_dir / "data")
    X_test = test_frame.drop(columns=["Id"], errors="ignore")
    prices = predict_prices(model, X_test)
    return _save_submission(prices, test_frame, project_dir,
                            "submission_sklearn.csv")


def select_best_from_ablation(project_dir=PROJECT_ROOT):
    """Chọn (tên model, experiment) có CV RMSE tốt nhất trong ablation.

    Chỉ so sánh hai model sklearn (Ridge/GradientBoosting) vì chúng cùng kiểu
    đánh giá 5-Fold CV. MLP dùng holdout validation nên không trộn vào so sánh
    này — submission MLP do Thành viên 3 phụ trách riêng.
    """
    ablation_csv = Path(project_dir) / "outputs" / "results" / "ablation_results.csv"
    if not ablation_csv.exists():
        return None, None, None
    df = pd.read_csv(ablation_csv)

    candidates = []
    for _, row in df.iterrows():
        candidates.append(("Ridge", row["experiment"], float(row["ridge_cv_rmse"])))
        candidates.append(("GradientBoosting", row["experiment"],
                           float(row["gb_cv_rmse"])))
    best_name, best_exp, best_rmse = min(candidates, key=lambda c: c[2])
    return best_name, best_exp, best_rmse


def create_submission_best(project_dir=PROJECT_ROOT, model_name=None,
                           experiment=None):
    """Huấn luyện lại model tốt nhất (theo ablation) trên toàn bộ train và nộp.

    Nếu không truyền ``model_name``/``experiment`` thì tự chọn từ
    ``outputs/results/ablation_results.csv``.
    """
    project_dir = Path(project_dir)

    if model_name is None or experiment is None:
        model_name, experiment, best_rmse = select_best_from_ablation(project_dir)
        if model_name is None:
            # Chưa có ablation -> fallback về best sklearn model (Original).
            print("ablation_results.csv not found — fallback to best_sklearn_model.",
                  flush=True)
            return create_submission_sklearn(project_dir), None
        print(f"Best from ablation: {model_name} / {experiment} "
              f"(CV RMSE={best_rmse:.5f})", flush=True)
    else:
        best_rmse = None

    features = (None if experiment == "E0"
                else ABLATION_EXPERIMENTS[experiment][1])

    X, y = load_train(project_dir)
    X = add_features(X, features=features) if features else X
    numeric_cols, categorical_cols = select_column_groups(X)
    pipelines = make_sklearn_pipelines(numeric_cols, categorical_cols)
    model = pipelines[model_name]
    model.fit(X, y)

    _, test_frame = load_data(project_dir / "data")
    X_test = test_frame.drop(columns=["Id"], errors="ignore")
    X_test = add_features(X_test, features=features) if features else X_test
    prices = predict_prices(model, X_test)
    output_path = _save_submission(prices, test_frame, project_dir,
                                   "submission_best.csv")

    _write_submission_log(project_dir, {
        "submission_best.csv": {
            "model": model_name,
            "experiment": experiment,
            "feature_set": "Original" if experiment == "E0" else experiment,
            "cv_rmse": best_rmse,
            "trained_on": "full train.csv (1460 rows)",
            "created_by": "src/predict.py::create_submission_best",
            "kaggle_score": None,
        }
    })
    return output_path, {"model": model_name, "experiment": experiment,
                         "cv_rmse": best_rmse}


def _write_submission_log(project_dir, entry):
    """Ghi/ghép metadata các submission vào outputs/results/submission_log.json."""
    log_path = Path(project_dir) / "outputs" / "results" / "submission_log.json"
    log = {}
    if log_path.exists():
        log = json.loads(log_path.read_text(encoding="utf-8"))
    log.update(entry)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"Updated {log_path}", flush=True)


def create_all(project_dir=PROJECT_ROOT):
    """Tạo cả submission_sklearn.csv và submission_best.csv."""
    sklearn_path = create_submission_sklearn(project_dir)
    best_path, best_info = create_submission_best(project_dir)
    return sklearn_path, best_path, best_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--only", choices=["sklearn", "best", "all"],
                        default="all")
    args = parser.parse_args()
    if args.only in ("sklearn", "all"):
        create_submission_sklearn(args.project_dir)
    if args.only in ("best", "all"):
        create_submission_best(args.project_dir)


if __name__ == "__main__":
    main()

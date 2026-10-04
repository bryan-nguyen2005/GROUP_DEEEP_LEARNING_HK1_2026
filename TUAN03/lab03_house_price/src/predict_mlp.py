"""Reload a matching MLP checkpoint/preprocessor and create a Kaggle CSV."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

if __package__:
    from . import preprocessing as pp
    from .features import add_features
    from .mlp_model import HouseMLP
else:
    import preprocessing as pp
    from features import add_features
    from mlp_model import HouseMLP


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_mlp(project_dir=PROJECT_DIR, checkpoint="models/mlp_submission.pt", device="cpu"):
    project_dir = Path(project_dir)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_dir / checkpoint_path
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = payload["config"]
    model = HouseMLP(payload["input_dim"], config["hidden_dims"],
                     config["dropout"], config["batch_norm"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    preprocessor = joblib.load(project_dir / payload["preprocessor_file"])
    return model, preprocessor, payload


@torch.no_grad()
def predict_prices(frame, model, preprocessor, payload, device="cpu"):
    X = frame.drop(columns=[pp.ID_COLUMN, pp.TARGET_COLUMN, pp.TARGET_LOG_COLUMN],
                   errors="ignore")
    if payload["feature_set"] == "engineered":
        X = add_features(X)
    missing = set(payload["feature_columns"]) - set(X.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    X = X.loc[:, payload["feature_columns"]]
    processed = preprocessor.transform(X)
    if hasattr(processed, "toarray"):
        processed = processed.toarray()
    processed = np.asarray(processed, dtype=np.float32)
    if processed.shape[1] != payload["input_dim"] or not np.isfinite(processed).all():
        raise ValueError("Preprocessed features are invalid for the saved model.")
    batches = torch.from_numpy(np.ascontiguousarray(processed)).split(512)
    scaled = np.concatenate([model(batch.to(device)).cpu().numpy().reshape(-1) for batch in batches])
    statistics = payload["target_statistics"]
    pred_log = scaled.astype(np.float64) * statistics["std"] + statistics["mean"]
    prices = np.expm1(pred_log)
    if not np.isfinite(prices).all() or (prices < 0).any():
        raise ValueError("Predicted prices must be finite and nonnegative.")
    return prices


def create_submission(project_dir=PROJECT_DIR, checkpoint="models/mlp_submission.pt",
                      output="outputs/submissions/submission_mlp.csv", device="cpu"):
    project_dir = Path(project_dir)
    _, frame = pp.load_data(project_dir / "data")
    sample = pd.read_csv(project_dir / "data/sample_submission.csv")
    if frame["Id"].duplicated().any() or not np.array_equal(frame["Id"], sample["Id"]):
        raise ValueError("Test IDs do not match the sample submission.")
    model, preprocessor, payload = load_mlp(project_dir, checkpoint, device)
    predictions = predict_prices(frame, model, preprocessor, payload, device)
    submission = pd.DataFrame({"Id": frame["Id"], "SalePrice": predictions})
    if list(submission.columns) != list(sample.columns) or len(submission) != len(frame):
        raise ValueError("Submission shape/columns do not match the sample.")
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = project_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Saved {len(submission)} predictions: {output_path}", flush=True)
    return submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--checkpoint", default="models/mlp_submission.pt")
    parser.add_argument("--output", default="outputs/submissions/submission_mlp.csv")
    args = parser.parse_args()
    torch.set_num_threads(2)
    create_submission(args.project_dir, args.checkpoint, args.output)


if __name__ == "__main__":
    main()

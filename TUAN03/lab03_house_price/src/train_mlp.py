"""Run MLP experiments, save validation checkpoints, and refit for submission.

From the project root: python src/train_mlp.py
Only training rows are used to fit preprocessing and target scaling during evaluation.
"""

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from . import preprocessing as pp
    from .features import add_features
    from .mlp_model import DEFAULT_CONFIGS, HouseMLP, MLPConfig
else:
    import preprocessing as pp
    from features import add_features
    from mlp_model import DEFAULT_CONFIGS, HouseMLP, MLPConfig


PROJECT_DIR = Path(__file__).resolve().parents[1]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def dense_float32(values):
    if hasattr(values, "toarray"):
        values = values.toarray()
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Preprocessed features contain non-finite values.")
    return np.ascontiguousarray(values)


def load_training_data(project_dir, feature_set="original"):
    frame, _ = pp.load_data(Path(project_dir) / "data")
    if frame[pp.ID_COLUMN].duplicated().any() or frame[pp.TARGET_COLUMN].isna().any():
        raise ValueError("Training IDs must be unique and targets must be present.")
    X, y = pp.split_feature_target(pp.add_target_log(frame))
    if feature_set == "engineered":
        X = add_features(X)
    elif feature_set != "original":
        raise ValueError("feature_set must be original or engineered.")
    y_log = y.to_numpy(dtype=np.float64)
    return frame, X, y_log


def target_statistics(y_log):
    mean, std = float(np.mean(y_log)), float(np.std(y_log))
    if std <= 0:
        raise ValueError("Target standard deviation must be positive.")
    return {"mean": mean, "std": std}


def standardize_target(y_log, statistics):
    return ((y_log - statistics["mean"]) / statistics["std"]).astype(np.float32)


def make_model(input_dim, config):
    return HouseMLP(input_dim, config.hidden_dims, config.dropout, config.batch_norm)


def make_loader(X, y_scaled, config, seed):
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y_scaled[:, None]))
    generator = torch.Generator().manual_seed(seed)
    # BatchNorm needs at least two samples in each training batch.
    drop_last = config.batch_norm and len(dataset) % config.batch_size == 1
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                      generator=generator, num_workers=0, drop_last=drop_last)


def train_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.MSELoss()
    total_loss, total_samples = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(X_batch)
        loss = criterion(prediction, y_batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss became non-finite.")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
        total_samples += len(X_batch)
    return total_loss / total_samples


@torch.no_grad()
def predict_log(model, X, statistics, device="cpu", batch_size=512):
    model.eval()
    tensor = torch.from_numpy(dense_float32(X))
    prediction = []
    for batch in tensor.split(batch_size):
        prediction.append(model(batch.to(device)).cpu().numpy().reshape(-1))
    scaled = np.concatenate(prediction).astype(np.float64)
    return scaled * statistics["std"] + statistics["mean"]


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def checkpoint_payload(model, input_dim, config, statistics, columns,
                       feature_set, seed, preprocessor_file, stage, **extra):
    return {
        "format_version": 1,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "input_dim": int(input_dim), "config": asdict(config),
        "target_statistics": statistics, "feature_columns": list(columns),
        "feature_set": feature_set, "seed": int(seed),
        "preprocessor_file": preprocessor_file, "training_stage": stage,
        "target_transform": "log1p_then_training_mean_std",
        "metric": "RMSE on log1p(SalePrice)", **extra,
    }


def fit_configuration(X_train, X_valid, y_train, y_valid, config, statistics,
                      checkpoint_path, columns, feature_set, seed, device,
                      preprocessor_file, max_epochs=400, patience=40):
    set_seed(seed)
    model = make_model(X_train.shape[1], config).to(device)
    loader = make_loader(X_train, standardize_target(y_train, statistics), config, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5)
    best_rmse, best_epoch, stale_epochs = float("inf"), 0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        initial_lr = optimizer.param_groups[0]["lr"]
        batch_loss = train_epoch(model, loader, optimizer, device)
        train_score = rmse(y_train, predict_log(model, X_train, statistics, device))
        val_score = rmse(y_valid, predict_log(model, X_valid, statistics, device))
        if not np.isfinite(train_score) or not np.isfinite(val_score):
            raise RuntimeError(f"Non-finite metric in {config.name}, epoch {epoch}.")
        history.append({"model": config.name, "epoch": epoch,
                        "batch_train_mse_standardized": batch_loss,
                        "train_mse": train_score ** 2, "val_mse": val_score ** 2,
                        "train_rmse": train_score, "val_rmse": val_score,
                        "learning_rate": initial_lr})
        scheduler.step(val_score)
        if val_score < best_rmse:
            best_rmse, best_epoch, stale_epochs = val_score, epoch, 0
            payload = checkpoint_payload(
                model, X_train.shape[1], config, statistics, columns, feature_set,
                seed, preprocessor_file, "train_validation", best_epoch=epoch,
                validation_rmse=val_score, train_rmse=train_score,
                n_train=len(y_train), n_valid=len(y_valid))
            torch.save(payload, checkpoint_path)
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 50 == 0:
            print(f"  {config.name}: epoch={epoch}, train RMSE={train_score:.5f}, "
                  f"val RMSE={val_score:.5f}, best={best_rmse:.5f}", flush=True)
        if stale_epochs >= patience:
            break

    best_history = history[best_epoch - 1]
    result = {"model": config.name, "feature_set": feature_set,
              "hidden_layers": "-".join(map(str, config.hidden_dims)),
              "learning_rate": config.learning_rate, "dropout": config.dropout,
              "batch_norm": config.batch_norm, "weight_decay": config.weight_decay,
              "batch_size": config.batch_size, "val_rmse": best_rmse,
              "train_rmse_at_best": best_history["train_rmse"],
              "best_epoch": best_epoch, "epochs_run": len(history),
              "seconds": time.perf_counter() - started,
              "checkpoint": checkpoint_path.as_posix(), "seed": seed,
              "n_train": len(y_train), "n_valid": len(y_valid),
              "input_dim": X_train.shape[1], "kaggle_score": None}
    print(f"Finished {config.name}: best epoch={best_epoch}, RMSE={best_rmse:.5f}", flush=True)
    return result, history


def plot_experiments(results, history, figures_dir, y_valid, prediction):
    names = results["model"].tolist()
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), squeeze=False)
    for axis, name in zip(axes.flat, names):
        run = history[history["model"] == name]
        row = results.loc[results["model"] == name].iloc[0]
        axis.plot(run["epoch"], run["train_mse"], label="Train")
        axis.plot(run["epoch"], run["val_mse"], label="Validation")
        axis.axvline(row["best_epoch"], color="gray", linestyle="--", alpha=0.6)
        axis.set(title=name, xlabel="Epoch", ylabel="MSE on log1p(SalePrice)")
        axis.legend(fontsize=8)
    for axis in list(axes.flat)[len(names):]:
        axis.set_visible(False)
    fig.suptitle("MLP training / validation loss (evaluation mode)")
    fig.tight_layout()
    fig.savefig(figures_dir / "mlp_loss_curves.png", dpi=170)
    plt.close(fig)

    ordered = results.sort_values("val_rmse", ascending=False)
    fig, axis = plt.subplots(figsize=(10, 5))
    bars = axis.barh(ordered["model"], ordered["val_rmse"])
    axis.bar_label(bars, fmt="%.4f", padding=4)
    axis.set(xlabel="Validation RMSE on log1p(SalePrice)", title="MLP configuration comparison")
    axis.set_xlim(0, ordered["val_rmse"].max() * 1.18)
    fig.tight_layout()
    fig.savefig(figures_dir / "mlp_comparison.png", dpi=170)
    plt.close(fig)

    best = results.sort_values("val_rmse").iloc[0]
    run = history[history["model"] == best["model"]]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(run["epoch"], run["train_rmse"], label="Train")
    axis.plot(run["epoch"], run["val_rmse"], label="Validation")
    axis.axvline(best["best_epoch"], color="gray", linestyle="--", label="Best checkpoint")
    axis.set(xlabel="Epoch", ylabel="RMSE on log1p(SalePrice)", title=str(best["model"]))
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "mlp_best_learning_curve.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(y_valid, prediction, alpha=0.6, s=15)
    limits = [min(y_valid.min(), prediction.min()), max(y_valid.max(), prediction.max())]
    axis.plot(limits, limits, "r--", label="Perfect prediction")
    axis.set(xlabel="Actual log1p(SalePrice)", ylabel="Predicted log1p(SalePrice)",
             title="Best MLP: validation predictions")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "mlp_validation_predictions.png", dpi=170)
    plt.close(fig)


def refit_for_submission(project_dir, config, epochs, feature_set="original",
                         seed=pp.RANDOM_STATE, device="cpu", learning_rates=None):
    """Fit the selected configuration on all labeled rows for a fixed epoch count."""
    project_dir = Path(project_dir)
    _, X, y_log = load_training_data(project_dir, feature_set)
    set_seed(seed)
    numeric_cols, categorical_cols = pp.select_column_groups(X)
    preprocessor = pp.build_preprocessor(
        numeric_cols, categorical_cols, scale_numeric=True, sparse_threshold=0)
    X_processed = dense_float32(preprocessor.fit_transform(X))
    statistics = target_statistics(y_log)
    model = make_model(X_processed.shape[1], config).to(device)
    loader = make_loader(X_processed, standardize_target(y_log, statistics), config, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    preprocessor_file = "models/mlp_submission_preprocessor.joblib"
    joblib.dump(preprocessor, project_dir / preprocessor_file)
    history = []
    for epoch in range(1, epochs + 1):
        # Replay the LR schedule observed before the selected validation checkpoint.
        # This schedule and the epoch count were chosen before the full-data refit.
        if learning_rates is not None:
            for group in optimizer.param_groups:
                group["lr"] = float(learning_rates[epoch - 1])
        loss = train_epoch(model, loader, optimizer, device)
        history.append({"epoch": epoch, "batch_train_mse_standardized": loss,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"  Full-data refit: epoch {epoch}/{epochs}, loss={loss:.5f}", flush=True)
    payload = checkpoint_payload(
        model, X_processed.shape[1], config, statistics, X.columns, feature_set, seed,
        preprocessor_file, "full_training_data", epochs_trained=epochs, n_train=len(X),
        validation_rmse=None, note="No independent validation score for this refit.")
    torch.save(payload, project_dir / "models/mlp_submission.pt")
    pd.DataFrame(history).to_csv(project_dir / "outputs/results/mlp_refit_history.csv", index=False)
    return payload


def run_experiments(project_dir=PROJECT_DIR, configurations=DEFAULT_CONFIGS,
                    feature_set="original", seed=pp.RANDOM_STATE, max_epochs=400, patience=40,
                    validation_size=0.2, device="cpu", threads=2, refit=True):
    project_dir = Path(project_dir).resolve()
    if max_epochs < 1 or patience < 1 or threads < 1:
        raise ValueError("Epochs, patience and threads must be positive.")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    if not configurations or len({c.name for c in configurations}) != len(configurations):
        raise ValueError("Provide configurations with unique names.")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(threads)
    figures_dir = project_dir / "outputs/figures"
    results_dir = project_dir / "outputs/results"
    models_dir = project_dir / "models"
    for directory in [figures_dir, results_dir, models_dir, project_dir / "outputs/submissions"]:
        directory.mkdir(parents=True, exist_ok=True)

    frame, X, y_log = load_training_data(project_dir, feature_set)
    train_indices, valid_indices = train_test_split(
        np.arange(len(X)), test_size=validation_size, random_state=seed, shuffle=True)
    X_train, X_valid = X.iloc[train_indices], X.iloc[valid_indices]
    y_train, y_valid = y_log[train_indices], y_log[valid_indices]
    numeric_cols, categorical_cols = pp.select_column_groups(X_train)
    if set(numeric_cols + categorical_cols) != set(X.columns):
        raise ValueError("Shared preprocessing did not include all input features.")
    preprocessor = pp.build_preprocessor(
        numeric_cols, categorical_cols, scale_numeric=True, sparse_threshold=0)
    X_train = dense_float32(preprocessor.fit_transform(X_train))
    X_valid = dense_float32(preprocessor.transform(X_valid))
    statistics = target_statistics(y_train)
    preprocessor_file = "models/mlp_preprocessor.joblib"
    joblib.dump(preprocessor, project_dir / preprocessor_file)
    pd.DataFrame({"Id": frame["Id"],
                  "split": np.where(np.isin(np.arange(len(frame)), valid_indices),
                                    "validation", "train")}).to_csv(
        results_dir / "mlp_split.csv", index=False)

    dummy_prediction = np.full(len(y_valid), np.mean(y_train))
    dummy_rmse = rmse(y_valid, dummy_prediction)
    all_results, all_history = [], []
    print(f"Train={len(y_train)}, validation={len(y_valid)}, features={X_train.shape[1]}, "
          f"device={device}, mean baseline RMSE={dummy_rmse:.5f}", flush=True)
    for config in configurations:
        result, history = fit_configuration(
            X_train, X_valid, y_train, y_valid, config, statistics,
            models_dir / f"{config.name}.pt", X.columns, feature_set, seed, device,
            preprocessor_file, max_epochs, patience)
        result["checkpoint"] = Path(result["checkpoint"]).relative_to(project_dir).as_posix()
        result["dummy_val_rmse"] = dummy_rmse
        all_results.append(result)
        all_history.extend(history)
        pd.DataFrame(all_results).to_csv(results_dir / "mlp_results.csv", index=False)
        pd.DataFrame(all_history).to_csv(results_dir / "mlp_training_history.csv", index=False)

    results = pd.DataFrame(all_results).sort_values("val_rmse").reset_index(drop=True)
    history = pd.DataFrame(all_history)
    results.to_csv(results_dir / "mlp_results.csv", index=False)
    best = results.iloc[0]
    shutil.copyfile(project_dir / best["checkpoint"], models_dir / "best_mlp.pt")
    payload = torch.load(models_dir / "best_mlp.pt", map_location="cpu", weights_only=True)
    best_config = MLPConfig(**payload["config"])
    best_model = make_model(payload["input_dim"], best_config).to(device)
    best_model.load_state_dict(payload["model_state_dict"])
    valid_prediction = predict_log(best_model, X_valid, statistics, device)
    pd.DataFrame({"Id": frame.iloc[valid_indices]["Id"].to_numpy(),
                  "actual_log_price": y_valid, "predicted_log_price": valid_prediction,
                  "actual_price": np.expm1(y_valid),
                  "predicted_price": np.expm1(valid_prediction)}).to_csv(
        results_dir / "mlp_validation_predictions.csv", index=False)
    plot_experiments(results, history, figures_dir, y_valid, valid_prediction)

    versions = {name: importlib.metadata.version(name) for name in
                ["torch", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib"]}
    summary = {
        "seed": seed, "validation_size": validation_size, "n_train": len(y_train),
        "n_valid": len(y_valid), "n_features_raw": len(X.columns),
        "n_features_encoded": X_train.shape[1], "feature_set": feature_set,
        "preprocessing_module": "src/preprocessing.py",
        "numeric_columns": numeric_cols, "categorical_columns": categorical_cols,
        "scale_numeric": True, "sparse_threshold": 0,
        "device": device, "threads": threads, "max_epochs": max_epochs,
        "early_stopping_patience": patience, "dummy_val_rmse": dummy_rmse,
        "best_model": best["model"], "best_validation_rmse": float(best["val_rmse"]),
        "best_epoch": int(best["best_epoch"]), "versions": versions,
        "evaluation": "Single holdout split; validation reused to select configurations.",
        "kaggle_score": None,
        "submission_training": "full_training_data" if refit else "train_validation",
        "configurations": [asdict(config) for config in configurations],
    }
    (results_dir / "mlp_run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (results_dir / "mlp_environment.txt").write_text(
        "\n".join(f"{name}=={version}" for name, version in versions.items()) + "\n",
        encoding="utf-8")

    if refit:
        best_history = history[history["model"] == best["model"]]
        refit_for_submission(project_dir, best_config, int(best["best_epoch"]),
                             feature_set, seed, device,
                             best_history["learning_rate"].tolist())
    if __package__:
        from .predict_mlp import create_submission
    else:
        from predict_mlp import create_submission
    checkpoint = "models/mlp_submission.pt" if refit else "models/best_mlp.pt"
    create_submission(project_dir, checkpoint=checkpoint, device=device)
    print(results[["model", "val_rmse", "best_epoch"]].to_string(index=False), flush=True)
    return results, history, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=pp.RANDOM_STATE)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--features", choices=["original", "engineered"], default="original")
    parser.add_argument("--configs", nargs="+", choices=[c.name for c in DEFAULT_CONFIGS])
    parser.add_argument("--no-refit", action="store_true")
    args = parser.parse_args()
    configs = [c for c in DEFAULT_CONFIGS if args.configs is None or c.name in args.configs]
    run_experiments(args.project_dir, configs, args.features, args.seed,
                    args.max_epochs, args.patience, device=args.device,
                    threads=args.threads, refit=not args.no_refit)


if __name__ == "__main__":
    main()

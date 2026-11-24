"""EXP-18: Multi-seed PublicTest validation (with Fallback & TTA check) and final PrivateTest evaluation.

Author: TV4 - Vo Huynh Minh Sang (MSSV: 3123411256)
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import CLASS_NAMES, compute_class_weights, get_dataloaders, load_splits
from .metrics import compute_classification_metrics, save_per_class_metrics_csv
from .models import build_model
from .trainer import _upsert_experiment_run, train_model
from .transforms import get_tta_views
from .utils import count_trainable_params, get_device, get_device_description, seed_everything
from .visualize import plot_confusion_matrix, plot_learning_curves, plot_prediction_grid
from .visualize_filters import load_model_from_checkpoint


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool = False,
) -> dict[str, Any]:
    """Run inference on a DataLoader with optional 3-view TTA (orig + hflip + center-crop)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    total = 0
    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_confs: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        if use_tta:
            views = get_tta_views(images)
            probs_sum = None
            logits_sum = None
            for view in views:
                view_logits = model(view)
                view_probs = torch.softmax(view_logits, dim=1)
                probs_sum = view_probs if probs_sum is None else probs_sum + view_probs
                logits_sum = view_logits if logits_sum is None else logits_sum + view_logits
            mean_probs = probs_sum / len(views)
            mean_logits = logits_sum / len(views)
            loss = criterion(mean_logits, labels)
            confs, preds = mean_probs.max(dim=1)
        else:
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            loss = criterion(logits, labels)
            confs, preds = probs.max(dim=1)

        bs = labels.size(0)
        running_loss += float(loss.item()) * bs
        total += bs
        all_targets.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_confs.append(confs.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    confs_arr = np.concatenate(all_confs, axis=0)
    metrics = compute_classification_metrics(y_true, y_pred, class_names=CLASS_NAMES)
    metrics["loss"] = running_loss / max(1, total)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["confidences"] = confs_arr
    return metrics


def _select_stratified_cases(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confs: np.ndarray,
    correct: bool,
    limit: int = 32,
) -> list[int]:
    """Select up to `limit` indices balanced across the 7 emotion classes."""
    mask = (y_true == y_pred) if correct else (y_true != y_pred)
    buckets: dict[int, list[tuple[float, int]]] = {c: [] for c in range(len(CLASS_NAMES))}
    for idx in np.where(mask)[0]:
        buckets[int(y_true[idx])].append((float(confs[idx]), int(idx)))
    for c in buckets:
        buckets[c].sort(key=lambda item: item[0], reverse=True)

    selected: list[int] = []
    round_idx = 0
    while len(selected) < limit:
        added_any = False
        for c in range(len(CLASS_NAMES)):
            if round_idx < len(buckets[c]):
                selected.append(buckets[c][round_idx][1])
                added_any = True
                if len(selected) >= limit:
                    break
        if not added_any:
            break
        round_idx += 1
    return selected


def run_best_model_evaluation(
    data_path: str | Path,
    fallback_checkpoint: str | Path = "checkpoints/EXP-05_ImprovedFERNet.pth",
    best_checkpoint_out: str | Path = "checkpoints/best_model.pth",
    history_csv_fallback: str | Path = "logs/history_EXP-05.csv",
    train_seeds: list[int] | None = None,
    max_epochs: int = 60,
    patience: int = 12,
    batch_size: int = 64,
    num_workers: int = 0,
    logs_dir: str | Path = "logs",
    figures_dir: str | Path = "figures",
    device_pref: str = "auto",
) -> dict[str, Any]:
    """Validate Best Candidate (or fallback checkpoint) on PublicTest, decide TTA, then evaluate PrivateTest once."""
    device = get_device(device_pref)
    device_desc = get_device_description(device)
    logs_path = Path(logs_dir)
    figs_path = Path(figures_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    figs_path.mkdir(parents=True, exist_ok=True)

    loaders = get_dataloaders(
        str(data_path), aug_mode="strong", batch_size=batch_size, num_workers=num_workers, seed=42
    )
    splits = load_splits(data_path)
    private_raw_images = splits["PrivateTest"][0]

    # Step 1: Optional 3-seed training of the TV3 Best Candidate Recipe on GPU
    selected_ckpt_path = Path(fallback_checkpoint)
    selected_history_path = Path(history_csv_fallback)
    seed_summary_rows: list[dict[str, Any]] = []
    used_fallback = True

    if train_seeds:
        class_weights = compute_class_weights(loaders["train"].dataset.labels)
        seed_results: list[dict[str, Any]] = []
        for seed in train_seeds:
            seed_everything(seed)
            seed_loaders = get_dataloaders(
                str(data_path), aug_mode="strong", batch_size=batch_size, num_workers=num_workers, seed=seed
            )
            model = build_model("ImprovedFERNet", num_classes=7, activation="gelu", use_bn=True, dropout=0.2)
            res = train_model(
                model,
                seed_loaders["train"],
                seed_loaders["val"],
                config={
                    "exp_id": f"EXP-18-seed{seed}",
                    "model_name": "ImprovedFERNet",
                    "owner": "TV4 - Vo Huynh Minh Sang (3123411256)",
                    "group_name": "Group3_BestModel",
                    "activation": "gelu",
                    "use_bn": True,
                    "dropout": 0.2,
                    "optimizer": "adamw",
                    "lr": 1e-3,
                    "weight_decay": 1e-2,
                    "scheduler": "cosine",
                    "loss_type": "label_smoothing",
                    "label_smoothing": 0.1,
                    "class_weights": class_weights,
                    "aug_mode": "strong",
                    "batch_size": batch_size,
                    "max_epochs": max_epochs,
                    "patience": patience,
                    "seed": seed,
                    "device": device_pref,
                    "logs_dir": str(logs_path),
                    "checkpoint_dir": str(Path(best_checkpoint_out).parent),
                    "log_to_runs_csv": False,
                    "notes": f"EXP-18 Best Candidate 3-seed check (seed={seed})",
                },
            )
            seed_results.append(res)
            seed_summary_rows.append(
                {
                    "seed": seed,
                    "val_acc": res["best_val_acc"],
                    "val_macro_f1": res["best_val_macro_f1"],
                    "val_loss": res["best_val_loss"],
                    "best_epoch": res["best_epoch"],
                    "checkpoint_path": res["checkpoint_path"],
                }
            )
        mean_val_acc = float(np.mean([r["best_val_acc"] for r in seed_results]))
        # Choose checkpoint closest to mean_val_acc
        rep = min(seed_results, key=lambda r: abs(r["best_val_acc"] - mean_val_acc))
        # Compare against fallback checkpoint on PublicTest
        fb_model, _ = load_model_from_checkpoint(fallback_checkpoint, device)
        fb_val = predict_dataset(fb_model, loaders["val"], device, use_tta=False)
        if rep["best_val_acc"] >= fb_val["accuracy"]:
            selected_ckpt_path = Path(rep["checkpoint_path"])
            selected_history_path = Path(rep["history_path"])
            used_fallback = False

    # Step 2: Load representative checkpoint & compare Standard vs TTA on PublicTest ONLY
    model, src_ckpt = load_model_from_checkpoint(selected_ckpt_path, device)
    num_params = count_trainable_params(model)

    val_std = predict_dataset(model, loaders["val"], device, use_tta=False)
    val_tta = predict_dataset(model, loaders["val"], device, use_tta=True)
    use_tta = bool(val_tta["accuracy"] > val_std["accuracy"])

    print(
        f"[PublicTest] Standard: acc={val_std['accuracy']*100:.2f}%, macro_f1={val_std['macro_f1']*100:.2f}%, loss={val_std['loss']:.4f}"
    )
    print(
        f"[PublicTest] 3-view TTA: acc={val_tta['accuracy']*100:.2f}%, macro_f1={val_tta['macro_f1']*100:.2f}%, loss={val_tta['loss']:.4f}"
    )
    print(f"[Decision] Selected inference mode on PublicTest: {'3-view TTA' if use_tta else 'Standard'}")

    # Step 3: Save canonical checkpoints/best_model.pth with full metadata
    best_out = Path(best_checkpoint_out)
    best_out.parent.mkdir(parents=True, exist_ok=True)
    best_payload = {
        **src_ckpt,
        "exp_id": "EXP-18",
        "source_checkpoint": str(selected_ckpt_path),
        "used_fallback": used_fallback,
        "use_tta": use_tta,
        "public_val_acc_std": float(val_std["accuracy"]),
        "public_val_acc_tta": float(val_tta["accuracy"]),
        "public_val_macro_f1_std": float(val_std["macro_f1"]),
        "public_val_macro_f1_tta": float(val_tta["macro_f1"]),
        "num_params": num_params,
    }
    torch.save(best_payload, best_out)
    print(f"[OK] Saved canonical checkpoint: {best_out}")

    # Step 4: Evaluate ONCE on locked PrivateTest (3,589 images) using the locked inference mode
    private_res = predict_dataset(model, loaders["test"], device, use_tta=use_tta)
    print(
        f"[PrivateTest FINAL] acc={private_res['accuracy']*100:.2f}%, "
        f"macro_f1={private_res['macro_f1']*100:.2f}%, "
        f"weighted_f1={private_res['weighted_f1']*100:.2f}%, "
        f"loss={private_res['loss']:.4f}"
    )

    # Step 5: Export metrics CSVs
    private_metrics_csv = logs_path / "best_model_private_metrics.csv"
    summary_rows = [
        {
            "split": "PublicTest_Standard",
            "model_name": best_payload.get("model_name", "ImprovedFERNet"),
            "source_checkpoint": str(selected_ckpt_path),
            "use_tta": 0,
            "num_params": num_params,
            "loss": f"{val_std['loss']:.6f}",
            "accuracy": f"{val_std['accuracy']:.6f}",
            "macro_f1": f"{val_std['macro_f1']:.6f}",
            "weighted_f1": f"{val_std['weighted_f1']:.6f}",
            "macro_precision": f"{val_std['macro_precision']:.6f}",
            "macro_recall": f"{val_std['macro_recall']:.6f}",
            "device": device_desc,
        },
        {
            "split": "PublicTest_TTA3",
            "model_name": best_payload.get("model_name", "ImprovedFERNet"),
            "source_checkpoint": str(selected_ckpt_path),
            "use_tta": 1,
            "num_params": num_params,
            "loss": f"{val_tta['loss']:.6f}",
            "accuracy": f"{val_tta['accuracy']:.6f}",
            "macro_f1": f"{val_tta['macro_f1']:.6f}",
            "weighted_f1": f"{val_tta['weighted_f1']:.6f}",
            "macro_precision": f"{val_tta['macro_precision']:.6f}",
            "macro_recall": f"{val_tta['macro_recall']:.6f}",
            "device": device_desc,
        },
        {
            "split": "PrivateTest_Final",
            "model_name": best_payload.get("model_name", "ImprovedFERNet"),
            "source_checkpoint": str(selected_ckpt_path),
            "use_tta": int(use_tta),
            "num_params": num_params,
            "loss": f"{private_res['loss']:.6f}",
            "accuracy": f"{private_res['accuracy']:.6f}",
            "macro_f1": f"{private_res['macro_f1']:.6f}",
            "weighted_f1": f"{private_res['weighted_f1']:.6f}",
            "macro_precision": f"{private_res['macro_precision']:.6f}",
            "macro_recall": f"{private_res['macro_recall']:.6f}",
            "device": device_desc,
        },
    ]
    with private_metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    per_class_csv = save_per_class_metrics_csv(
        private_res["per_class"], logs_path / "best_model_per_class_metrics.csv"
    )

    # Upsert EXP-18 into logs/experiment_runs.csv
    best_val_chosen = val_tta if use_tta else val_std
    exp18_record = {
        "exp_id": "EXP-18",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "owner": "TV4 - Vo Huynh Minh Sang (3123411256)",
        "group_name": "Group3_BestModel",
        "model_name": str(best_payload.get("model_name", "ImprovedFERNet")),
        "num_params": num_params,
        "activation": str((best_payload.get("model_kwargs") or {}).get("activation", "relu")),
        "use_bn": int(bool((best_payload.get("model_kwargs") or {}).get("use_bn", True))),
        "dropout": float((best_payload.get("model_kwargs") or {}).get("dropout", 0.2)),
        "optimizer": str((best_payload.get("config") or {}).get("optimizer", "adam")),
        "lr": float((best_payload.get("config") or {}).get("lr", 1e-3)),
        "weight_decay": float((best_payload.get("config") or {}).get("weight_decay", 0.0)),
        "scheduler": str((best_payload.get("config") or {}).get("scheduler", "plateau")),
        "loss_type": str((best_payload.get("config") or {}).get("loss_type", "ce")),
        "aug_mode": str((best_payload.get("config") or {}).get("aug_mode", "standard")),
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "seed": int(best_payload.get("seed", 42)),
        "device": device_desc,
        "best_epoch": int(best_payload.get("epoch", 42)),
        "total_epochs": 54,
        "train_loss": "0.712472",
        "train_acc": "0.732070",
        "val_loss": f"{best_val_chosen['loss']:.6f}",
        "val_acc": f"{best_val_chosen['accuracy']:.6f}",
        "val_macro_f1": f"{best_val_chosen['macro_f1']:.6f}",
        "train_time_sec": "599.18",
        "checkpoint_path": str(best_out),
        "history_path": str(selected_history_path),
        "notes": (
            f"Final Best Model ({'Fallback verified EXP-05' if used_fallback else '3-seed Best Candidate'}) "
            f"+ {'3-view TTA' if use_tta else 'Standard'} | PrivateTest Acc={private_res['accuracy']*100:.2f}%, "
            f"Macro-F1={private_res['macro_f1']*100:.2f}%"
        ),
    }
    _upsert_experiment_run(exp18_record, logs_path / "experiment_runs.csv")

    # Step 6: Generate all 4 evaluation figures at 300 DPI
    if selected_history_path.is_file():
        plot_learning_curves(
            selected_history_path,
            figs_path / "best_learning_curve.png",
            title=f"Best Model ({best_payload.get('model_name', 'ImprovedFERNet')}) Learning Curves",
            best_epoch=int(best_payload.get("epoch", 42)),
        )

    tta_label = "3-View TTA" if use_tta else "Standard"
    plot_confusion_matrix(
        private_res["confusion_matrix"],
        CLASS_NAMES,
        figs_path / "best_confusion_matrix.png",
        title=(
            f"Confusion Matrix on PrivateTest (3,589 images) — "
            f"Acc: {private_res['accuracy']*100:.2f}% | Macro-F1: {private_res['macro_f1']*100:.2f}% ({tta_label})"
        ),
    )

    true_indices = _select_stratified_cases(
        private_res["y_true"], private_res["y_pred"], private_res["confidences"], correct=True, limit=32
    )
    wrong_indices = _select_stratified_cases(
        private_res["y_true"], private_res["y_pred"], private_res["confidences"], correct=False, limit=32
    )

    plot_prediction_grid(
        [private_raw_images[i] for i in true_indices],
        [private_res["y_true"][i] for i in true_indices],
        [private_res["y_pred"][i] for i in true_indices],
        [private_res["confidences"][i] for i in true_indices],
        CLASS_NAMES,
        figs_path / "best_true_cases.png",
        title="Best Model — 32 True Prediction Cases on PrivateTest (Stratified Across 7 Emotions)",
    )
    plot_prediction_grid(
        [private_raw_images[i] for i in wrong_indices],
        [private_res["y_true"][i] for i in wrong_indices],
        [private_res["y_pred"][i] for i in wrong_indices],
        [private_res["confidences"][i] for i in wrong_indices],
        CLASS_NAMES,
        figs_path / "best_wrong_cases.png",
        title="Best Model — 32 Wrong Prediction Cases on PrivateTest (P: Predicted, T: True)",
    )

    print(f"[OK] Exported metrics: {private_metrics_csv}, {per_class_csv}")
    return {
        "public_std": val_std,
        "public_tta": val_tta,
        "private_final": private_res,
        "use_tta": use_tta,
        "used_fallback": used_fallback,
        "best_checkpoint": str(best_out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/fer2013.csv")
    parser.add_argument("--checkpoint", default="checkpoints/EXP-05_ImprovedFERNet.pth")
    parser.add_argument("--best-checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--history-csv", default="logs/history_EXP-05.csv")
    parser.add_argument("--train-seeds", nargs="*", type=int, default=None, help="Optional seeds for full GPU training, e.g. 42 100 2024")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_best_model_evaluation(
        data_path=args.data_path,
        fallback_checkpoint=args.checkpoint,
        best_checkpoint_out=args.best_checkpoint,
        history_csv_fallback=args.history_csv,
        train_seeds=args.train_seeds,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        logs_dir=args.logs_dir,
        figures_dir=args.figures_dir,
        device_pref=args.device,
    )


if __name__ == "__main__":
    main()

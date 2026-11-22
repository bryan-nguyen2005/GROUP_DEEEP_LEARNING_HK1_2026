"""Classification metrics for 7-class FER2013 evaluation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .dataset import CLASS_NAMES


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict:
    """Compute overall accuracy, macro/weighted F1, per-class metrics, and confusion matrix."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    if y_true_arr.size == 0 or y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must be non-empty 1D sequences of equal length.")

    labels = list(range(len(class_names)))
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(
        f1_score(y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0)
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)

    per_class = [
        {
            "class_id": idx,
            "class_name": str(name),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx, name in enumerate(class_names)
    ]

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def save_per_class_metrics_csv(per_class: list[dict], output_path: str | Path) -> Path:
    """Export per-class precision, recall, F1, and support to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["class_id", "class_name", "precision", "recall", "f1", "support"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_class:
            writer.writerow(
                {
                    "class_id": row["class_id"],
                    "class_name": row["class_name"],
                    "precision": f"{row['precision']:.6f}",
                    "recall": f"{row['recall']:.6f}",
                    "f1": f"{row['f1']:.6f}",
                    "support": row["support"],
                }
            )
    return path

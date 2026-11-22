"""Standardized PyTorch training loop, callbacks, checkpointing, and CSV logging."""

from __future__ import annotations

import copy
import csv
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import CLASS_NAMES
from .metrics import compute_classification_metrics
from .utils import count_trainable_params, format_duration, get_device, get_device_description, seed_everything


EXPERIMENT_RUN_COLUMNS = (
    "exp_id",
    "date",
    "owner",
    "group_name",
    "model_name",
    "num_params",
    "activation",
    "use_bn",
    "dropout",
    "optimizer",
    "lr",
    "weight_decay",
    "scheduler",
    "loss_type",
    "aug_mode",
    "batch_size",
    "max_epochs",
    "seed",
    "device",
    "best_epoch",
    "total_epochs",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_macro_f1",
    "train_time_sec",
    "checkpoint_path",
    "history_path",
    "notes",
)

HISTORY_COLUMNS = (
    "epoch",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_macro_f1",
    "lr",
    "epoch_time_sec",
)


class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience: int = 12, min_delta: float = 0.0, mode: str = "min") -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: float | None = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.should_stop: bool = False

    def is_improvement(self, value: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return value < (self.best_score - self.min_delta)
        return value > (self.best_score + self.min_delta)

    def step(self, value: float, epoch: int) -> bool:
        """Update state and return True when the metric reached a new best."""
        if self.is_improvement(value):
            self.best_score = float(value)
            self.best_epoch = int(epoch)
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


class ModelCheckpoint:
    """Persist the best model weights together with restoration metadata."""

    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.best_state_dict: dict[str, torch.Tensor] | None = None

    def save(self, model: nn.Module, metadata: dict[str, Any]) -> Path:
        self.best_state_dict = copy.deepcopy(
            {k: v.detach().cpu() for k, v in model.state_dict().items()}
        )
        payload = {
            "model_state_dict": self.best_state_dict,
            "class_names": list(CLASS_NAMES),
            **metadata,
        }
        torch.save(payload, self.filepath)
        return self.filepath

    def restore_best(self, model: nn.Module, device: torch.device) -> None:
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            model.to(device)


def build_criterion(
    loss_type: str = "ce",
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    device: torch.device | None = None,
) -> nn.Module:
    """Create CrossEntropyLoss supporting standard, class-weighted, and label-smoothed variants."""
    key = loss_type.lower().strip()
    if key in {"ce", "cross_entropy", "standard"}:
        return nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
    if key in {"weighted_ce", "weighted", "class_weighted"}:
        if class_weights is None:
            raise ValueError("class_weights must be provided for weighted_ce loss.")
        weights = class_weights.to(device) if device is not None else class_weights
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=float(label_smoothing))
    if key in {"label_smoothing", "ls"}:
        smoothing = float(label_smoothing) if label_smoothing > 0 else 0.1
        return nn.CrossEntropyLoss(label_smoothing=smoothing)
    raise ValueError(f"Unsupported loss_type: {loss_type!r}")


def build_optimizer(
    model: nn.Module,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    nesterov: bool = True,
) -> torch.optim.Optimizer:
    """Create Adam, AdamW, or SGD optimizer."""
    key = optimizer_name.lower().strip()
    params = [p for p in model.parameters() if p.requires_grad]
    if key == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if key == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if key == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "plateau",
    max_epochs: int = 60,
    factor: float = 0.5,
    patience: int = 4,
    min_lr: float = 1e-6,
):
    """Create ReduceLROnPlateau, CosineAnnealingLR, or None."""
    key = (scheduler_name or "none").lower().strip()
    if key in {"none", "off", ""}:
        return None
    if key in {"plateau", "reduce_on_plateau", "reducelronplateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience, min_lr=min_lr
        )
    if key in {"cosine", "cosine_annealing", "cosineannealinglr"}:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(max_epochs)), eta_min=min_lr
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name!r}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    grad_clip_norm: float | None = None,
) -> tuple[float, float]:
    """Run one training epoch and return (mean_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None and use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += batch_size

    return running_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, Any]:
    """Evaluate a model on a validation or test DataLoader without gradient tracking."""
    model.eval()
    running_loss = 0.0
    total = 0
    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")

        with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        total += batch_size
        all_targets.append(labels.detach().cpu().numpy())
        all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    metrics = compute_classification_metrics(y_true, y_pred, class_names=CLASS_NAMES)
    metrics["loss"] = running_loss / max(1, total)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    return metrics


def _write_history_csv(history_rows: list[dict[str, Any]], history_path: Path) -> Path:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for row in history_rows:
            writer.writerow(row)
    return history_path


def _upsert_experiment_run(run_record: dict[str, Any], runs_csv_path: Path) -> Path:
    """Insert or replace an experiment row keyed by exp_id in logs/experiment_runs.csv."""
    runs_csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, str]] = []
    if runs_csv_path.is_file():
        with runs_csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_rows = [row for row in reader if row.get("exp_id") != run_record["exp_id"]]

    formatted = {col: run_record.get(col, "") for col in EXPERIMENT_RUN_COLUMNS}
    existing_rows.append(formatted)
    existing_rows.sort(key=lambda r: r.get("exp_id", ""))

    with runs_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_RUN_COLUMNS)
        writer.writeheader()
        writer.writerows(existing_rows)
    return runs_csv_path


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Train a PyTorch model using the standardized FER2013 pipeline.

    Required / supported keys in ``config``:
        exp_id, model_name, owner, group_name, activation, use_bn, dropout,
        optimizer, lr, weight_decay, momentum, nesterov, scheduler,
        scheduler_factor, scheduler_patience, min_lr, loss_type, class_weights,
        label_smoothing, aug_mode, batch_size, max_epochs, patience, seed,
        device, use_amp, grad_clip_norm, checkpoint_dir, checkpoint_path,
        logs_dir, runs_csv, notes, verbose.
    """
    exp_id = str(config.get("exp_id", "EXP-00"))
    model_name = str(config.get("model_name", model.__class__.__name__))
    seed = int(config.get("seed", 42))
    max_epochs = int(config.get("max_epochs", 60))
    patience = int(config.get("patience", 12))
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 0.0))
    optimizer_name = str(config.get("optimizer", "adam"))
    scheduler_name = str(config.get("scheduler", "plateau"))
    loss_type = str(config.get("loss_type", "ce"))
    label_smoothing = float(config.get("label_smoothing", 0.0))
    grad_clip_norm = config.get("grad_clip_norm")
    verbose = bool(config.get("verbose", True))

    seed_everything(seed, deterministic=bool(config.get("deterministic", False)))
    device = get_device(config.get("device"))
    device_desc = get_device_description(device)
    use_amp = bool(config.get("use_amp", device.type == "cuda")) and device.type == "cuda"

    model = model.to(device)
    num_params = count_trainable_params(model)

    criterion = build_criterion(
        loss_type=loss_type,
        class_weights=config.get("class_weights"),
        label_smoothing=label_smoothing,
        device=device,
    )
    optimizer = build_optimizer(
        model,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        momentum=float(config.get("momentum", 0.9)),
        nesterov=bool(config.get("nesterov", True)),
    )
    scheduler = build_scheduler(
        optimizer,
        scheduler_name=scheduler_name,
        max_epochs=max_epochs,
        factor=float(config.get("scheduler_factor", 0.5)),
        patience=int(config.get("scheduler_patience", 4)),
        min_lr=float(config.get("min_lr", 1e-6)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    logs_dir = Path(config.get("logs_dir", "logs"))
    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    checkpoint_path = Path(
        config.get("checkpoint_path", checkpoint_dir / f"{exp_id}_{model_name}.pth")
    )
    history_path = Path(config.get("history_path", logs_dir / f"history_{exp_id}.csv"))
    runs_csv_path = Path(config.get("runs_csv", logs_dir / "experiment_runs.csv"))

    early_stopping = EarlyStopping(patience=patience, mode="min")
    checkpoint = ModelCheckpoint(checkpoint_path)

    history_rows: list[dict[str, Any]] = []
    best_snapshot: dict[str, Any] = {}
    start_time = time.perf_counter()

    if verbose:
        print(
            f"\n[{exp_id}] {model_name} | params={num_params:,} | device={device_desc} | "
            f"amp={use_amp} | opt={optimizer_name}(lr={lr}) | max_epochs={max_epochs}"
        )

    for epoch in range(1, max_epochs + 1):
        ep_start = time.perf_counter()
        current_lr = float(optimizer.param_groups[0]["lr"])

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )
        val_loss = float(val_metrics["loss"])
        val_acc = float(val_metrics["accuracy"])
        val_macro_f1 = float(val_metrics["macro_f1"])
        val_weighted_f1 = float(val_metrics["weighted_f1"])

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        ep_time = time.perf_counter() - ep_start
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.6f}",
                "val_macro_f1": f"{val_macro_f1:.6f}",
                "lr": f"{current_lr:.8g}",
                "epoch_time_sec": f"{ep_time:.3f}",
            }
        )

        improved = early_stopping.step(val_loss, epoch)
        if improved:
            best_snapshot = {
                "best_epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
                "val_weighted_f1": val_weighted_f1,
                "per_class": val_metrics["per_class"],
            }
            checkpoint.save(
                model,
                metadata={
                    "exp_id": exp_id,
                    "model_name": model_name,
                    "model_kwargs": {
                        "num_classes": int(config.get("num_classes", 7)),
                        "activation": str(config.get("activation", "relu")),
                        "use_bn": bool(config.get("use_bn", True)),
                        "dropout": float(config.get("dropout", 0.2)),
                    },
                    "epoch": epoch,
                    "best_val_loss": val_loss,
                    "best_val_acc": val_acc,
                    "best_val_macro_f1": val_macro_f1,
                    "seed": seed,
                    "config": {
                        k: v for k, v in config.items() if k != "class_weights" and not callable(v)
                    },
                },
            )

        if verbose:
            marker = "*" if improved else " "
            print(
                f"  Epoch {epoch:02d}/{max_epochs:02d} {marker} | "
                f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_macro_f1:.4f} | "
                f"lr={current_lr:.2e} | {ep_time:.1f}s"
            )

        if early_stopping.should_stop:
            if verbose:
                print(
                    f"  -> EarlyStopping triggered at epoch {epoch} "
                    f"(best_epoch={early_stopping.best_epoch}, "
                    f"best_val_loss={best_snapshot['val_loss']:.4f})"
                )
            break

    total_time_sec = time.perf_counter() - start_time
    total_epochs = len(history_rows)

    checkpoint.restore_best(model, device)
    _write_history_csv(history_rows, history_path)

    record = {
        "exp_id": exp_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "owner": str(config.get("owner", "TV2 - Chau Quoc Bao (3123411024)")),
        "group_name": str(config.get("group_name", "Group1_Architectures")),
        "model_name": model_name,
        "num_params": num_params,
        "activation": str(config.get("activation", "relu")),
        "use_bn": int(bool(config.get("use_bn", True))),
        "dropout": float(config.get("dropout", 0.2)),
        "optimizer": optimizer_name,
        "lr": lr,
        "weight_decay": weight_decay,
        "scheduler": scheduler_name,
        "loss_type": loss_type,
        "aug_mode": str(config.get("aug_mode", "standard")),
        "batch_size": int(config.get("batch_size", train_loader.batch_size or 64)),
        "max_epochs": max_epochs,
        "seed": seed,
        "device": device_desc,
        "best_epoch": int(best_snapshot["best_epoch"]),
        "total_epochs": total_epochs,
        "train_loss": f"{best_snapshot['train_loss']:.6f}",
        "train_acc": f"{best_snapshot['train_acc']:.6f}",
        "val_loss": f"{best_snapshot['val_loss']:.6f}",
        "val_acc": f"{best_snapshot['val_acc']:.6f}",
        "val_macro_f1": f"{best_snapshot['val_macro_f1']:.6f}",
        "train_time_sec": f"{total_time_sec:.2f}",
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "notes": str(config.get("notes", "")),
    }
    if bool(config.get("log_to_runs_csv", True)):
        _upsert_experiment_run(record, runs_csv_path)

    if verbose:
        print(
            f"[{exp_id}] Done in {format_duration(total_time_sec)} | "
            f"best_epoch={best_snapshot['best_epoch']}/{total_epochs} | "
            f"val_loss={best_snapshot['val_loss']:.4f} | "
            f"val_acc={best_snapshot['val_acc'] * 100:.2f}% | "
            f"val_macro_f1={best_snapshot['val_macro_f1']:.4f}"
        )

    return {
        "exp_id": exp_id,
        "best_epoch": int(best_snapshot["best_epoch"]),
        "total_epochs": total_epochs,
        "best_val_loss": float(best_snapshot["val_loss"]),
        "best_val_acc": float(best_snapshot["val_acc"]),
        "best_val_macro_f1": float(best_snapshot["val_macro_f1"]),
        "best_val_weighted_f1": float(best_snapshot["val_weighted_f1"]),
        "best_train_loss": float(best_snapshot["train_loss"]),
        "best_train_acc": float(best_snapshot["train_acc"]),
        "best_per_class": best_snapshot["per_class"],
        "train_time_sec": float(total_time_sec),
        "num_params": num_params,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "device": device_desc,
    }

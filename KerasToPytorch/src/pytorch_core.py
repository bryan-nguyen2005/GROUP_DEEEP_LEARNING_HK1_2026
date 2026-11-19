"""Huấn luyện và đánh giá mô hình PyTorch cho các bài thực hành."""

import csv
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn


def seed_everything(seed=42):
    """Cố định nguồn ngẫu nhiên của Python, NumPy, PyTorch và CUDA."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    """Theo dõi validation loss và giữ bản sao trọng số có loss thấp nhất."""

    def __init__(self, patience=5, min_delta=0.0):
        if patience < 1 or min_delta < 0:
            raise ValueError("patience phải dương và min_delta phải không âm")
        self.patience = patience
        self.min_delta = min_delta
        self.reset()

    def reset(self):
        self.best_loss = float("inf")
        self.best_model_state = None
        self.stale_epochs = 0
        self.reference_loss = float("inf")

    def step(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model_state = deepcopy(model.state_dict())
        if val_loss < self.reference_loss - self.min_delta:
            self.reference_loss = val_loss
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return self.stale_epochs >= self.patience

    def restore(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


_METRICS = {
    "classification": "accuracy",
    "regression": "mae",
    "segmentation": "pixel_accuracy",
}


def _run_epoch(model, loader, criterion, device, task, optimizer=None):
    if task not in _METRICS:
        raise ValueError(f"Task không hợp lệ: {task}")
    reduction = getattr(criterion, "reduction", "mean")
    if reduction not in {"mean", "sum"}:
        raise ValueError("criterion phải dùng reduction='mean' hoặc 'sum'")
    training = optimizer is not None
    model.train(training)
    loss_total = torch.zeros((), device=device, dtype=torch.float64)
    metric_total = torch.zeros((), device=device, dtype=torch.float64)
    observed = torch.zeros((), device=device, dtype=torch.float64)
    loss_count = torch.zeros((), device=device, dtype=torch.float64)
    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            if task == "regression":
                if predictions.shape != targets.shape:
                    raise ValueError("Shape dự đoán và nhãn hồi quy phải trùng nhau")
                values = (predictions.detach() - targets).abs()
                count = targets.numel()
                weight = count
            else:
                values = predictions.detach().argmax(dim=1).eq(targets)
                count = targets.numel()
                weight = count
                if isinstance(criterion, nn.CrossEntropyLoss):
                    valid = targets.ne(criterion.ignore_index)
                    values = values & valid
                    count = valid.sum()
                    if criterion.weight is None:
                        weight = count
                    else:
                        safe_targets = targets.masked_fill(~valid, 0)
                        weight = (criterion.weight[safe_targets] * valid).sum()
                        if weight == 0:
                            # Mean cross-entropy không có mẫu số trong batch này.
                            # Vẫn tính accuracy trên các nhãn hợp lệ.
                            metric_total += values.sum()
                            observed += count
                            continue
            loss = criterion(predictions, targets)
            if loss.ndim != 0:
                raise ValueError("criterion phải trả về loss vô hướng")
            if isinstance(weight, torch.Tensor):
                # Batch toàn ignore có thể khiến CrossEntropyLoss trả NaN.
                loss = torch.where(weight > 0, loss, torch.zeros_like(loss))
            if training:
                loss.backward()
                optimizer.step()

            loss_total += loss.detach() * weight if reduction == "mean" else loss.detach()
            metric_total += values.sum()
            observed += count
            loss_count += weight

    loss_sum, metric_sum, items, denominator = torch.stack(
        (loss_total, metric_total, observed, loss_count)
    ).tolist()
    if not items or not denominator:
        raise ValueError("DataLoader phải chứa dữ liệu hợp lệ để tính loss và metric")
    return {"loss": loss_sum / denominator, _METRICS[task]: metric_sum / items}


def train_one_epoch(model, loader, criterion, optimizer, device, task="classification"):
    """Cập nhật mô hình một epoch và trả về loss cùng metric tương ứng."""
    return _run_epoch(model, loader, criterion, device, task, optimizer)


def evaluate(model, loader, criterion, device, task="classification"):
    """Đánh giá với model.eval() và tắt tính gradient."""
    return _run_epoch(model, loader, criterion, device, task)


def fit(
    model, train_loader, val_loader, criterion, optimizer, device, *,
    epochs, task="classification", early_stopping=None, scheduler=None,
    csv_path=None, verbose=True,
):
    """Ghi train/val loss, metric và learning rate.

    Scheduler kiểu ReduceLROnPlateau nhận val_loss sau mỗi epoch.
    Di chuyển model và loss có trọng số sang device trước khi gọi.
    """
    if epochs < 1:
        raise ValueError("epochs phải lớn hơn 0")
    if task not in _METRICS:
        raise ValueError(f"Task không hợp lệ: {task}")
    if early_stopping is not None:
        early_stopping.reset()

    metric = _METRICS[task]
    history = []
    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train = train_one_epoch(model, train_loader, criterion, optimizer, device, task)
        val = evaluate(model, val_loader, criterion, device, task)
        row = {
            "epoch": epoch,
            "train_loss": train["loss"],
            f"train_{metric}": train[metric],
            "val_loss": val["loss"],
            f"val_{metric}": val[metric],
            "lr": lr,
        }
        history.append(row)
        if verbose:
            print(
                f"Epoch {epoch:02}/{epochs} | train {train['loss']:.4f}, "
                f"{metric} {train[metric]:.4f} | val {val['loss']:.4f}, "
                f"{metric} {val[metric]:.4f} | lr {lr:g}", flush=True,
            )
        if scheduler is not None:
            scheduler.step(val["loss"])
        if early_stopping is not None and early_stopping.step(val["loss"], model):
            break

    if early_stopping is not None:
        early_stopping.restore(model)
    if csv_path is not None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    return history

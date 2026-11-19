"""Tiền xử lý dữ liệu và ghi kết quả cho hai thí nghiệm MNIST."""

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms.functional import convert_image_dtype


KERAS_REFERENCE = {"mlp": (0.11284465673938393, 0.9657), "cnn": (0.03192299669235945, 0.9889)}


def make_loaders(batch_size=128, seed=42, *, standardize=False):
    data_root = Path(os.environ.get("MNIST_DATA_DIR", "data"))
    original_train = MNIST(data_root, train=True, download=True)
    original_test = MNIST(data_root, train=False, download=True)

    def to_tensors(dataset):
        images = convert_image_dtype(dataset.data.unsqueeze(1), torch.float32)
        return TensorDataset(images, dataset.targets.long())

    pool = to_tensors(original_train)
    test = to_tensors(original_test)
    train, val = random_split(
        pool, (54_000, 6_000),
        generator=torch.Generator().manual_seed(seed),
    )
    if standardize:
        train_pixels = pool.tensors[0][train.indices]
        mean, std = train_pixels.mean(), train_pixels.std()
        for images, _ in (pool.tensors, test.tensors):
            images.sub_(mean).div_(std)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   generator=torch.Generator().manual_seed(seed)),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
    )


def use_glorot_initialization(layer):
    """Dùng quy tắc khởi tạo kernel/bias của Dense và Conv2D trong Keras."""
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)


def save_result(model_name, model, test, seconds, epochs, device, *, variant=None):
    keras_loss, keras_accuracy = KERAS_REFERENCE[model_name]
    result = {
        "model": model_name.upper(),
        "epochs": epochs,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "test_loss": test["loss"], "test_accuracy": test["accuracy"],
        "keras_test_loss": keras_loss, "keras_test_accuracy": keras_accuracy,
        "accuracy_delta_pp": (test["accuracy"] - keras_accuracy) * 100,
        "fit_seconds": seconds, "device": str(device),
        "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
    }
    prefix = "01" if model_name == "mlp" else "02"
    suffix = f"_{variant}" if variant else ""
    with Path(f"logs/result_{prefix}_mnist_{model_name}{suffix}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    return result


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def plot_curves(series, output):
    colors = {"MLP": "#245b9e", "CNN": "#ae4c24"}
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), layout="constrained")
    for name, history in series.items():
        epochs = [int(row["epoch"]) for row in history]
        for split, style in (("train", "-"), ("val", "--")):
            label = f"{name} {'train' if split == 'train' else 'validation'}"
            axes[0].plot(epochs, [float(row[f"{split}_loss"]) for row in history],
                         style, label=label, color=colors[name])
            axes[1].plot(epochs, [100 * float(row[f"{split}_accuracy"]) for row in history],
                         style, label=label, color=colors[name])
    for axis, title in zip(axes, ("Cross-entropy", "Độ chính xác (%)")):
        axis.set(xlabel="Epoch", ylabel=title)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, facecolor="white")
    return figure

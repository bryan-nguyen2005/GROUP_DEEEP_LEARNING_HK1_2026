"""Inspect FER2013 splits and export figures and summary statistics."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from .dataset import CLASS_NAMES, compute_train_statistics, load_splits
from .transforms import get_transforms


def _save(figure, path):
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _show_image(axis, image, title):
    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    axis.set_title(title, fontsize=8)
    axis.axis("off")


def run_eda(data_path, output_dir="figures", seed=42, *, stats_csv=None):
    splits = load_splits(data_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    figure, axis = plt.subplots(figsize=(10, 4.5), layout="constrained")
    positions = np.arange(len(CLASS_NAMES))
    split_rows = []
    print(f"{'Split':<13} {'Total':>7} {'All black':>10}", *[f"{x:>9}" for x in CLASS_NAMES])
    for index, (split, (images, labels)) in enumerate(splits.items()):
        counts = np.bincount(labels, minlength=len(CLASS_NAMES))
        black_count = np.count_nonzero(~images.any(axis=(1, 2)))
        split_rows.append((split, len(labels), black_count, *counts))
        print(f"{split:<13} {len(labels):>7} {black_count:>10}", *[f"{x:>9}" for x in counts])
        axis.bar(positions + (index - 1) * 0.25, counts, width=0.24, label=split)
    axis.set(xticks=positions, xticklabels=CLASS_NAMES, ylabel="Image count",
             title="FER2013 class distribution by split")
    axis.legend()
    _save(figure, output / "eda_class_distribution.png")

    figure, axes = plt.subplots(4, 8, figsize=(12, 6), layout="constrained")
    figure.suptitle("FER2013 examples from Training and PublicTest")
    for index, split in enumerate(("Training", "PublicTest")):
        images, labels = splits[split]
        for axis in axes[index * 2:(index + 1) * 2].flat:
            axis.axis("off")
        sample_indices = rng.choice(len(images), size=min(16, len(images)), replace=False)
        for axis, sample in zip(axes[index * 2:(index + 1) * 2].flat, sample_indices):
            _show_image(axis, images[sample], f"{split}\n{CLASS_NAMES[labels[sample]]}")
    _save(figure, output / "eda_sample_images.png")

    train_images, train_labels = splits["Training"]
    mean, std = compute_train_statistics(train_images)
    print(f"Training mean={mean:.8f}, std={std:.8f} (pixel scale [0, 1])")
    standard = get_transforms("standard", mean=mean, std=std)
    strong = get_transforms("strong", mean=mean, std=std)
    sample_indices = rng.choice(len(train_images), size=min(8, len(train_images)), replace=False)
    figure, axes = plt.subplots(len(sample_indices), 3, figsize=(6, 1.35 * len(sample_indices)),
                               squeeze=False, layout="constrained")
    figure.suptitle("Training augmentation (original, standard, strong)")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for row, sample in enumerate(sample_indices):
            raw = train_images[sample]
            image = Image.fromarray(raw)
            _show_image(axes[row, 0], raw, CLASS_NAMES[train_labels[sample]])
            for col, (name, transform) in enumerate((("Standard", standard), ("Strong", strong)), 1):
                augmented = (transform(image)[0] * std + mean).clamp(0, 1) * 255
                _show_image(axes[row, col], augmented.numpy(), name)
    _save(figure, output / "eda_augmented_samples.png")

    if stats_csv is not None:
        path = Path(stats_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("split", "total_images", "all_black_images", *CLASS_NAMES,
                             "pixel_mean_0_to_1", "pixel_std_0_to_1"))
            for row in split_rows:
                statistics = (f"{mean:.8f}", f"{std:.8f}") if row[0] == "Training" else ("", "")
                writer.writerow((*row, *statistics))
        print(f"EDA summary: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", help="FER2013 CSV or split directory")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats-csv", default="logs/tv1_eda_summary.csv",
                        help="CSV path for class counts and Training normalization")
    arguments = parser.parse_args()
    run_eda(arguments.data_path, arguments.output_dir, arguments.seed,
            stats_csv=arguments.stats_csv)


if __name__ == "__main__":
    main()

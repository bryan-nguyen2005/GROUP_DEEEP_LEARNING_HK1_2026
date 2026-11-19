"""Load the original FER2013 splits from CSV or class-labeled images."""

import csv
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import eval_transform, get_transforms


CLASS_NAMES = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")
SPLITS = {"train": "Training", "val": "PublicTest", "test": "PrivateTest"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
IMAGE_SIZE = 48


def _pack(rows_by_split):
    result = {}
    for split, rows in rows_by_split.items():
        if not rows:
            raise ValueError(f"No images found in {split}")
        images, labels = zip(*rows)
        result[split] = (np.stack(images), np.asarray(labels, dtype=np.int64))
    return result


def _read_csv(path):
    rows_by_split = {split: [] for split in SPLITS.values()}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = {name.strip().lower(): name for name in reader.fieldnames or []}
        if not {"emotion", "pixels", "usage"} <= headers.keys():
            raise ValueError(f"{path} must contain emotion, pixels and Usage columns")
        for row in reader:
            try:
                label = int(row[headers["emotion"]])
                split = row[headers["usage"]].strip()
                tokens = row[headers["pixels"]].split()
                pixels = np.asarray(tokens, dtype=np.int32)
                if (
                    split not in rows_by_split
                    or not 0 <= label < len(CLASS_NAMES)
                    or pixels.size != IMAGE_SIZE ** 2
                    or np.any((pixels < 0) | (pixels > 255))
                ):
                    raise ValueError
            except (AttributeError, TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"Invalid FER2013 record in {path}:{reader.line_num}") from error
            image = pixels.astype(np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE)
            rows_by_split[split].append((image, label))
    return _pack(rows_by_split)


def _load_cached_csv(path):
    cache = path.with_suffix(".npz")
    source = path.stat()
    version = (source.st_mtime_ns, source.st_size)
    try:
        with np.load(cache, allow_pickle=False) as saved:
            if tuple(saved["source_version"]) == version:
                return {
                    split: (saved[f"{split}_images"], saved[f"{split}_labels"])
                    for split in SPLITS.values()
                }
    except (OSError, ValueError, KeyError, BadZipFile, EOFError):
        pass

    splits = _read_csv(path)
    arrays = {}
    for split, (images, labels) in splits.items():
        arrays[f"{split}_images"] = images
        arrays[f"{split}_labels"] = labels

    temporary_path = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            np.savez(temporary, source_version=version, **arrays)
        temporary_path.replace(cache)
    except OSError:
        # Loading still works when the input directory is read-only.
        pass
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return splits


def _read_image_folders(root):
    class_ids = {name.lower(): i for i, name in enumerate(CLASS_NAMES)}
    class_ids.update({str(i): i for i in range(len(CLASS_NAMES))})
    rows_by_split = {split: [] for split in SPLITS.values()}

    for split, rows in rows_by_split.items():
        for folder in sorted((root / split).iterdir()):
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            try:
                label = class_ids[folder.name.lower()]
            except KeyError as error:
                raise ValueError(f"Unknown emotion folder: {folder}") from error
            for path in sorted(folder.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                try:
                    with Image.open(path) as image:
                        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                            raise ValueError(f"expected {IMAGE_SIZE}×{IMAGE_SIZE}, got {image.size}")
                        rows.append((np.array(image.convert("L")), label))
                except (OSError, ValueError) as error:
                    raise ValueError(f"Invalid FER2013 image {path}: {error}") from error
    return _pack(rows_by_split)


def load_splits(data_path):
    """Return raw uint8 images and integer labels keyed by original split."""
    path = Path(data_path)
    if path.is_dir():
        csv_path = path / "fer2013.csv"
        if csv_path.is_file():
            return _load_cached_csv(csv_path)
        return _read_image_folders(path)
    if path.is_file() and path.suffix.lower() == ".csv":
        return _load_cached_csv(path)
    raise FileNotFoundError(f"Provide a FER2013 CSV or split directory: {path}")


def compute_train_statistics(train_images):
    """Compute population mean/std in [0, 1] using only Training pixels."""
    histogram = np.bincount(train_images.reshape(-1), minlength=256)
    intensities = np.arange(256) / 255.0
    mean = float(histogram @ intensities / histogram.sum())
    if np.count_nonzero(histogram) == 1:
        return mean, 1.0
    variance = histogram @ ((intensities - mean) ** 2) / histogram.sum()
    return mean, float(np.sqrt(variance))


def compute_class_weights(train_labels):
    """Return N / (7 * class_count) for present classes and zero for absent ones."""
    labels = torch.as_tensor(train_labels, dtype=torch.long)
    if labels.ndim != 1 or labels.numel() == 0 or ((labels < 0) | (labels >= len(CLASS_NAMES))).any():
        raise ValueError("train_labels must be a non-empty 1D sequence of labels 0–6")
    counts = torch.bincount(labels, minlength=len(CLASS_NAMES))
    weights = torch.zeros(len(CLASS_NAMES), dtype=torch.float32)
    present = counts > 0
    weights[present] = labels.numel() / (len(CLASS_NAMES) * counts[present])
    return weights


class FER2013Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform if transform is not None else eval_transform()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index])
        return self.transform(image), int(self.labels[index])


def get_dataloaders(
    data_path: str, aug_mode: str = "standard", batch_size: int = 64,
    num_workers: int = 2, seed: int = 42,
) -> dict[str, DataLoader]:
    splits = load_splits(data_path)
    mean, std = compute_train_statistics(splits["Training"][0])
    train_transform = get_transforms(aug_mode, mean=mean, std=std)
    evaluation = eval_transform(mean=mean, std=std)

    return {
        role: DataLoader(
            FER2013Dataset(*splits[split], train_transform if role == "train" else evaluation),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=role == "train",
            generator=torch.Generator().manual_seed(seed + index),
        )
        for index, (role, split) in enumerate(SPLITS.items())
    }

"""Check both input formats, cache recovery and Train-only normalization."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
import torch

from src.dataset import SPLITS, compute_class_weights, get_dataloaders, load_splits
from src.transforms import get_tta_views


class DatasetTests(unittest.TestCase):
    def test_split_identity_and_train_only_normalization(self):
        rows = {"Training": [(0, 0), (1, 255)], "PublicTest": [(5, 64)],
                "PrivateTest": [(6, 128)]}
        with TemporaryDirectory() as temp:
            root = Path(temp)
            csv_path = root / "fer2013.csv"
            with csv_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["emotion", "pixels", "Usage"])
                for split, samples in rows.items():
                    for label, raw in samples:
                        writer.writerow([label, " ".join([str(raw)] * (48 * 48)), split])
                        folder = root / "images" / split / str(label)
                        folder.mkdir(parents=True)
                        Image.new("L", (48, 48), color=raw).save(folder / "sample.png")

            cache = csv_path.with_suffix(".npz")
            cache.write_bytes(b"incomplete cache")
            from_csv = load_splits(csv_path)
            self.assertEqual(cache.read_bytes()[:2], b"PK")
            from_folders = load_splits(root / "images")
            for split in SPLITS.values():
                for from_csv_array, from_folders_array in zip(from_csv[split], from_folders[split]):
                    np.testing.assert_array_equal(from_csv_array, from_folders_array)

            for mode in ("none", "standard", "strong"):
                loaders = get_dataloaders(csv_path, aug_mode=mode, num_workers=0)
                self.assertEqual(tuple(loaders), ("train", "val", "test"))
                batch, _ = next(iter(loaders["train"]))
                self.assertEqual(batch.shape, (2, 1, 48, 48))
                for role, label, pixel in (("val", 5, 64), ("test", 6, 128)):
                    dataset = loaders[role].dataset
                    image, actual = dataset[0]
                    self.assertEqual(actual, label)
                    torch.testing.assert_close(image, torch.full_like(image, 2 * pixel / 255 - 1))
                    self.assertTrue(torch.equal(dataset[0][0], image))

            torch.testing.assert_close(
                compute_class_weights([0, 0, 1]),
                torch.tensor([3 / 14, 3 / 7, 0, 0, 0, 0, 0]),
            )

    def test_tta_views_preserve_batch_shape(self):
        images = torch.arange(2 * 48 * 48, dtype=torch.float32).reshape(2, 1, 48, 48)
        original, mirrored, cropped = get_tta_views(images)
        self.assertIs(original, images)
        self.assertTrue(torch.equal(mirrored, images.flip(-1)))
        self.assertEqual(cropped.shape, images.shape)
        self.assertFalse(torch.equal(cropped, images))

"""Bài 6: Hand Depth Image Segmentation (SegNet 96x96) training & evaluation script.

Supports loading an external directory (images/ + masks/) or generating a deterministic
1,050-sample 96x96 Kinect-style Hand Depth Segmentation benchmark (7 semantic classes:
Background, Palm, Thumb, Index, Middle, Ring, Pinky) when the original 2018 Kinect
dataset is not publicly hosted.

Author: TV4 - Vo Huynh Minh Sang (MSSV: 3123411256)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .models.segnet import SegNet
from .pytorch_core import EarlyStopping, evaluate, fit, get_device, seed_everything
from .visualize import plot_learning_curves, plot_segmentation_results


HAND_CLASS_NAMES = ("Background", "Palm", "Thumb", "Index", "Middle", "Ring", "Pinky")
IMAGE_SIZE = 96


def _draw_capsule(
    yy: np.ndarray,
    xx: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean mask and normalized radial distance inside a 2D capsule."""
    vx, vy = p1[0] - p0[0], p1[1] - p0[1]
    seg_len2 = max(vx * vx + vy * vy, 1e-6)
    t = np.clip(((xx - p0[0]) * vx + (yy - p0[1]) * vy) / seg_len2, 0.0, 1.0)
    px = p0[0] + t * vx
    py = p0[1] + t * vy
    dist = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
    mask = dist <= radius
    radial = np.clip(1.0 - dist / max(radius, 1e-6), 0.0, 1.0)
    return mask, radial


def generate_synthetic_hand_depth_dataset(
    num_samples: int = 1050,
    size: int = IMAGE_SIZE,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate deterministic 96x96 Kinect-like hand depth images and 7-class masks."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    images = np.zeros((num_samples, size, size), dtype=np.float32)
    masks = np.zeros((num_samples, size, size), dtype=np.int64)

    # Finger specs: (class_id, base_x_rel, base_y_rel, base_angle_deg, length_rel, radius_rel)
    finger_specs = (
        (2, -0.82, 0.12, -142.0, 1.15, 0.28),  # Thumb
        (3, -0.52, -0.85, -104.0, 1.55, 0.23), # Index
        (4, -0.10, -0.96, -90.0, 1.72, 0.24),  # Middle
        (5, 0.34, -0.88, -77.0, 1.56, 0.23),   # Ring
        (6, 0.72, -0.68, -62.0, 1.25, 0.20),   # Pinky
    )

    for idx in range(num_samples):
        # Randomize hand center, scale, global rotation, and flip (left/right hand)
        cx = float(rng.uniform(36.0, 60.0))
        cy = float(rng.uniform(50.0, 66.0))
        scale = float(rng.uniform(13.0, 18.5))
        global_rot = float(np.deg2rad(rng.uniform(-24.0, 24.0)))
        mirror = -1.0 if rng.random() < 0.5 else 1.0

        cos_r, sin_r = float(np.cos(global_rot)), float(np.sin(global_rot))

        def to_world(rx: float, ry: float) -> tuple[float, float]:
            mx = rx * mirror * scale
            my = ry * scale
            wx = cx + mx * cos_r - my * sin_r
            wy = cy + mx * sin_r + my * cos_r
            return wx, wy

        # Background depth plane (farther = lower intensity in normalized depth map)
        bg_tilt_x = float(rng.uniform(-0.04, 0.04))
        bg_tilt_y = float(rng.uniform(-0.04, 0.04))
        depth = 0.12 + bg_tilt_x * (xx / size - 0.5) + bg_tilt_y * (yy / size - 0.5)
        mask = np.zeros((size, size), dtype=np.int64)

        # Draw subtle wrist/forearm silhouette (background class 0, medium depth)
        w0 = to_world(0.0, 0.95)
        w1 = to_world(0.0, 2.30)
        wrist_m, wrist_rad = _draw_capsule(yy, xx, w0, w1, radius=scale * 0.62)
        depth[wrist_m] = np.maximum(depth[wrist_m], 0.42 + 0.16 * wrist_rad[wrist_m])

        # Palm (class 1): slightly elliptical center region
        dx = xx - cx
        dy = yy - cy
        lx = (dx * cos_r + dy * sin_r) / (scale * 1.02)
        ly = (-dx * sin_r + dy * cos_r) / (scale * 1.08)
        palm_dist = np.sqrt(lx * lx + ly * ly)
        palm_m = palm_dist <= 1.0
        palm_depth = 0.62 + 0.24 * np.clip(1.0 - palm_dist, 0.0, 1.0)
        depth[palm_m] = np.maximum(depth[palm_m], palm_depth[palm_m])
        mask[palm_m] = 1

        # Articulated 5 fingers (classes 2..6) with random extension & splay
        for cls_id, bx, by, base_deg, base_len, base_rad in finger_specs:
            # Random finger curl (shorter projected length) or splay
            ext = float(rng.uniform(0.55, 1.12))
            angle_jitter = float(rng.uniform(-14.0, 14.0))
            ang = np.deg2rad(base_deg + angle_jitter)
            dx_rel = float(np.cos(ang)) * base_len * ext
            dy_rel = float(np.sin(ang)) * base_len * ext

            p0 = to_world(bx, by)
            p1 = to_world(bx + dx_rel, by + dy_rel)
            f_mask, f_rad = _draw_capsule(yy, xx, p0, p1, radius=scale * base_rad)
            # Curled fingers are slightly closer to camera (brighter depth)
            curl_boost = (1.12 - ext) * 0.16
            f_depth = 0.60 + curl_boost + 0.25 * f_rad
            closer = f_mask & (f_depth >= depth - 0.06)
            depth[closer] = np.maximum(depth[closer], f_depth[closer])
            mask[closer] = cls_id

        # Add Kinect sensor noise + occasional small depth dropout specks
        noise = rng.normal(0.0, 0.045, size=(size, size)).astype(np.float32)
        depth = np.clip(depth + noise, 0.0, 1.0)
        dropout_mask = rng.random((size, size)) < 0.015
        depth[dropout_mask] = 0.0

        images[idx] = depth
        masks[idx] = mask

    return images, masks


def load_external_segmentation_dir(data_dir: Path, size: int = IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Load paired grayscale depth images (images/) and class-index masks (masks/)."""
    img_dir = data_dir / "images"
    mask_dir = data_dir / "masks"
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected images/ and masks/ inside {data_dir}")

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        mask_path = mask_dir / (img_path.stem + ".png")
        if not mask_path.is_file():
            mask_path = mask_dir / img_path.name
        if not mask_path.is_file():
            continue
        with Image.open(img_path) as im:
            im_arr = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
        with Image.open(mask_path) as mk:
            mk_arr = np.asarray(mk.convert("L").resize((size, size), Image.NEAREST), dtype=np.int64)
        images.append(im_arr)
        masks.append(mk_arr)
    if not images:
        raise ValueError(f"No valid image/mask pairs found in {data_dir}")
    return np.stack(images), np.stack(masks)


class HandDepthSegmentationDataset(Dataset):
    """PyTorch Dataset returning (1, 96, 96) normalized depth tensor and (96, 96) Long mask."""

    def __init__(self, images: np.ndarray, masks: np.ndarray, augment: bool = False) -> None:
        self.images = images
        self.masks = masks
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.images[index].copy()
        mask = self.masks[index].copy()
        if self.augment:
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            noise = np.random.normal(0.0, 0.015, size=img.shape).astype(np.float32)
            img = np.clip(img + noise, 0.0, 1.0)
        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(mask).long()


@torch.no_grad()
def compute_segmentation_metrics(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> dict[str, Any]:
    """Compute Loss, Pixel Accuracy, Mean IoU (mIoU), and per-class IoU."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0
    total_pixels = 0
    sample_depths: list[np.ndarray] = []
    sample_gts: list[np.ndarray] = []
    sample_preds: list[np.ndarray] = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)

        n_pix = targets.numel()
        total_loss += float(loss.item()) * n_pix
        total_pixels += n_pix

        t_np = targets.cpu().numpy().reshape(-1)
        p_np = preds.cpu().numpy().reshape(-1)
        valid = (t_np >= 0) & (t_np < num_classes)
        bins = np.bincount(t_np[valid] * num_classes + p_np[valid], minlength=num_classes * num_classes)
        conf += bins.reshape(num_classes, num_classes)

        if len(sample_depths) < 6:
            need = 6 - len(sample_depths)
            sample_depths.extend(inputs[:need, 0].cpu().numpy())
            sample_gts.extend(targets[:need].cpu().numpy())
            sample_preds.extend(preds[:need].cpu().numpy())

    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    denom = tp + fp + fn
    per_class_iou = np.where(denom > 0, tp / np.maximum(denom, 1.0), 0.0)
    present = denom > 0
    miou = float(per_class_iou[present].mean()) if np.any(present) else 0.0
    pixel_acc = float(tp.sum() / max(1, conf.sum()))

    return {
        "loss": total_loss / max(1, total_pixels),
        "pixel_accuracy": pixel_acc,
        "miou": miou,
        "per_class_iou": per_class_iou.tolist(),
        "samples": (sample_depths, sample_gts, sample_preds),
    }


def run_segnet_experiment(
    data_dir: str | Path | None = None,
    num_samples: int = 1050,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    base_channels: int = 32,
    seed: int = 42,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Train and evaluate SegNet on Hand Depth Segmentation, exporting logs, checkpoints, and figures."""
    seed_everything(seed)
    device = get_device()
    root = Path(project_root)

    if data_dir is not None and Path(data_dir).is_dir():
        images, masks = load_external_segmentation_dir(Path(data_dir))
        source_desc = f"External directory ({data_dir})"
    else:
        images, masks = generate_synthetic_hand_depth_dataset(num_samples=num_samples, seed=seed)
        source_desc = f"Synthetic Kinect-style Hand Depth Benchmark ({num_samples} samples, 96x96, 7 classes)"

    num_classes = int(masks.max()) + 1
    class_names = HAND_CLASS_NAMES[:num_classes]

    # Deterministic 70 / 15 / 15 split
    n = len(images)
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train_ids = perm[:n_train]
    val_ids = perm[n_train : n_train + n_val]
    test_ids = perm[n_train + n_val :]

    train_loader = DataLoader(HandDepthSegmentationDataset(images[train_ids], masks[train_ids], augment=True), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(HandDepthSegmentationDataset(images[val_ids], masks[val_ids], augment=False), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(HandDepthSegmentationDataset(images[test_ids], masks[test_ids], augment=False), batch_size=batch_size, shuffle=False)

    model = SegNet(in_channels=1, num_classes=num_classes, base_channels=base_channels, use_bn=True).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)
    early_stopping = EarlyStopping(patience=6)

    print(f"[SegNet] {source_desc}")
    print(f"[SegNet] Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)} | params={num_params:,} | device={device}")

    t0 = time.perf_counter()
    history = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs=epochs,
        task="segmentation",
        early_stopping=early_stopping,
        scheduler=scheduler,
        csv_path=root / "logs/history_06_segnet_hand.csv",
        verbose=True,
    )
    train_time_sec = time.perf_counter() - t0

    # Save checkpoint
    ckpt_path = root / "checkpoints/06_segnet_hand.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "base_channels": base_channels,
            "class_names": list(class_names),
            "source_desc": source_desc,
        },
        ckpt_path,
    )

    # Evaluate on held-out Test split
    test_metrics = compute_segmentation_metrics(model, test_loader, num_classes=num_classes, device=device)
    print(
        f"[SegNet TEST] loss={test_metrics['loss']:.4f} | "
        f"pixel_acc={test_metrics['pixel_accuracy']*100:.2f}% | "
        f"mIoU={test_metrics['miou']*100:.2f}% | time={train_time_sec:.1f}s"
    )
    for cname, ciou in zip(class_names, test_metrics["per_class_iou"]):
        print(f"  Class {cname:<12}: IoU = {ciou*100:.2f}%")

    # Save result CSV
    result_csv = root / "logs/result_06_segnet_hand.csv"
    with result_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "model_name", "parameters", "train_samples", "val_samples", "test_samples",
            "epochs_run", "train_time_sec", "test_loss", "test_pixel_accuracy", "test_miou",
            "device", "dataset_source",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "model_name": "SegNet",
            "parameters": num_params,
            "train_samples": len(train_ids),
            "val_samples": len(val_ids),
            "test_samples": len(test_ids),
            "epochs_run": len(history),
            "train_time_sec": f"{train_time_sec:.2f}",
            "test_loss": f"{test_metrics['loss']:.6f}",
            "test_pixel_accuracy": f"{test_metrics['pixel_accuracy']:.6f}",
            "test_miou": f"{test_metrics['miou']:.6f}",
            "device": str(device),
            "dataset_source": source_desc,
        })

    # Generate 300-DPI figures
    plot_learning_curves(
        history,
        root / "figures/06_segnet_loss_acc.png",
        metric_key="pixel_accuracy",
        metric_label="Pixel Accuracy (%)",
        title="Bài 6: SegNet Hand Depth Segmentation — Loss & Pixel Accuracy",
    )
    depths, gts, preds = test_metrics["samples"]
    plot_segmentation_results(
        depths,
        gts,
        preds,
        class_names,
        root / "figures/06_segnet_mask_predictions.png",
        title=(
            f"Bài 6: SegNet Hand Depth Predictions on Test Set — "
            f"Pixel Acc: {test_metrics['pixel_accuracy']*100:.2f}% | mIoU: {test_metrics['miou']*100:.2f}%"
        ),
    )

    return {
        "history": history,
        "test_metrics": test_metrics,
        "num_params": num_params,
        "train_time_sec": train_time_sec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None, help="Optional folder containing images/ and masks/")
    parser.add_argument("--num-samples", type=int, default=1050)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_segnet_experiment(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base_channels=args.base_channels,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

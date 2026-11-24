"""Visualize first/final convolution filters, feature maps, and Grad-CAM for FER2013.

Fulfills the mandatory Slide 5 requirement: "Visualize the final convolution filters".
Author: TV4 - Vo Huynh Minh Sang (MSSV: 3123411256)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .dataset import CLASS_NAMES, compute_train_statistics, load_splits
from .models import build_model
from .transforms import eval_transform
from .utils import get_device, seed_everything
from .visualize import _save_figure


def load_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild and restore a PyTorch model from a saved checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = str(ckpt.get("model_name", "ImprovedFERNet"))
    model_kwargs = dict(ckpt.get("model_kwargs") or {})
    if not model_kwargs:
        model_kwargs = {
            "num_classes": len(ckpt.get("class_names", CLASS_NAMES)),
            "activation": str(ckpt.get("activation", "relu")),
            "use_bn": bool(ckpt.get("use_bn", True)),
            "dropout": float(ckpt.get("dropout", 0.2)),
        }
    model = build_model(model_name, pretrained=False, **model_kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def visualize_first_conv_filters(model: nn.Module, output_path: str | Path, dpi: int = 300) -> Path:
    """Visualize all 3x3 kernels in the first convolution layer (edge/contrast detectors)."""
    first_conv = model.get_first_conv_layer()
    weights = first_conv.weight.detach().cpu().numpy()  # (out_c, in_c, kH, kW)
    num_filters = min(64, weights.shape[0])
    ncols = 8
    nrows = int(np.ceil(num_filters / ncols))

    # Collapse input channels if > 1 (e.g., VGG16Transfer)
    kernels = weights[:num_filters].mean(axis=1)
    vmax = float(np.max(np.abs(kernels))) + 1e-8

    fig, axes = plt.subplots(nrows, ncols, figsize=(10.0, nrows * 1.28), layout="constrained")
    fig.suptitle(
        f"First Convolution Layer Filters ({first_conv.out_channels} × {first_conv.kernel_size[0]}×{first_conv.kernel_size[1]})",
        fontsize=13,
        fontweight="bold",
        color="#1F2328",
    )
    flat_axes = np.asarray(axes).reshape(-1)
    for idx, ax in enumerate(flat_axes):
        ax.axis("off")
        if idx >= num_filters:
            continue
        im = ax.imshow(kernels[idx], cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(f"F{idx:02d}", fontsize=7.5, pad=2, color="#24292F")

    cbar = fig.colorbar(im, ax=flat_axes.tolist(), fraction=0.022, pad=0.02)
    cbar.set_label("Filter Weight", fontsize=9.5)
    return _save_figure(fig, output_path, dpi=dpi)


def visualize_final_conv_filters(model: nn.Module, output_path: str | Path, top_k: int = 36, dpi: int = 300) -> Path:
    """Visualize top-K strongest 3x3 filter kernels and L2-norm energy distribution of final_conv."""
    final_conv = model.get_final_conv_layer()
    weights = final_conv.weight.detach().cpu().numpy()  # (out_c, in_c, kH, kW)
    out_c, in_c, kh, kw = weights.shape

    # L2 norm per output filter across all input channels and spatial dims
    filter_norms = np.linalg.norm(weights.reshape(out_c, -1), axis=1)
    top_indices = np.argsort(filter_norms)[::-1][:top_k]

    fig = plt.figure(figsize=(13.6, 6.8), layout="constrained")
    fig.suptitle(
        f"Final Convolution Layer Filters ({out_c} filters × {in_c} in-channels × {kh}×{kw})",
        fontsize=13.5,
        fontweight="bold",
        color="#1F2328",
    )
    subfigs = fig.subfigures(1, 2, width_ratios=[1.25, 1.0])

    # Left subfigure: 6x6 grid of top-36 filters (strongest input-channel slice per filter)
    grid_side = int(np.ceil(np.sqrt(top_k)))
    axes_left = subfigs[0].subplots(grid_side, grid_side)
    subfigs[0].suptitle(f"Top-{top_k} Strongest Final Conv Filters (Peak Input Channel 3×3 Kernel)", fontsize=10.5, fontweight="bold")
    flat_left = np.asarray(axes_left).reshape(-1)

    selected_slices = []
    for f_idx in top_indices:
        ch_norms = np.linalg.norm(weights[f_idx].reshape(in_c, -1), axis=1)
        best_in_ch = int(np.argmax(ch_norms))
        selected_slices.append(weights[f_idx, best_in_ch])
    vmax = max(float(np.max(np.abs(selected_slices))), 1e-8)

    for slot, ax in enumerate(flat_left):
        ax.axis("off")
        if slot >= len(top_indices):
            continue
        f_idx = int(top_indices[slot])
        im = ax.imshow(selected_slices[slot], cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(f"#{f_idx} ({filter_norms[f_idx]:.2f})", fontsize=7.2, pad=2, color="#24292F")

    cbar = subfigs[0].colorbar(im, ax=flat_left.tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("Weight", fontsize=8.5)

    # Right subfigure: 2 stacked plots (histogram + ranked energy curve)
    axes_right = subfigs[1].subplots(2, 1)
    ax_hist, ax_rank = axes_right

    ax_hist.hist(filter_norms, bins=28, color="#0969DA", alpha=0.82, edgecolor="white")
    mean_norm = float(np.mean(filter_norms))
    std_norm = float(np.std(filter_norms))
    ax_hist.axvline(mean_norm, color="#CF222E", linestyle="--", linewidth=1.6, label=f"Mean = {mean_norm:.3f} ± {std_norm:.3f}")
    ax_hist.set_title("Filter L2-Norm Distribution (All 512 Filters)", fontsize=10.5, fontweight="bold")
    ax_hist.set_xlabel("Filter L2 Norm")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(True, linestyle=":", alpha=0.5)
    ax_hist.legend(fontsize=8.5)

    sorted_norms = np.sort(filter_norms)[::-1]
    ranks = np.arange(1, out_c + 1)
    ax_rank.plot(ranks, sorted_norms, color="#1A7F37", linewidth=2.0)
    ax_rank.fill_between(ranks[:top_k], sorted_norms[:top_k], alpha=0.2, color="#0969DA", label=f"Top-{top_k} Highlighted")
    ax_rank.set_title("Ranked Filter Energy Profile", fontsize=10.5, fontweight="bold")
    ax_rank.set_xlabel("Filter Rank (Sorted by L2 Norm)")
    ax_rank.set_ylabel("L2 Norm")
    ax_rank.grid(True, linestyle=":", alpha=0.5)
    ax_rank.legend(fontsize=8.5)

    return _save_figure(fig, output_path, dpi=dpi)


class GradCAM:
    """Lightweight pure-PyTorch Grad-CAM extractor on a target Conv2d layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def compute(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int, float, np.ndarray]:
        """Compute Grad-CAM heatmap (H, W) in [0, 1], predicted class, confidence, and raw activation maps."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        conf = float(probs[0, pred_idx].item())
        target = pred_idx if class_idx is None else int(class_idx)

        score = logits[0, target]
        score.backward()

        assert self.activations is not None and self.gradients is not None
        # Global average pooling of gradients over spatial dimensions
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)
        return cam_np, pred_idx, conf, self.activations[0].cpu().numpy()


def _select_representative_samples(
    model: nn.Module,
    raw_images: np.ndarray,
    labels: np.ndarray,
    transform,
    device: torch.device,
) -> tuple[dict[int, int], list[tuple[int, int, int, float]]]:
    """Find the highest-confidence correct sample per class and top wrong cases."""
    from PIL import Image

    best_correct: dict[int, tuple[float, int]] = {}
    wrong_candidates: list[tuple[float, int, int, int]] = []

    batch_size = 128
    model.eval()
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            end = min(len(labels), start + batch_size)
            tensors = torch.stack([transform(Image.fromarray(raw_images[i])) for i in range(start, end)]).to(device)
            probs = torch.softmax(model(tensors), dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            for offset in range(end - start):
                idx = start + offset
                true_y = int(labels[idx])
                pred_y = int(preds[offset])
                conf = float(probs[offset, pred_y])
                if pred_y == true_y:
                    prev = best_correct.get(true_y)
                    if prev is None or conf > prev[0]:
                        best_correct[true_y] = (conf, idx)
                else:
                    wrong_candidates.append((conf, idx, true_y, pred_y))

    correct_by_class = {cls: item[1] for cls, item in best_correct.items()}
    # Pick diverse wrong cases (different true classes) with confident mistakes
    wrong_candidates.sort(key=lambda item: item[0], reverse=True)
    selected_wrong: list[tuple[int, int, int, float]] = []
    seen_true: set[int] = set()
    for conf, idx, true_y, pred_y in wrong_candidates:
        if true_y not in seen_true:
            selected_wrong.append((idx, true_y, pred_y, conf))
            seen_true.add(true_y)
        if len(selected_wrong) == len(CLASS_NAMES):
            break
    return correct_by_class, selected_wrong


def visualize_final_conv_feature_maps(
    model: nn.Module,
    raw_images: np.ndarray,
    labels: np.ndarray,
    correct_by_class: dict[int, int],
    transform,
    device: torch.device,
    output_path: str | Path,
    top_maps: int = 6,
    dpi: int = 300,
) -> Path:
    """Plot original face + top-6 highest-activation final_conv feature maps for all 7 emotions."""
    from PIL import Image

    final_conv = model.get_final_conv_layer()
    captured: list[torch.Tensor] = []
    handle = final_conv.register_forward_hook(lambda m, inp, out: captured.append(out.detach().cpu()))

    nrows = len(CLASS_NAMES)
    ncols = 1 + top_maps
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.65, nrows * 1.55), layout="constrained")
    fig.suptitle(
        f"Final Convolution Layer Feature Maps Across 7 Emotions (Top-{top_maps} Activated Channels)",
        fontsize=13.5,
        fontweight="bold",
        color="#1F2328",
    )

    try:
        model.eval()
        with torch.no_grad():
            for row, cls_idx in enumerate(range(len(CLASS_NAMES))):
                sample_idx = correct_by_class.get(cls_idx, int(np.where(labels == cls_idx)[0][0]))
                raw_img = raw_images[sample_idx]
                tensor = transform(Image.fromarray(raw_img)).unsqueeze(0).to(device)
                captured.clear()
                probs = torch.softmax(model(tensor), dim=1)[0]
                conf = float(probs[cls_idx].item())
                fmap = F.relu(captured[0][0]).numpy()  # (C, H, W)
                energies = fmap.mean(axis=(1, 2))
                top_channels = np.argsort(energies)[::-1][:top_maps]

                ax0 = axes[row, 0]
                ax0.imshow(raw_img, cmap="gray", vmin=0, vmax=255)
                ax0.set_title(f"{CLASS_NAMES[cls_idx]}\n({conf * 100:.1f}%)", fontsize=8.5, fontweight="bold", color="#0969DA")
                ax0.axis("off")

                for col_offset, ch_idx in enumerate(top_channels, start=1):
                    ax = axes[row, col_offset]
                    ax.imshow(fmap[ch_idx], cmap="inferno", interpolation="bilinear")
                    ax.set_title(f"Ch #{ch_idx}", fontsize=8.0, color="#24292F")
                    ax.axis("off")
    finally:
        handle.remove()

    return _save_figure(fig, output_path, dpi=dpi)


def visualize_gradcam_7emotions(
    model: nn.Module,
    raw_images: np.ndarray,
    labels: np.ndarray,
    correct_by_class: dict[int, int],
    wrong_samples: list[tuple[int, int, int, float]],
    transform,
    device: torch.device,
    output_path: str | Path,
    dpi: int = 300,
) -> Path:
    """Plot Grad-CAM overlays on final_conv for 7 correct emotions (top 2 rows) + 7 wrong cases (bottom row)."""
    from PIL import Image

    gradcam = GradCAM(model, model.get_final_conv_layer())
    fig, axes = plt.subplots(3, 7, figsize=(13.6, 6.2), layout="constrained")
    fig.suptitle(
        "Grad-CAM on Final Convolution Layer: 7 Emotions (Correct) & Misclassified Cases",
        fontsize=13.5,
        fontweight="bold",
        color="#1F2328",
    )

    try:
        # Row 0: Original faces (correct); Row 1: Grad-CAM overlay (correct)
        for col, cls_idx in enumerate(range(len(CLASS_NAMES))):
            sample_idx = correct_by_class.get(cls_idx, int(np.where(labels == cls_idx)[0][0]))
            raw_img = raw_images[sample_idx]
            tensor = transform(Image.fromarray(raw_img)).unsqueeze(0).to(device)
            cam, pred_idx, conf, _ = gradcam.compute(tensor, class_idx=cls_idx)

            ax_orig = axes[0, col]
            ax_orig.imshow(raw_img, cmap="gray", vmin=0, vmax=255)
            ax_orig.set_title(f"{CLASS_NAMES[cls_idx]}", fontsize=9.5, fontweight="bold", color="#1F2328")
            ax_orig.axis("off")

            ax_cam = axes[1, col]
            ax_cam.imshow(raw_img, cmap="gray", vmin=0, vmax=255)
            ax_cam.imshow(cam, cmap="jet", alpha=0.48, vmin=0, vmax=1)
            ax_cam.set_title(f"Pred: {CLASS_NAMES[pred_idx]} ({conf * 100:.0f}%)", fontsize=8.3, color="#1A7F37", fontweight="bold")
            ax_cam.axis("off")

        # Row 2: Wrong cases with Grad-CAM overlay
        for col in range(len(CLASS_NAMES)):
            ax_w = axes[2, col]
            ax_w.axis("off")
            if col >= len(wrong_samples):
                continue
            sample_idx, true_y, pred_y, conf = wrong_samples[col]
            raw_img = raw_images[sample_idx]
            tensor = transform(Image.fromarray(raw_img)).unsqueeze(0).to(device)
            cam, _, _, _ = gradcam.compute(tensor, class_idx=pred_y)
            ax_w.imshow(raw_img, cmap="gray", vmin=0, vmax=255)
            ax_w.imshow(cam, cmap="jet", alpha=0.48, vmin=0, vmax=1)
            ax_w.set_title(
                f"P: {CLASS_NAMES[pred_y]} ({conf * 100:.0f}%)\nT: {CLASS_NAMES[true_y]}",
                fontsize=8.0,
                color="#CF222E",
                fontweight="bold",
            )
    finally:
        gradcam.remove()

    # Row labels on left margin
    row_labels = ["Original\n(Correct)", "Grad-CAM\n(Correct)", "Grad-CAM\n(Wrong Cases)"]
    for row_idx, text in enumerate(row_labels):
        axes[row_idx, 0].text(
            -0.22, 0.5, text, transform=axes[row_idx, 0].transAxes,
            fontsize=9.5, fontweight="bold", va="center", ha="right", color="#0969DA" if row_idx < 2 else "#CF222E",
        )

    return _save_figure(fig, output_path, dpi=dpi)


def run_filter_visualizations(
    checkpoint_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path = "figures",
    split: str = "PrivateTest",
    device_pref: str = "auto",
    seed: int = 42,
) -> dict[str, Path]:
    """Generate all 4 mandatory filter/feature-map/Grad-CAM figures at 300 DPI."""
    seed_everything(seed)
    device = get_device(device_pref)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, _ = load_model_from_checkpoint(checkpoint_path, device)
    splits = load_splits(data_path)
    mean, std = compute_train_statistics(splits["Training"][0])
    transform = eval_transform(mean=mean, std=std)
    raw_images, labels = splits[split]

    correct_by_class, wrong_samples = _select_representative_samples(
        model, raw_images, labels, transform, device
    )

    paths = {
        "first_conv_filters": visualize_first_conv_filters(model, out / "viz_first_conv_filters.png"),
        "final_conv_filters": visualize_final_conv_filters(model, out / "viz_final_conv_filters_weights.png"),
        "final_conv_feature_maps": visualize_final_conv_feature_maps(
            model, raw_images, labels, correct_by_class, transform, device, out / "viz_final_conv_feature_maps.png"
        ),
        "gradcam_7emotions": visualize_gradcam_7emotions(
            model, raw_images, labels, correct_by_class, wrong_samples, transform, device, out / "viz_gradcam_7emotions.png"
        ),
    }
    for key, path in paths.items():
        print(f"[OK] {key}: {path}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth", help="Path to best_model.pth")
    parser.add_argument("--data-path", default="data/fer2013.csv", help="Path to FER2013 CSV or split folder")
    parser.add_argument("--output-dir", default="figures", help="Directory for 300-DPI figures")
    parser.add_argument("--split", default="PrivateTest", choices=["PublicTest", "PrivateTest"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_filter_visualizations(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        output_dir=args.output_dir,
        split=args.split,
        device_pref=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

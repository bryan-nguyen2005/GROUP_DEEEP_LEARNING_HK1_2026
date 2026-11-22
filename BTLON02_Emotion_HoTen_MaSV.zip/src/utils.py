"""Shared runtime utilities: reproducibility seeds, device selection, and parameter stats."""

from __future__ import annotations

import os
import random
import torch
from torch import nn
import numpy as np


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = torch.cuda.is_available()


def get_device(prefer: str | None = None) -> torch.device:
    """Select computation device automatically (cuda > mps > cpu) unless overridden."""
    if prefer is not None and prefer.lower() != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_description(device: torch.device) -> str:
    """Return a concise human-readable hardware string for experiment logs."""
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
    if device.type == "mps":
        return "mps (Apple Silicon)"
    return "cpu"


def count_trainable_params(model: nn.Module) -> int:
    """Count parameters with requires_grad=True."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def count_total_params(model: nn.Module) -> int:
    """Count all model parameters."""
    return sum(param.numel() for param in model.parameters())


def format_duration(seconds: float) -> str:
    """Format elapsed time in seconds as a readable string."""
    total = max(0.0, float(seconds))
    if total < 60.0:
        return f"{total:.1f}s"
    minutes, rem = divmod(total, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {rem:04.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes):02d}m {rem:04.1f}s"

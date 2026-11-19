"""FER2013 preprocessing. Statistics are fitted by the training loader."""

from torchvision import transforms


def get_transforms(mode: str, *, mean: float = 0.0, std: float = 1.0):
    """Build a transform for a grayscale Pillow image."""
    if mode not in {"none", "standard", "strong"}:
        raise ValueError(f"Unknown augmentation mode: {mode}")

    steps = []
    if mode in {"standard", "strong"}:
        steps.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
            ),
        ])
    steps.extend([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])
    if mode == "strong":
        steps.append(transforms.RandomErasing(p=0.25))
    return transforms.Compose(steps)


def eval_transform(*, mean: float = 0.0, std: float = 1.0):
    """Scale and normalize without random augmentation."""
    return get_transforms("none", mean=mean, std=std)


def get_tta_views(images):
    """Return original, mirrored and center-cropped views of a normalized batch."""
    height, width = images.shape[-2:]
    cropped = transforms.functional.center_crop(images, [height - 4, width - 4])
    resized = transforms.functional.resize(cropped, [height, width], antialias=True)
    return images, transforms.functional.hflip(images), resized

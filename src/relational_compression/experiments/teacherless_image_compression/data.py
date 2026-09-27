"""Dataset and reproducibility helpers for reconstruction-trained image experiments."""

import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sized, cast

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, random_split

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageDirectoryDataset(Dataset[Tensor]):
    """Load cropped RGB images from a directory tree."""

    def __init__(self, root: Path, image_size: int) -> None:
        """Initialize the image directory dataset.

        Args:
            root: Directory recursively searched for image files.
            image_size: Square side length after center cropping and resizing.

        """
        self.paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES)

        if not self.paths:
            raise ValueError(f"No images found under {root}")

        self.image_size = image_size

    def __len__(self) -> int:
        """Return the number of discovered images."""
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        """Load, center-crop, and normalize one RGB image."""
        with Image.open(self.paths[index]) as opened_image:
            image: Image.Image = opened_image.convert("RGB")
            width, height = image.size
            side = min(width, height)
            left = (width - side) // 2
            top = (height - side) // 2
            image = image.crop((left, top, left + side, top + side))
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
            array = np.asarray(image, dtype=np.float32) / 255.0

        return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def split_dataset(
    dataset: Dataset[Tensor], validation_fraction: float, seed: int
) -> tuple[Dataset[Tensor], Dataset[Tensor]]:
    """Split an image dataset reproducibly into training and validation subsets."""
    val_size = max(1, int(round(len(cast(Sized, dataset)) * validation_fraction)))
    train_size = len(cast(Sized, dataset)) - val_size

    if train_size < 1:
        raise ValueError("Dataset is too small for the requested validation split")

    splits = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    return splits[0], splits[1]


class ImageOnlyDataset(Dataset[Tensor]):
    """Adapt datasets with labels or mappings to image tensors only."""

    def __init__(self, dataset: Dataset[Any]) -> None:
        """Initialize the adapter around an image-bearing dataset."""
        self.dataset = dataset

    def __len__(self) -> int:
        """Return the number of wrapped samples."""
        return len(cast(Sized, self.dataset))

    def __getitem__(self, index: int) -> Tensor:
        """Return the image tensor from one wrapped sample."""
        item = self.dataset[index]

        if isinstance(item, tuple):
            return cast(Tensor, item[0])

        if isinstance(item, Mapping):
            return cast(Tensor, item["image"])

        return cast(Tensor, item)


def build_flowers102_datasets(root: Path, image_size: int, *, download: bool) -> dict[str, Dataset[Tensor]]:
    """Build the train, validation, and test Flowers102 image datasets."""
    from torchvision.datasets import Flowers102
    from torchvision.transforms import v2

    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomResizedCrop(size=image_size),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.05, hue=0.05, saturation=0.05, contrast=0.05),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    eval_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=image_size),
            v2.CenterCrop(size=image_size),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    # Swap the official training and test splits to use about a 75/12.5/12.5 train/validation/test split.
    return {
        "train": ImageOnlyDataset(
            Flowers102(root=str(root), split="test", transform=train_transform, download=download)
        ),
        "valid": ImageOnlyDataset(Flowers102(root=str(root), split="val", transform=eval_transform, download=download)),
        "test": ImageOnlyDataset(
            Flowers102(root=str(root), split="train", transform=eval_transform, download=download)
        ),
    }


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

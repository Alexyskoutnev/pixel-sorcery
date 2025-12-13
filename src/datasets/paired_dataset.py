"""
Dataset loader for paired image-to-image translation.

Handles loading paired images where:
- Input: source image (e.g., unedited photo)
- Target: target image (e.g., professionally edited photo)

Key features:
- Synchronized augmentation (same transform applied to both images)
- Configurable image size
- Train/validation split support
- Memory-efficient loading
"""

import random
from pathlib import Path
from typing import Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF

from src.utils.logging import logger


class PairedImageDataset(Dataset):
    """
    Dataset for paired image-to-image translation.

    Expects images in format:
        {id}_src.jpg  (source/input)
        {id}_tar.jpg  (target/output)

    Args:
        root_dir: Directory containing images
        image_size: Target size for images (height, width) or single int for square
        augment: Whether to apply augmentation
        normalize: Whether to normalize to [0, 1]
        src_suffix: Suffix for source images (default: "_src")
        tar_suffix: Suffix for target images (default: "_tar")
        extensions: Valid image extensions
    """

    def __init__(
        self,
        root_dir: str,
        image_size: int = 512,
        augment: bool = True,
        normalize: bool = True,
        src_suffix: str = "_src",
        tar_suffix: str = "_tar",
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.augment = augment
        self.normalize = normalize
        self.src_suffix = src_suffix
        self.tar_suffix = tar_suffix

        # Find all paired images
        self.pairs = self._find_pairs(extensions)

        if len(self.pairs) == 0:
            raise ValueError(f"No paired images found in {root_dir}")

        logger.info(f"Found {len(self.pairs)} image pairs in {root_dir}")

    def _find_pairs(self, 
                    extensions: Tuple[str, ...]) -> List[tuple[Path, Path]]:
        """Find all valid source-target pairs."""
        pairs = []
        seen_ids = set()

        for file in self.root_dir.iterdir():
            if not file.is_file():
                continue

            # Check if it's a source file
            stem = file.stem
            if not stem.endswith(self.src_suffix):
                continue

            # Extract the ID
            img_id = stem[: -len(self.src_suffix)]
            if img_id in seen_ids:
                continue

            # Find corresponding target
            src_path = file
            tar_path = None

            for ext in extensions:
                potential_tar = self.root_dir / f"{img_id}{self.tar_suffix}{ext}"
                if potential_tar.exists():
                    tar_path = potential_tar
                    break

            if tar_path is not None:
                pairs.append((src_path, tar_path))
                seen_ids.add(img_id)

        pairs.sort(key=lambda x: x[0].stem)
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a paired sample.

        Returns:
            Dictionary with:
                - 'input': Source image tensor (C, H, W)
                - 'target': Target image tensor (C, H, W)
                - 'filename': Source filename (for reference)
        """
        src_path, tar_path = self.pairs[idx]
        src_img = Image.open(src_path).convert("RGB")
        tar_img = Image.open(tar_path).convert("RGB")
        src_tensor, tar_tensor = self._transform(src_img, tar_img)
        return {
            "input": src_tensor,
            "target": tar_tensor,
            "filename": src_path.stem,
        }

    def _transform(
        self, 
        src_img: Image.Image, 
        tar_img: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor]:
        

        src_img = TF.resize(src_img, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
        tar_img = TF.resize(tar_img, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)

        # Data augmentation (training only)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                src_img = TF.hflip(src_img)
                tar_img = TF.hflip(tar_img)

            # Random vertical flip (optional - may not make sense for real estate)
            # if random.random() > 0.5:
            #     src_img = TF.vflip(src_img)
            #     tar_img = TF.vflip(tar_img)

            # Random rotation (small angles)
            if random.random() > 0.7:
                angle = random.uniform(-5, 5)
                src_img = TF.rotate(src_img, angle, interpolation=TF.InterpolationMode.BILINEAR)
                tar_img = TF.rotate(tar_img, angle, interpolation=TF.InterpolationMode.BILINEAR)

        # Convert to tensor (this also scales to [0, 1])
        src_tensor = TF.to_tensor(src_img)
        tar_tensor = TF.to_tensor(tar_img)

        return src_tensor, tar_tensor


class PairedImageDataModule:
    """
    Data module that handles train/val splits and DataLoader creation.

    Usage:
        data = PairedImageDataModule(
            root_dir="path/to/images",
            image_size=512,
            batch_size=8,
            val_split=0.1,
        )
        train_loader = data.train_dataloader()
        val_loader = data.val_dataloader()
    """

    def __init__(
        self,
        root_dir: str,
        image_size: int = 512,
        batch_size: int = 8,
        val_split: float = 0.1,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

        # Create datasets
        self._setup()

    def _setup(self):

        full_dataset = PairedImageDataset(
            root_dir=self.root_dir,
            image_size=self.image_size,
            augment=True,
        )

        # Split into train and val
        total_size = len(full_dataset)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size

        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

        # Create validation dataset without augmentation
        # We need to wrap it to disable augmentation
        self.val_dataset_no_aug = PairedImageDataset(
            root_dir=self.root_dir,
            image_size=self.image_size,
            augment=False,  # No augmentation for validation
        )

        # Get the validation indices
        val_indices = self.val_dataset.indices
        self.val_dataset_no_aug = torch.utils.data.Subset(
            self.val_dataset_no_aug, val_indices
        )

        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")

    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation DataLoader (no augmentation, no shuffle)."""
        return DataLoader(
            self.val_dataset_no_aug,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


def get_dataloaders(
    root_dir: str,
    image_size: int = 512,
    batch_size: int = 8,
    val_split: float = 0.1,
    num_workers: int = 16,
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience function to get train and val dataloaders.

    Args:
        root_dir: Path to images directory
        image_size: Target image size
        batch_size: Batch size
        val_split: Fraction for validation
        num_workers: Number of data loading workers

    Returns:
        (train_dataloader, val_dataloader)
    """
    data_module = PairedImageDataModule(
        root_dir=root_dir,
        image_size=image_size,
        batch_size=batch_size,
        val_split=val_split,
        num_workers=num_workers,
    )
    return data_module.train_dataloader(), data_module.val_dataloader()

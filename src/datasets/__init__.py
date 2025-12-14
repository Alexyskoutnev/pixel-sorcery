"""
Datasets module - contains data loading utilities.

Usage:
    from src.datasets import PairedImageDataset, get_dataloaders

    # Quick setup
    train_loader, val_loader = get_dataloaders(
        root_dir="path/to/images",
        image_size=512,
        batch_size=8,
    )

    # Or use the full data module
    data = PairedImageDataModule(
        root_dir="path/to/images",
        image_size=512,
        batch_size=8,
    )
"""

from .paired_dataset import (
    PairedImageDataset,
    PairedImageDataModule,
    get_dataloaders,
)

__all__ = [
    "PairedImageDataset",
    "PairedImageDataModule",
    "get_dataloaders",
]

"""
Models module - contains all model architectures.

To add a new model:
1. Create a new file (e.g., my_model.py)
2. Inherit from BaseModel
3. Implement the forward() method
4. Add to MODEL_REGISTRY in this file

Usage:
    from src.models import get_model, UNet, BaseModel

    # Get model by name
    model = get_model("unet", base_channels=64)

    # Or instantiate directly
    model = UNet(in_channels=3, out_channels=3)

    # Get discriminator for GAN training
    from src.models import get_discriminator
    discriminator = get_discriminator("patch")
"""

from .base import BaseModel, ResidualWrapper
from .unet import UNet, UNetSmall, UNetLarge, get_model, MODEL_REGISTRY
from .discriminator import (
    PatchDiscriminator,
    MultiScaleDiscriminator,
    get_discriminator,
    DISCRIMINATOR_REGISTRY,
)

__all__ = [
    # Base classes
    "BaseModel",
    "ResidualWrapper",
    # U-Net variants (generators)
    "UNet",
    "UNetSmall",
    "UNetLarge",
    # Discriminators
    "PatchDiscriminator",
    "MultiScaleDiscriminator",
    # Utilities
    "get_model",
    "get_discriminator",
    "MODEL_REGISTRY",
    "DISCRIMINATOR_REGISTRY",
]

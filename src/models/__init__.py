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
"""

from .base import BaseModel, ResidualWrapper
from .unet import UNet, UNetSmall, UNetLarge, get_model, MODEL_REGISTRY

__all__ = [
    # Base classes
    "BaseModel",
    "ResidualWrapper",
    # U-Net variants
    "UNet",
    "UNetSmall",
    "UNetLarge",
    # Utilities
    "get_model",
    "MODEL_REGISTRY",
]


from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):

    def __init__(self):
        super().__init__()
        self._name = self.__class__.__name__

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (B, C, H, W)
               Typically (batch_size, 3, height, width) for RGB images
               Values should be normalized to [0, 1] or [-1, 1]

        Returns:
            Output tensor of same shape as input (B, C, H, W)
        """
        pass

    @property
    def name(self) -> str:
        return self._name

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_params": self.get_num_params(),
            "num_trainable_params": self.get_num_params(trainable_only=True),
        }

    def get_num_params(self, 
                       trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_memory_usage_mb(self) -> float:
        """
        Estimate model memory usage in MB.

        Returns:
            Estimated memory in megabytes
        """
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def save(self, 
             path: str):
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.get_config(),
        }, path)

    def load(self, 
             path: str, 
             strict: bool = True):
        checkpoint = torch.load(path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.load_state_dict(checkpoint['state_dict'], strict=strict)
        else:
            # Handle case where only state_dict was saved
            self.load_state_dict(checkpoint, strict=strict)


class ResidualWrapper(BaseModel):
    """
    Wrapper that adds residual learning to any model.

    Instead of learning: output = model(input)
    Learns: output = input + model(input)

    This is useful when input and output are similar (like photo enhancement).
    The model only needs to learn the "difference" or "edit".

    Example:
        base_model = UNet(in_channels=3, out_channels=3)
        model = ResidualWrapper(base_model)
    """

    def __init__(self, model: BaseModel):
        super().__init__()
        self.model = model
        self._name = f"Residual{model.name}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor

        Returns:
            x + model(x), clamped to valid range
        """
        residual = self.model(x)
        output = x + residual
        # Clamp to valid image range [0, 1]
        return torch.clamp(output, 0, 1)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config["wrapped_model"] = self.model.get_config()
        config["residual"] = True
        return config

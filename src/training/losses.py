"""
Loss functions for image-to-image translation.

Available losses:
- L1Loss: Simple pixel-wise L1 loss
- L2Loss: Simple pixel-wise L2 (MSE) loss
- PerceptualLoss: VGG-based perceptual loss
- SSIMLoss: Structural similarity loss
- CombinedLoss: Weighted combination of multiple losses

Usage:
    from src.training.losses import CombinedLoss, get_loss

    # Use combined loss
    loss_fn = CombinedLoss(
        l1_weight=1.0,
        perceptual_weight=0.1,
        ssim_weight=0.1,
    )

    # Or get by name
    loss_fn = get_loss("l1")
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class L1Loss(nn.Module):
    """
    L1 (Mean Absolute Error) loss.

    Simple and stable, but can produce blurry results.
    Good as a baseline or combined with perceptual loss.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class L2Loss(nn.Module):
    """
    L2 (Mean Squared Error) loss.

    Penalizes large errors more than L1.
    Often produces blurrier results than L1.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.

    Compares high-level features rather than raw pixels.
    Produces sharper, more perceptually pleasing results.

    Args:
        layers: Which VGG layers to use for comparison
        weights: Weight for each layer's contribution
        normalize: Whether to normalize input to VGG's expected range
    """

    def __init__(
        self,
        layers: List[int] = [3, 8, 15, 22],  # relu1_2, relu2_2, relu3_2, relu4_2
        weights: Optional[List[float]] = None,
        normalize: bool = True,
    ):
        super().__init__()

        # Load pretrained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        # Freeze VGG weights
        for param in vgg.parameters():
            param.requires_grad = False

        # Store layers we want to extract features from
        self.layers = layers
        self.weights = weights if weights else [1.0 / len(layers)] * len(layers)

        # Create feature extractors for each layer
        self.feature_extractors = nn.ModuleList()
        prev_layer = 0
        for layer_idx in layers:
            self.feature_extractors.append(
                nn.Sequential(*list(vgg.children())[prev_layer:layer_idx + 1])
            )
            prev_layer = layer_idx + 1

        self.normalize = normalize

        # VGG normalization parameters
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input from [0, 1] to VGG's expected range."""
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            pred: Predicted image (B, 3, H, W) in [0, 1]
            target: Target image (B, 3, H, W) in [0, 1]

        Returns:
            Scalar loss value
        """
        if self.normalize:
            pred = self._normalize(pred)
            target = self._normalize(target)

        loss = 0.0
        pred_features = pred
        target_features = target

        for extractor, weight in zip(self.feature_extractors, self.weights):
            pred_features = extractor(pred_features)
            target_features = extractor(target_features)
            loss += weight * F.l1_loss(pred_features, target_features)

        return loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) loss.

    Measures structural similarity between images.
    SSIM ranges from -1 to 1, where 1 means identical.
    Loss = 1 - SSIM (so we minimize it).

    Args:
        window_size: Size of the Gaussian window
        sigma: Standard deviation of Gaussian window
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = 3

        # Create Gaussian window
        self.register_buffer("window", self._create_window(window_size, sigma, 3))

    def _create_window(self, window_size: int, sigma: float, channel: int) -> torch.Tensor:
        """Create a Gaussian window."""
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2

        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        window = g.unsqueeze(1) * g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)
        window = window.expand(channel, 1, window_size, window_size).contiguous()

        return window

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            1 - SSIM (scalar loss)
        """
        channel = pred.size(1)
        window = self.window

        if channel != self.channel:
            window = self._create_window(self.window_size, self.sigma, channel)
            window = window.to(pred.device)

        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim_map.mean()


class CombinedLoss(nn.Module):
    """
    Weighted combination of multiple losses.

    This is typically the best choice for image-to-image translation:
    - L1 for pixel accuracy
    - Perceptual for visual quality
    - SSIM for structural preservation

    Args:
        l1_weight: Weight for L1 loss
        l2_weight: Weight for L2 loss
        perceptual_weight: Weight for perceptual loss
        ssim_weight: Weight for SSIM loss

    Example:
        loss_fn = CombinedLoss(
            l1_weight=1.0,
            perceptual_weight=0.1,
            ssim_weight=0.1,
        )
        loss, loss_dict = loss_fn(pred, target)
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        l2_weight: float = 0.0,
        perceptual_weight: float = 0.0,
        ssim_weight: float = 0.0,
    ):
        super().__init__()

        self.weights = {
            "l1": l1_weight,
            "l2": l2_weight,
            "perceptual": perceptual_weight,
            "ssim": ssim_weight,
        }

        # Only create loss functions that will be used
        self.losses = nn.ModuleDict()
        if l1_weight > 0:
            self.losses["l1"] = L1Loss()
        if l2_weight > 0:
            self.losses["l2"] = L2Loss()
        if perceptual_weight > 0:
            self.losses["perceptual"] = VGGPerceptualLoss()
        if ssim_weight > 0:
            self.losses["ssim"] = SSIMLoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            pred: Predicted image
            target: Target image

        Returns:
            (total_loss, loss_dict) where loss_dict contains individual losses
        """
        total_loss = 0.0
        loss_dict = {}

        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(pred, target)
            weighted_loss = self.weights[name] * loss_value
            total_loss += weighted_loss
            loss_dict[name] = loss_value.item()

        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict


LOSS_REGISTRY = {
    "l1": L1Loss,
    "l2": L2Loss,
    "mse": L2Loss,
    "perceptual": VGGPerceptualLoss,
    "vgg": VGGPerceptualLoss,
    "ssim": SSIMLoss,
    "combined": CombinedLoss,
}


def get_loss(name: str, **kwargs) -> nn.Module:
    """
    Get a loss function by name.

    Args:
        name: Loss name (e.g., "l1", "perceptual", "combined")
        **kwargs: Additional arguments for the loss function

    Returns:
        Loss function module
    """
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name](**kwargs)

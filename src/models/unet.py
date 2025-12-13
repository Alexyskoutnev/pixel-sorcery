"""
U-Net implementation for image-to-image translation.

This is a clean, configurable U-Net that can be used for:
- Image enhancement / color grading
- Image restoration
- Any paired image-to-image task

Architecture:
    Encoder (downsampling) -> Bottleneck -> Decoder (upsampling)
    with skip connections between encoder and decoder at each level.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel


class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv -> Norm -> Activation -> Conv -> Norm -> Activation

    This is the fundamental building block of U-Net.
    Two convolutions allow the network to learn more complex features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_batchnorm: bool = True,
        activation: str = "relu",
    ):
        super().__init__()

        # Select activation function
        if activation == "relu":
            act_fn = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "gelu":
            act_fn = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build the block
        layers = []

        # First conv
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(act_fn)

        # Second conv
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(act_fn if activation != "gelu" else nn.GELU())

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """
    Encoder block: Downsample -> ConvBlock

    Reduces spatial dimensions by 2x while increasing channel depth.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        activation: str = "relu",
        downsample_mode: str = "maxpool",  # "maxpool" or "stride"
    ):
        super().__init__()

        # Downsampling method
        if downsample_mode == "maxpool":
            self.downsample = nn.MaxPool2d(2)
            self.conv = ConvBlock(in_channels, out_channels, use_batchnorm=use_batchnorm, activation=activation)
        elif downsample_mode == "stride":
            # Strided convolution for downsampling (learnable)
            self.downsample = nn.Identity()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
                nn.ReLU(inplace=True) if activation == "relu" else nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
                nn.ReLU(inplace=True) if activation == "relu" else nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            raise ValueError(f"Unknown downsample_mode: {downsample_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        return self.conv(x)


class UpBlock(nn.Module):
    """
    Decoder block: Upsample -> Concat skip -> ConvBlock

    Increases spatial dimensions by 2x while decreasing channel depth.
    Concatenates with skip connection from encoder for detail preservation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        activation: str = "relu",
        upsample_mode: str = "bilinear",  # "bilinear", "nearest", or "transpose"
    ):
        super().__init__()

        # Upsampling method
        if upsample_mode in ["bilinear", "nearest"]:
            self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=True if upsample_mode == "bilinear" else None)
        elif upsample_mode == "transpose":
            # Transposed convolution (learnable upsampling)
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            raise ValueError(f"Unknown upsample_mode: {upsample_mode}")

        # Conv after concatenation with skip connection
        # in_channels + out_channels because we concat skip connection
        self.conv = ConvBlock(
            in_channels + out_channels,  # After concat
            out_channels,
            use_batchnorm=use_batchnorm,
            activation=activation,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input from previous decoder layer
            skip: Skip connection from corresponding encoder layer
        """
        x = self.upsample(x)

        # Handle size mismatch (can happen with odd dimensions)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

        # Concatenate along channel dimension
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(BaseModel):
    """
    U-Net for image-to-image translation.

    A flexible U-Net implementation with configurable:
    - Number of encoder/decoder levels (depth)
    - Base channel count (width)
    - Normalization, activation, up/downsampling methods

    Architecture diagram (depth=4, base_channels=64):

        Input (3) ──► [Conv 64] ──────────────────────────────► [Conv 64] ──► Output (3)
                         │                                           ▲
                         ▼                                           │
                    [Down 128] ──────────────────────────► [Up 64] ──┘
                         │                                     ▲
                         ▼                                     │
                    [Down 256] ──────────────────► [Up 128] ──┘
                         │                             ▲
                         ▼                             │
                    [Down 512] ──────► [Up 256] ──────┘
                         │                 ▲
                         ▼                 │
                    [Bottleneck 512] ──────┘

    Args:
        in_channels: Number of input channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        base_channels: Number of channels in first layer (doubles each level)
        depth: Number of encoder/decoder levels (4-5 is typical)
        use_batchnorm: Whether to use batch normalization
        activation: Activation function ("relu", "leaky_relu", "gelu")
        upsample_mode: Upsampling method ("bilinear", "nearest", "transpose")
        downsample_mode: Downsampling method ("maxpool", "stride")
        final_activation: Final activation ("sigmoid", "tanh", or None)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        depth: int = 4,
        use_batchnorm: bool = True,
        activation: str = "relu",
        upsample_mode: str = "bilinear",
        downsample_mode: str = "maxpool",
        final_activation: Optional[str] = "sigmoid",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth
        self.final_activation_name = final_activation

        # Calculate channel sizes for each level
        # e.g., depth=4, base=64: [64, 128, 256, 512]
        channels = [base_channels * (2 ** i) for i in range(depth)]

        # Initial convolution (no downsampling)
        self.initial_conv = ConvBlock(
            in_channels, channels[0],
            use_batchnorm=use_batchnorm,
            activation=activation,
        )

        # Encoder (downsampling path)
        self.encoders = nn.ModuleList()
        for i in range(depth - 1):
            self.encoders.append(
                DownBlock(
                    channels[i], channels[i + 1],
                    use_batchnorm=use_batchnorm,
                    activation=activation,
                    downsample_mode=downsample_mode,
                )
            )

        # Bottleneck (deepest layer)
        self.bottleneck = ConvBlock(
            channels[-1], channels[-1],
            use_batchnorm=use_batchnorm,
            activation=activation,
        )

        # Decoder (upsampling path)
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.decoders.append(
                UpBlock(
                    channels[i], channels[i - 1],
                    use_batchnorm=use_batchnorm,
                    activation=activation,
                    upsample_mode=upsample_mode,
                )
            )

        # Final convolution to output channels
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

        # Final activation
        if final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        elif final_activation == "tanh":
            self.final_activation = nn.Tanh()
        else:
            self.final_activation = nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, C, H, W)
        """
        # Store skip connections
        skips = []

        # Initial convolution
        x = self.initial_conv(x)
        skips.append(x)

        # Encoder path
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path (reverse order of skips, excluding the last one which is bottleneck input)
        skips = skips[:-1][::-1]  # Reverse and exclude last
        for decoder, skip in zip(self.decoders, skips):
            x = decoder(x, skip)

        # Final convolution and activation
        x = self.final_conv(x)
        x = self.final_activation(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "base_channels": self.base_channels,
            "depth": self.depth,
            "final_activation": self.final_activation_name,
        })
        return config


class UNetSmall(UNet):
    """Small U-Net variant for faster training/inference."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=32,
            depth=4,
            **kwargs
        )


class UNetLarge(UNet):
    """Large U-Net variant for higher quality."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=64,
            depth=5,
            **kwargs
        )


MODEL_REGISTRY = {
    "unet": UNet,
    "unet_small": UNetSmall,
    "unet_large": UNetLarge,
}


def get_model(name: str, **kwargs) -> BaseModel:
    """
    Get a model by name.

    Args:
        name: Model name (e.g., "unet", "unet_small")
        **kwargs: Model configuration

    Returns:
        Instantiated model
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)

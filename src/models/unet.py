"""
U-Net Architecture for Image-to-Image Translation

This module implements a flexible U-Net, originally designed for biomedical image
segmentation (Ronneberger et al., 2015), now widely used for image-to-image tasks.

================================================================================
CORE CONCEPT: WHY U-NET WORKS FOR IMAGE TRANSLATION
================================================================================

The key insight is that image-to-image tasks need BOTH:
  1. HIGH-LEVEL UNDERSTANDING: "What is in this image?" (semantics)
  2. LOW-LEVEL DETAILS: "Where exactly are the edges?" (spatial precision)

Traditional CNNs lose spatial information as they go deeper:

  Input (512x512) → Conv → Pool → Conv → Pool → Conv → Pool → Output (64x64)
                    ↓
            Lost all the fine details!

U-Net solves this with SKIP CONNECTIONS:

  Encoder                              Decoder
  (compress)                           (expand)

  512x512 ─────────────────────────────────────────► 512x512
      │                                                 ▲
      ▼                                                 │
  256x256 ─────────────────────────────────────► 256x256
      │                                             ▲
      ▼                                             │
  128x128 ─────────────────────────► 128x128 ──────┘
      │                                 ▲
      ▼                                 │
    64x64 ────► Bottleneck ────► 64x64 ┘

The horizontal arrows are "skip connections" - they pass fine details directly
from encoder to decoder, bypassing the information bottleneck.

================================================================================
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel



class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv -> Norm -> Activation -> Conv -> Norm -> Activation

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        WHY TWO CONVOLUTIONS?                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  One 3x3 conv has a "receptive field" of 3x3 pixels.                   │
    │  Two 3x3 convs have a receptive field of 5x5 pixels.                   │
    │                                                                         │
    │  But two 3x3 convs have FEWER parameters than one 5x5 conv:            │
    │    - Two 3x3: 2 × (3×3×C×C) = 18C²                                     │
    │    - One 5x5: 5×5×C×C = 25C²                                           │
    │                                                                         │
    │  Plus, two convs = two nonlinearities = more expressive power!         │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      WHAT IS BATCH NORMALIZATION?                       │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Problem: As training progresses, the distribution of layer inputs     │
    │  changes (called "internal covariate shift"). This makes training      │
    │  unstable and slow.                                                    │
    │                                                                         │
    │  Solution: Normalize each batch to have mean=0, variance=1:            │
    │                                                                         │
    │            x - mean(x)                                                  │
    │    x̂ = ─────────────────  (then scale and shift with learned γ, β)    │
    │          sqrt(var(x) + ε)                                              │
    │                                                                         │
    │  Benefits:                                                              │
    │    - Faster training (can use higher learning rates)                   │
    │    - Acts as regularization (reduces overfitting)                      │
    │    - Makes the network less sensitive to initialization                │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                       ACTIVATION FUNCTIONS                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Without activations, stacking layers = one linear transformation.     │
    │  Activations add NON-LINEARITY, letting us learn complex patterns.     │
    │                                                                         │
    │  ReLU: f(x) = max(0, x)                                                │
    │    ╭────────╮                                                          │
    │    │      ╱ │  Simple, fast, but "dead neurons" if x < 0 always       │
    │    │─────╱  │                                                          │
    │    ╰────────╯                                                          │
    │                                                                         │
    │  LeakyReLU: f(x) = max(0.2x, x)                                        │
    │    ╭────────╮                                                          │
    │    │    ╱   │  Small slope for x < 0 prevents dead neurons            │
    │    │__╱     │  Popular in GANs and discriminators                      │
    │    ╰────────╯                                                          │
    │                                                                         │
    │  GELU: f(x) = x · Φ(x)  (Φ = Gaussian CDF)                            │
    │    ╭────────╮                                                          │
    │    │   _╱   │  Smooth, used in transformers (BERT, GPT)               │
    │    │__╱     │  Slightly better for some tasks, slightly slower         │
    │    ╰────────╯                                                          │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
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
        # inplace=True saves memory by modifying tensor directly (no copy)
        if activation == "relu":
            act_fn = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU(0.2, inplace=True)  # 0.2 is the negative slope
        elif activation == "gelu":
            act_fn = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build the block: Conv → BatchNorm → Activation → Conv → BatchNorm → Activation
        layers = []

        # First convolution: changes channel count (in_channels → out_channels)
        # kernel_size=3, padding=1 preserves spatial dimensions: H×W → H×W
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(act_fn)

        # Second convolution: refines features (out_channels → out_channels)
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(act_fn if activation != "gelu" else nn.GELU())

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ==============================================================================
# DOWNBLOCK: The Encoder (Compression) Block
# ==============================================================================

class DownBlock(nn.Module):
    """
    Encoder block: Downsample -> ConvBlock

    Reduces spatial dimensions by 2x while increasing channel depth.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     WHY DOWNSAMPLE (COMPRESS)?                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Think of it like summarizing a book:                                  │
    │                                                                         │
    │    512x512 = "This is a photo of a living room with a brown couch,    │
    │               beige walls, two windows on the left..."                 │
    │                                                                         │
    │    256x256 = "Living room, couch, windows"                             │
    │                                                                         │
    │    64x64   = "Indoor residential space"                                │
    │                                                                         │
    │  Each level captures more ABSTRACT features but loses SPATIAL DETAIL.  │
    │  We compensate by increasing CHANNELS (more "feature detectors").      │
    │                                                                         │
    │  512×512×64  → 256×256×128 → 128×128×256 → 64×64×512                   │
    │  (4M values)   (8M values)   (8M values)   (2M values)                 │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      DOWNSAMPLING METHODS                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  1. MAX POOLING (default, recommended for most cases)                  │
    │     ─────────────────────────────────────────────────                  │
    │     Takes the MAX value in each 2×2 window:                            │
    │                                                                         │
    │     ┌───┬───┬───┬───┐         ┌───┬───┐                                │
    │     │ 1 │ 3 │ 2 │ 1 │         │   │   │                                │
    │     ├───┼───┼───┼───┤   →     │ 4 │ 5 │  (kept the max values)        │
    │     │ 4 │ 2 │ 5 │ 3 │         │   │   │                                │
    │     └───┴───┴───┴───┘         └───┴───┘                                │
    │                                                                         │
    │     Pros: Simple, no learnable parameters, keeps strong activations    │
    │     Cons: Loses some information, can't learn what to keep             │
    │                                                                         │
    │  2. STRIDED CONVOLUTION (learnable)                                    │
    │     ────────────────────────────────                                   │
    │     Conv with stride=2 moves kernel by 2 pixels each step:             │
    │                                                                         │
    │     ┌───┬───┬───┬───┐         ┌───┬───┐                                │
    │     │ ■ │ ■ │   │   │         │   │   │                                │
    │     ├───┼───┼───┼───┤   →     │ a │ b │  (learned weighted sums)      │
    │     │ ■ │ ■ │   │   │         │   │   │                                │
    │     └───┴───┴───┴───┘         └───┴───┘                                │
    │                                                                         │
    │     Pros: Learnable (network decides what's important)                 │
    │     Cons: More parameters, can create checkerboard artifacts           │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
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
            # MaxPool2d(2) = 2×2 window, stride 2 → halves spatial dimensions
            self.downsample = nn.MaxPool2d(2)
            self.conv = ConvBlock(in_channels, out_channels, use_batchnorm=use_batchnorm, activation=activation)

        elif downsample_mode == "stride":
            # Strided convolution: kernel_size=4, stride=2, padding=1
            # Output size = (input_size - 4 + 2*1) / 2 + 1 = input_size / 2
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


# ==============================================================================
# UPBLOCK: The Decoder (Expansion) Block
# ==============================================================================

class UpBlock(nn.Module):
    """
    Decoder block: Upsample -> Concat skip -> ConvBlock

    Increases spatial dimensions by 2x while decreasing channel depth.
    Concatenates with skip connection from encoder for detail preservation.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   THE MAGIC OF SKIP CONNECTIONS                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Problem: The bottleneck (64×64) has lost all fine details.            │
    │  How can we reconstruct a sharp 512×512 image from it?                 │
    │                                                                         │
    │  Solution: SKIP CONNECTIONS!                                            │
    │                                                                         │
    │  We CONCATENATE the upsampled features with encoder features:          │
    │                                                                         │
    │    Encoder (has details)     Decoder (has semantics)                   │
    │    ┌──────────────────┐      ┌──────────────────┐                      │
    │    │  256×256×128     │  +   │  256×256×256     │                      │
    │    │  "edges, colors" │      │  "it's a room"   │                      │
    │    └──────────────────┘      └──────────────────┘                      │
    │              │                        │                                 │
    │              └──────────┬─────────────┘                                 │
    │                         ▼                                               │
    │              ┌──────────────────┐                                       │
    │              │  256×256×384     │  (128 + 256 = 384 channels)          │
    │              │  "sharp room"    │  Best of both worlds!                │
    │              └──────────────────┘                                       │
    │                                                                         │
    │  The network learns to COMBINE:                                        │
    │    - WHERE things are (from encoder's spatial details)                 │
    │    - WHAT to do (from decoder's semantic understanding)                │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                       UPSAMPLING METHODS                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  1. BILINEAR INTERPOLATION (default, recommended)                      │
    │     ─────────────────────────────────────────────                      │
    │     Weighted average of 4 nearest pixels:                              │
    │                                                                         │
    │     ┌───┬───┐                   ┌───┬───┬───┬───┐                      │
    │     │ a │ b │                   │ a │ · │ · │ b │                      │
    │     ├───┼───┤    bilinear  →   ├───┼───┼───┼───┤                      │
    │     │ c │ d │                   │ · │ · │ · │ · │                      │
    │     └───┴───┘                   ├───┼───┼───┼───┤                      │
    │                                 │ c │ · │ · │ d │                      │
    │                                 └───┴───┴───┴───┘                      │
    │                                                                         │
    │     Pros: Smooth, no checkerboard artifacts, no learnable params       │
    │     Cons: Can be slightly blurry                                       │
    │                                                                         │
    │  2. NEAREST NEIGHBOR                                                   │
    │     ────────────────                                                   │
    │     Just copies the nearest pixel value:                               │
    │                                                                         │
    │     ┌───┬───┐                   ┌───┬───┬───┬───┐                      │
    │     │ a │ b │                   │ a │ a │ b │ b │                      │
    │     ├───┼───┤    nearest   →   ├───┼───┼───┼───┤                      │
    │     │ c │ d │                   │ a │ a │ b │ b │                      │
    │     └───┴───┘                   ├───┼───┼───┼───┤                      │
    │                                 │ c │ c │ d │ d │                      │
    │                                 └───┴───┴───┴───┘                      │
    │                                                                         │
    │     Pros: Fast, preserves exact values (good for segmentation)         │
    │     Cons: Blocky/pixelated appearance                                  │
    │                                                                         │
    │  3. TRANSPOSED CONVOLUTION (learnable)                                 │
    │     ────────────────────────────────                                   │
    │     Also called "deconvolution" (misleading name).                     │
    │     Learns how to upsample with learnable kernels.                     │
    │                                                                         │
    │     Pros: Learnable, can produce sharp results                         │
    │     Cons: CHECKERBOARD ARTIFACTS if not careful!                       │
    │                                                                         │
    │     ┌─────────────────────────────────────┐                            │
    │     │  Checkerboard artifacts happen when │                            │
    │     │  kernel_size isn't divisible by     │                            │
    │     │  stride, causing uneven overlap.    │                            │
    │     │                                     │                            │
    │     │  We use kernel=2, stride=2 to avoid │                            │
    │     │  this (2 is divisible by 2).        │                            │
    │     └─────────────────────────────────────┘                            │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
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

        # Upsampling method: doubles spatial dimensions (H×W → 2H×2W)
        if upsample_mode in ["bilinear", "nearest"]:
            # scale_factor=2 doubles the size
            # align_corners=True ensures corner pixels align exactly
            self.upsample = nn.Upsample(
                scale_factor=2,
                mode=upsample_mode,
                align_corners=True if upsample_mode == "bilinear" else None
            )
        elif upsample_mode == "transpose":
            # Transposed convolution (learnable upsampling)
            # kernel_size=2, stride=2 → exactly doubles size with no overlap issues
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            raise ValueError(f"Unknown upsample_mode: {upsample_mode}")

        # Conv AFTER concatenation with skip connection
        # Input channels = upsampled (in_channels) + skip (out_channels)
        # This is where the network learns to COMBINE encoder details + decoder semantics
        self.conv = ConvBlock(
            in_channels + out_channels,  # After concat: e.g., 512 + 256 = 768
            out_channels,                 # Output: e.g., 256
            use_batchnorm=use_batchnorm,
            activation=activation,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input from previous decoder layer (deeper, more semantic)
            skip: Skip connection from corresponding encoder layer (has spatial details)

        The skip connection is the KEY to U-Net's success!
        """
        # Step 1: Upsample to match skip connection's spatial size
        x = self.upsample(x)

        # Step 2: Handle size mismatch (can happen with odd input dimensions)
        # e.g., 513→256→128→64→128→257 (off by one!)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

        # Step 3: Concatenate along channel dimension (dim=1)
        # x: (B, in_channels, H, W)
        # skip: (B, out_channels, H, W)
        # result: (B, in_channels + out_channels, H, W)
        x = torch.cat([x, skip], dim=1)

        # Step 4: Process combined features
        return self.conv(x)


# ==============================================================================
# UNET: The Complete Architecture
# ==============================================================================

class UNet(BaseModel):
    """
    U-Net for image-to-image translation.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    U-NET ARCHITECTURE OVERVIEW                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  U-Net gets its name from its U-shaped architecture:                   │
    │                                                                         │
    │    Input (3) ──► [Conv 64] ─────────────────────────► [Conv 64] ──► Output (3)
    │                      │                                     ▲           │
    │                      ▼                                     │           │
    │                 [Down 128] ─────────────────────► [Up 64] ─┘           │
    │                      │           SKIP               ▲                  │
    │                      ▼        CONNECTIONS           │                  │
    │                 [Down 256] ───────────► [Up 128] ───┘                  │
    │                      │                     ▲                           │
    │                      ▼                     │                           │
    │                 [Down 512] ──► [Up 256] ───┘                           │
    │                      │             ▲                                   │
    │                      ▼             │                                   │
    │                 [Bottleneck 512] ──┘                                   │
    │                                                                         │
    │  LEFT SIDE (Encoder):                                                  │
    │    - Compresses spatial info: 512→256→128→64 pixels                   │
    │    - Expands channels: 64→128→256→512 features                        │
    │    - Learns "what" is in the image (semantics)                        │
    │                                                                         │
    │  BOTTOM (Bottleneck):                                                  │
    │    - Smallest spatial size, most channels                              │
    │    - Captures high-level understanding                                 │
    │    - "This is a living room that needs warm tones"                    │
    │                                                                         │
    │  RIGHT SIDE (Decoder):                                                 │
    │    - Expands spatial info: 64→128→256→512 pixels                      │
    │    - Reduces channels: 512→256→128→64 features                        │
    │    - Reconstructs the output image                                     │
    │                                                                         │
    │  SKIP CONNECTIONS (horizontal arrows):                                 │
    │    - Pass fine details from encoder to decoder                        │
    │    - Without these, output would be blurry!                           │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   CHOOSING HYPERPARAMETERS                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  BASE_CHANNELS (width): More = more capacity but more memory/compute   │
    │    - 32: Small, fast, good for simple tasks                           │
    │    - 64: Standard, good balance (recommended)                          │
    │    - 128: Large, for complex tasks                                     │
    │                                                                         │
    │  DEPTH (levels): More = larger receptive field but more memory         │
    │    - 3: For small images (128×128)                                    │
    │    - 4: Standard, works for 256-512px (recommended)                   │
    │    - 5: For large images (512-1024px)                                 │
    │                                                                         │
    │  FINAL_ACTIVATION: Constrains output range                             │
    │    - sigmoid: Output in [0, 1] - for normalized images (recommended)  │
    │    - tanh: Output in [-1, 1] - if inputs are normalized to [-1, 1]    │
    │    - None: Unbounded output - for residual learning                   │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

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
        """
        Initialize weights using Kaiming (He) initialization.

        ┌─────────────────────────────────────────────────────────────────────┐
        │                 WHY WEIGHT INITIALIZATION MATTERS                   │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                     │
        │  Bad initialization → gradients explode or vanish → training fails │
        │                                                                     │
        │  KAIMING INITIALIZATION (for ReLU networks):                       │
        │  ─────────────────────────────────────────────                     │
        │  Weights ~ N(0, sqrt(2/fan_out))                                   │
        │                                                                     │
        │  Why sqrt(2/fan_out)?                                              │
        │    - ReLU kills ~50% of activations (sets negative to 0)           │
        │    - The "2" compensates for this halving                          │
        │    - fan_out = number of output connections                        │
        │                                                                     │
        │  This keeps the variance of activations ~1 through the network,    │
        │  preventing exploding/vanishing gradients.                         │
        │                                                                     │
        │  BATCHNORM INITIALIZATION:                                         │
        │  ─────────────────────────                                         │
        │  γ (weight) = 1: Start with identity scaling                       │
        │  β (bias) = 0: Start with no shift                                 │
        │  This means BatchNorm starts as a no-op and learns from there.     │
        │                                                                     │
        └─────────────────────────────────────────────────────────────────────┘
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming init: designed for ReLU activations
                # mode='fan_out' preserves magnitude in backward pass
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # Zero bias is standard
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  # γ = 1 (no scaling initially)
                nn.init.constant_(m.bias, 0)    # β = 0 (no shift initially)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.

        Data flow for depth=4, base_channels=64, input 512×512:

        ENCODER PATH (going down the U):
        ─────────────────────────────────
        Input:      (B, 3, 512, 512)
        InitConv:   (B, 64, 512, 512)   ──► skip[0]
        Down 1:     (B, 128, 256, 256)  ──► skip[1]
        Down 2:     (B, 256, 128, 128)  ──► skip[2]
        Down 3:     (B, 512, 64, 64)    ──► skip[3]
        Bottleneck: (B, 512, 64, 64)

        DECODER PATH (going up the U):
        ─────────────────────────────────
        Up 1:       (B, 256, 128, 128)  ◄── skip[2]
        Up 2:       (B, 128, 256, 256)  ◄── skip[1]
        Up 3:       (B, 64, 512, 512)   ◄── skip[0]
        Output:     (B, 3, 512, 512)

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, C, H, W) - same spatial size as input!
        """
        # Store skip connections - these are the "horizontal arrows" in the U
        skips = []

        # ═══════════════════════════════════════════════════════════════════
        # ENCODER PATH: Going DOWN the left side of the U
        # ═══════════════════════════════════════════════════════════════════

        # Initial convolution (no downsampling, just expand channels)
        # (B, 3, 512, 512) → (B, 64, 512, 512)
        x = self.initial_conv(x)
        skips.append(x)  # Save for later! This has fine details.

        # Encoder blocks: progressively downsample and increase channels
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)  # Save each level's features

        # ═══════════════════════════════════════════════════════════════════
        # BOTTLENECK: The deepest point of the U
        # ═══════════════════════════════════════════════════════════════════
        # This is where we have maximum semantic understanding but minimum
        # spatial resolution. The network "knows" what the image is about
        # but has lost the precise pixel locations.
        x = self.bottleneck(x)

        # ═══════════════════════════════════════════════════════════════════
        # DECODER PATH: Going UP the right side of the U
        # ═══════════════════════════════════════════════════════════════════

        # Prepare skip connections: reverse order, exclude the last one
        # (the last one would be the bottleneck's input, which we don't need)
        skips = skips[:-1][::-1]

        # Decoder blocks: progressively upsample and combine with skips
        for decoder, skip in zip(self.decoders, skips):
            # Each decoder: upsample x, concatenate with skip, convolve
            x = decoder(x, skip)

        # ═══════════════════════════════════════════════════════════════════
        # OUTPUT: Final 1×1 convolution to get desired output channels
        # ═══════════════════════════════════════════════════════════════════
        x = self.final_conv(x)    # (B, 64, 512, 512) → (B, 3, 512, 512)
        x = self.final_activation(x)  # Sigmoid: constrain to [0, 1]

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

"""
PatchGAN Discriminator for adversarial training.

================================================================================
                        GENERATIVE ADVERSARIAL NETWORKS (GANs)
                              The Science & Engineering
================================================================================

GANs were introduced by Ian Goodfellow in 2014 and revolutionized generative AI.
The core idea is beautifully simple: train two networks that compete against
each other.

┌─────────────────────────────────────────────────────────────────────────────┐
│                           THE MINIMAX GAME                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  GENERATOR (G): "The Forger"                                               │
│    - Goal: Create fake images that look real                               │
│    - Tries to FOOL the discriminator                                       │
│    - Gets better by learning what the discriminator catches                │
│                                                                             │
│  DISCRIMINATOR (D): "The Detective"                                        │
│    - Goal: Distinguish real images from fakes                              │
│    - Tries to CATCH the generator's fakes                                  │
│    - Gets better by seeing more examples of both                           │
│                                                                             │
│  They play a minimax game:                                                  │
│                                                                             │
│    min_G max_D  E[log D(real)] + E[log(1 - D(G(z)))]                       │
│                                                                             │
│  In English:                                                                │
│    - D wants to maximize: say "real" for real, "fake" for fake            │
│    - G wants to minimize: make D say "real" for fakes                     │
│                                                                             │
│  At equilibrium (Nash equilibrium):                                        │
│    - G produces perfect fakes                                              │
│    - D outputs 0.5 for everything (can't tell the difference)             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      WHY GANs PRODUCE SHARP IMAGES                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  The L1/L2 Loss Problem:                                                   │
│  ─────────────────────────                                                 │
│  When a model is uncertain, L1/L2 loss encourages the AVERAGE:             │
│                                                                             │
│    If the target could be:   [sharp edge]  or  [sharp edge slightly left] │
│    L1/L2 optimal output:     [blurry edge in the middle]                  │
│                                                                             │
│  The "safe" answer minimizes pixel error but looks terrible!               │
│                                                                             │
│  The GAN Solution:                                                         │
│  ─────────────────                                                         │
│  The discriminator learns to detect blur:                                  │
│                                                                             │
│    D sees: [blurry edge]  →  "FAKE! Real images don't have blur."         │
│                                                                             │
│  So the generator MUST produce sharp outputs:                              │
│                                                                             │
│    G learns: "I have to commit to ONE sharp edge, or D will catch me"     │
│                                                                             │
│  Result: Sharp images because blur is penalized!                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                       CONDITIONAL GANs (cGAN)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Standard GAN:                                                              │
│    G(noise) → image                                                        │
│    D(image) → real/fake                                                    │
│                                                                             │
│  Conditional GAN (what we use):                                            │
│    G(input_image) → output_image                                           │
│    D(input_image, output_image) → real/fake                                │
│                                                                             │
│  Why condition on input?                                                   │
│  ────────────────────────                                                  │
│  We don't just want ANY realistic image - we want a realistic              │
│  TRANSFORMATION of the specific input.                                     │
│                                                                             │
│    Input: [dark living room photo]                                         │
│    Bad output: [beautiful kitchen photo]  ← Wrong room!                   │
│    Good output: [bright living room photo] ← Same room, enhanced!         │
│                                                                             │
│  By concatenating input+output, D learns:                                  │
│    "Is this output a realistic edit of THIS SPECIFIC input?"              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
                              PATCHGAN DISCRIMINATOR
================================================================================

Traditional discriminators output ONE number for the whole image.
PatchGAN outputs a GRID of numbers, one per image patch.

┌─────────────────────────────────────────────────────────────────────────────┐
│                     WHY PATCHGAN WORKS BETTER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. DENSE GRADIENTS                                                        │
│     ─────────────────                                                      │
│     Traditional D: 1 gradient signal for entire image                     │
│     PatchGAN: 1 gradient signal per patch (e.g., 900 for 30x30 grid)      │
│                                                                             │
│     More gradients = faster, more stable training                         │
│                                                                             │
│  2. CAPTURES LOCAL ARTIFACTS                                               │
│     ──────────────────────────                                             │
│     Traditional D might miss small artifacts if global structure is OK    │
│     PatchGAN catches local problems:                                      │
│       - Blurry corners                                                    │
│       - Color bleeding in specific regions                                │
│       - Texture inconsistencies                                           │
│                                                                             │
│  3. FULLY CONVOLUTIONAL                                                    │
│     ───────────────────                                                    │
│     No fully-connected layers = works on any image size                   │
│     Train on 256x256, test on 1024x1024 - no problem!                     │
│                                                                             │
│  4. FEWER PARAMETERS                                                       │
│     ─────────────────                                                      │
│     No FC layers = much smaller model                                     │
│     Faster training, less overfitting                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        RECEPTIVE FIELD                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Each output pixel "sees" a region of the input called its receptive field.│
│                                                                             │
│  For PatchGAN with 3 layers (n_layers=3):                                  │
│                                                                             │
│    Layer 1: 4x4 kernel, stride 2  →  RF = 4                               │
│    Layer 2: 4x4 kernel, stride 2  →  RF = 4 + (4-1)*2 = 10                │
│    Layer 3: 4x4 kernel, stride 2  →  RF = 10 + (4-1)*4 = 22               │
│    Layer 4: 4x4 kernel, stride 1  →  RF = 22 + (4-1)*8 = 46               │
│    Layer 5: 4x4 kernel, stride 1  →  RF = 46 + (4-1)*8 = 70               │
│                                                                             │
│  So each output pixel judges a 70x70 patch of the input!                   │
│                                                                             │
│  Tradeoffs:                                                                │
│    - Larger RF (more layers): Better global coherence, but slower         │
│    - Smaller RF (fewer layers): Better local detail, but may miss global  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
                          GAN TRAINING DYNAMICS
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                      TRAINING STABILITY TRICKS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  GANs are notoriously hard to train. Common problems and solutions:        │
│                                                                             │
│  1. MODE COLLAPSE                                                          │
│     ─────────────────                                                      │
│     Problem: G produces same output for all inputs                         │
│     Solution: Use high L1 weight to anchor outputs to targets             │
│                                                                             │
│  2. DISCRIMINATOR TOO STRONG                                               │
│     ────────────────────────────                                           │
│     Problem: D always outputs 0 for fakes, G gets no gradient             │
│     Solutions:                                                             │
│       - Label smoothing: Use 0.9 instead of 1.0 for "real"               │
│       - Train D less frequently                                           │
│       - Use instance noise (add small noise to D inputs)                  │
│                                                                             │
│  3. DISCRIMINATOR TOO WEAK                                                 │
│     ──────────────────────────                                             │
│     Problem: D always outputs 0.5, G gets no useful signal                │
│     Solutions:                                                             │
│       - Train D more frequently                                           │
│       - Use larger discriminator                                          │
│                                                                             │
│  4. TRAINING OSCILLATION                                                   │
│     ────────────────────────                                               │
│     Problem: G and D keep one-upping each other, never converge           │
│     Solutions:                                                             │
│       - Lower learning rates                                              │
│       - Use Adam with beta1=0.5 (less momentum)                           │
│       - Spectral normalization in D                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    LOSS FUNCTION VARIANTS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. VANILLA GAN (Binary Cross Entropy)                                     │
│     ───────────────────────────────────                                    │
│     D_loss = BCE(D(real), 1) + BCE(D(fake), 0)                            │
│     G_loss = BCE(D(fake), 1)                                              │
│                                                                             │
│     Pros: Simple, well-understood                                         │
│     Cons: Can saturate (vanishing gradients)                              │
│                                                                             │
│  2. LSGAN (Least Squares)                                                  │
│     ───────────────────────                                                │
│     D_loss = MSE(D(real), 1) + MSE(D(fake), 0)                            │
│     G_loss = MSE(D(fake), 1)                                              │
│                                                                             │
│     Pros: More stable, no saturation                                      │
│     Cons: Slightly worse quality                                          │
│                                                                             │
│  3. WGAN (Wasserstein)                                                     │
│     ───────────────────                                                    │
│     D_loss = D(fake) - D(real)  (D is a "critic", not classifier)        │
│     G_loss = -D(fake)                                                     │
│                                                                             │
│     Pros: Very stable, meaningful loss metric                             │
│     Cons: Requires gradient penalty, slower                               │
│                                                                             │
│  4. HINGE LOSS                                                             │
│     ───────────                                                            │
│     D_loss = max(0, 1-D(real)) + max(0, 1+D(fake))                        │
│     G_loss = -D(fake)                                                     │
│                                                                             │
│     Pros: Used in BigGAN, StyleGAN - very stable                         │
│     Cons: Slightly more complex                                           │
│                                                                             │
│  We use vanilla GAN (BCE) with label smoothing - simple and works well.   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
                         ARCHITECTURE CHOICES
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                     INSTANCENORM VS BATCHNORM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BatchNorm: Normalizes across the BATCH dimension                          │
│    mean, var = computed over (N, H, W) for each channel                   │
│    Problem: Statistics depend on what other images are in the batch       │
│                                                                             │
│  InstanceNorm: Normalizes each IMAGE independently                         │
│    mean, var = computed over (H, W) for each (N, C)                       │
│    Better: Each image normalized by its own statistics                    │
│                                                                             │
│  For image-to-image tasks, InstanceNorm is preferred because:             │
│    - Style/color info shouldn't leak between batch samples               │
│    - Works consistently regardless of batch size                          │
│    - Better preserves per-image characteristics                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        LEAKY RELU                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Standard ReLU:  f(x) = max(0, x)                                          │
│                                                                             │
│  Problem in discriminator:                                                 │
│    - Many activations become 0 (especially early in training)             │
│    - Zero activations = zero gradients = "dead neurons"                   │
│    - D can't learn, G can't get useful feedback                           │
│                                                                             │
│  LeakyReLU:  f(x) = max(0.2*x, x)                                          │
│                                                                             │
│                ReLU                LeakyReLU                                │
│              ╭────────╮            ╭────────╮                              │
│              │      ╱ │            │      ╱ │                              │
│              │_____╱  │            │    ╱   │                              │
│              │        │            │__╱     │                              │
│              ╰────────╯            ╰────────╯                              │
│               dead zone            small gradient                          │
│                                                                             │
│  The 0.2 slope for negative values keeps gradients flowing!               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    WEIGHT INITIALIZATION                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  GANs are sensitive to initialization. Standard practice:                  │
│                                                                             │
│    weights ~ N(0, 0.02)    # Small random values                          │
│    biases = 0              # Start unbiased                               │
│                                                                             │
│  Why 0.02?                                                                 │
│    - Large enough to break symmetry                                       │
│    - Small enough to not explode gradients                                │
│    - Empirically found to work well for GANs                              │
│                                                                             │
│  This is different from Kaiming init used in classification networks!     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
"""

import torch
import torch.nn as nn
from .base import BaseModel


class ConvBlock(nn.Module):
    """
    Basic discriminator conv block: Conv -> InstanceNorm -> LeakyReLU

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      CONVBLOCK ARCHITECTURE                             │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Input (B, C_in, H, W)                                                 │
    │         │                                                               │
    │         ▼                                                               │
    │  ┌─────────────────┐                                                   │
    │  │    Conv2d       │  kernel=4, stride=2 → halves spatial dims        │
    │  │  (C_in → C_out) │  kernel=4, stride=1 → maintains spatial dims     │
    │  └────────┬────────┘                                                   │
    │           │                                                             │
    │           ▼                                                             │
    │  ┌─────────────────┐                                                   │
    │  │  InstanceNorm   │  Normalize per-image (optional)                  │
    │  └────────┬────────┘                                                   │
    │           │                                                             │
    │           ▼                                                             │
    │  ┌─────────────────┐                                                   │
    │  │   LeakyReLU     │  slope=0.2 for negative values                   │
    │  └────────┬────────┘                                                   │
    │           │                                                             │
    │           ▼                                                             │
    │  Output (B, C_out, H', W')                                             │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_norm: bool = True,
    ):
        super().__init__()

        layers = [
            # Conv2d with kernel=4, stride=2, padding=1:
            # Output size = (input_size - 4 + 2*1) / 2 + 1 = input_size / 2
            # This halves spatial dimensions each layer
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        ]

        if use_norm:
            # InstanceNorm: Normalize each image independently
            # Better than BatchNorm for style-sensitive tasks
            # Each image keeps its own color/brightness statistics
            layers.append(nn.InstanceNorm2d(out_channels))

        # LeakyReLU: Standard for discriminators
        # - slope=0.2 for negative values prevents dead neurons
        # - inplace=True saves memory (modifies tensor directly)
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PatchDiscriminator(BaseModel):
    """
    PatchGAN Discriminator from the pix2pix paper (Isola et al., 2017).

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      PATCHGAN ARCHITECTURE                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Input: Concatenation of [source_image, output_image]                  │
    │         Shape: (B, 6, H, W) for RGB images                             │
    │                                                                         │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │ Layer 1: Conv(6→64), k=4, s=2, NO norm    512→256              │   │
    │  │   - No norm on first layer (empirically better)                │   │
    │  │   - Stride 2 halves spatial dimensions                         │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              ▼                                          │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │ Layer 2: Conv(64→128), k=4, s=2           256→128              │   │
    │  │   - InstanceNorm + LeakyReLU                                   │   │
    │  │   - Channels double, spatial halves                            │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              ▼                                          │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │ Layer 3: Conv(128→256), k=4, s=2          128→64               │   │
    │  │   - Continue doubling channels                                 │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              ▼                                          │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │ Layer 4: Conv(256→512), k=4, s=1          64→63                │   │
    │  │   - Stride 1: maintain spatial size (almost)                   │   │
    │  │   - Channels capped at 512                                     │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              ▼                                          │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │ Layer 5: Conv(512→1), k=4, s=1            63→62                │   │
    │  │   - Output: 1 channel = real/fake score per patch             │   │
    │  │   - No activation (logits for BCEWithLogitsLoss)              │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                                                                         │
    │  Output: (B, 1, 62, 62) for 512x512 input                              │
    │          Each value = logit(P(this 70x70 patch is real))               │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      GAN TRAINING PROCEDURE                             │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  STEP 1: TRAIN DISCRIMINATOR                                           │
    │  ────────────────────────────────                                       │
    │                                                                         │
    │  # Forward pass on REAL pairs                                          │
    │  real_pred = D(input, target)           # Shape: (B, 1, 62, 62)       │
    │  real_labels = ones_like(real_pred)     # All 1s (or 0.9 for smooth)  │
    │  loss_real = BCE(real_pred, real_labels)                               │
    │                                                                         │
    │  # Forward pass on FAKE pairs                                          │
    │  fake_output = G(input)                 # Generator creates fake      │
    │  fake_pred = D(input, fake_output)      # D judges the fake           │
    │  fake_labels = zeros_like(fake_pred)    # All 0s (or 0.1 for smooth)  │
    │  loss_fake = BCE(fake_pred, fake_labels)                               │
    │                                                                         │
    │  # Update D                                                            │
    │  loss_D = (loss_real + loss_fake) / 2                                  │
    │  loss_D.backward()                                                      │
    │  optimizer_D.step()                                                     │
    │                                                                         │
    │  STEP 2: TRAIN GENERATOR                                               │
    │  ────────────────────────────                                          │
    │                                                                         │
    │  # Generate output                                                     │
    │  fake_output = G(input)                                                │
    │                                                                         │
    │  # Reconstruction loss (keeps output close to target)                  │
    │  loss_recon = L1(fake_output, target) + Perceptual(fake_output, target)│
    │                                                                         │
    │  # Adversarial loss (fool D into saying "real")                        │
    │  fake_pred = D(input, fake_output)                                     │
    │  real_labels = ones_like(fake_pred)     # We WANT D to say "real"     │
    │  loss_adv = BCE(fake_pred, real_labels)                                │
    │                                                                         │
    │  # Update G (note: D is frozen, gradients flow through D to G)        │
    │  loss_G = loss_recon + lambda_adv * loss_adv                           │
    │  loss_G.backward()                                                      │
    │  optimizer_G.step()                                                     │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    MONITORING GAN TRAINING                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Key metrics to watch:                                                 │
    │                                                                         │
    │  D(real): Should be ~0.7-0.9                                           │
    │    - Too high (>0.95): D is too confident, might overfit              │
    │    - Too low (<0.5): D is failing to recognize real images            │
    │                                                                         │
    │  D(fake): Should be ~0.3-0.5                                           │
    │    - Too low (<0.1): D is too strong, G gets no gradient              │
    │    - Too high (>0.7): D is too weak, G gets no useful signal          │
    │                                                                         │
    │  Healthy training:                                                     │
    │    - D(real) ≈ 0.8, D(fake) ≈ 0.4                                     │
    │    - Both values slowly converge toward 0.5                           │
    │    - G loss decreases over time                                       │
    │                                                                         │
    │  Warning signs:                                                        │
    │    - D(real) = 1.0, D(fake) = 0.0: D too strong, try label smoothing  │
    │    - Both stuck at 0.5 early: D too weak, increase D capacity         │
    │    - G loss oscillates wildly: Lower learning rate                    │
    │    - Same output for all inputs: Mode collapse, increase L1 weight    │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        in_channels: Input channels (default 6 = input RGB + output RGB concatenated)
        base_channels: Base number of filters (doubles each layer up to 512)
        n_layers: Number of downsampling layers (controls receptive field size)
    """

    def __init__(
        self,
        in_channels: int = 6,  # input image (3) + output image (3) concatenated
        base_channels: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.n_layers = n_layers

        layers = []

        # ═══════════════════════════════════════════════════════════════════
        # FIRST LAYER: No normalization
        # ═══════════════════════════════════════════════════════════════════
        # Empirically, not normalizing the first layer works better.
        # The input is already in a known range [0, 1], so normalization
        # could actually hurt by removing useful signal.
        layers.append(
            ConvBlock(in_channels, base_channels, stride=2, use_norm=False)
        )

        # ═══════════════════════════════════════════════════════════════════
        # INTERMEDIATE LAYERS: Downsample and increase channels
        # ═══════════════════════════════════════════════════════════════════
        # Each layer:
        #   - Halves spatial dimensions (stride=2)
        #   - Doubles channels (up to 512 max)
        #   - Applies InstanceNorm + LeakyReLU
        channels = base_channels
        for i in range(1, n_layers):
            prev_channels = channels
            channels = min(channels * 2, 512)  # Cap at 512 to limit params
            layers.append(
                ConvBlock(prev_channels, channels, stride=2, use_norm=True)
            )

        # ═══════════════════════════════════════════════════════════════════
        # SECOND-TO-LAST LAYER: Stride 1 (no spatial reduction)
        # ═══════════════════════════════════════════════════════════════════
        # This layer increases receptive field without reducing spatial size.
        # Helps the final layer have more context.
        prev_channels = channels
        channels = min(channels * 2, 512)
        layers.append(
            ConvBlock(prev_channels, channels, stride=1, use_norm=True)
        )

        # ═══════════════════════════════════════════════════════════════════
        # FINAL LAYER: Output real/fake scores
        # ═══════════════════════════════════════════════════════════════════
        # - 1 output channel: each spatial location is a real/fake score
        # - No normalization: we want raw logits
        # - No activation: BCEWithLogitsLoss applies sigmoid internally
        #   (more numerically stable than sigmoid + BCE)
        layers.append(
            nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=1)
        )

        self.model = nn.Sequential(*layers)

        # Initialize weights using GAN-specific initialization
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights with normal distribution.

        GAN-specific initialization:
        - weights ~ N(0, 0.02): Small random values
        - biases = 0: Start unbiased

        This is different from Kaiming/Xavier init used in other networks!
        The 0.02 std was found empirically to work well for GANs.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Normal init with std=0.02 (GAN standard)
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                # InstanceNorm: start as identity transform
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the discriminator.

        This is a CONDITIONAL discriminator - it sees both input and output
        to judge whether the output is a realistic transformation of the input.

        Args:
            x: Input/source image (B, 3, H, W) - the original image
            y: Output/target image (B, 3, H, W) - either G(x) or real target

        Returns:
            Patch predictions (B, 1, H', W')
            Each value is a logit: P(this patch is from real pair)
            Positive = more likely real, Negative = more likely fake
        """
        # Concatenate input and output along channel dimension
        # Shape: (B, 3, H, W) + (B, 3, H, W) → (B, 6, H, W)
        # This lets D learn: "Is y a realistic output for input x?"
        combined = torch.cat([x, y], dim=1)
        return self.model(combined)

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "base_channels": self.base_channels,
            "n_layers": self.n_layers,
        })
        return config


class MultiScaleDiscriminator(BaseModel):
    """
    Multi-scale discriminator - uses multiple PatchGANs at different scales.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    MULTI-SCALE DISCRIMINATOR                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  WHY MULTIPLE SCALES?                                                  │
    │  ─────────────────────                                                 │
    │  A single discriminator might miss artifacts at certain scales:        │
    │                                                                         │
    │    - Fine-scale D (small RF): Catches texture, sharpness issues       │
    │      But misses global color shifts                                   │
    │                                                                         │
    │    - Coarse-scale D (large RF): Catches global structure, color       │
    │      But misses local blur, artifacts                                 │
    │                                                                         │
    │  SOLUTION: Use 2-3 discriminators at different resolutions!            │
    │                                                                         │
    │         Input Image (512×512)                                          │
    │               │                                                         │
    │      ┌────────┼────────┐                                               │
    │      │        │        │                                               │
    │      ▼        ▼        ▼                                               │
    │    512×512  256×256  128×128                                           │
    │      │        │        │                                               │
    │      ▼        ▼        ▼                                               │
    │     D1       D2       D3                                               │
    │   (fine)  (medium)  (coarse)                                           │
    │      │        │        │                                               │
    │      └────────┴────────┘                                               │
    │              │                                                          │
    │              ▼                                                          │
    │    Total Loss = L1 + L2 + L3                                           │
    │                                                                         │
    │  Each D has the same architecture but sees different scale.            │
    │  Together they catch artifacts at all scales!                          │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

    Args:
        in_channels: Input channels per discriminator
        base_channels: Base filters for each discriminator
        n_layers: Layers per discriminator
        num_scales: Number of discriminators (default 2)
    """

    def __init__(
        self,
        in_channels: int = 6,
        base_channels: int = 64,
        n_layers: int = 3,
        num_scales: int = 2,
    ):
        super().__init__()

        self.num_scales = num_scales

        # Create identical discriminators for each scale
        # They share architecture but see different resolutions
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(in_channels, base_channels, n_layers)
            for _ in range(num_scales)
        ])

        # Downsampler for creating multi-scale inputs
        # AvgPool is smooth (no aliasing artifacts)
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass through all discriminators.

        Args:
            x: Input image (B, 3, H, W)
            y: Output image (B, 3, H, W)

        Returns:
            List of predictions from each scale discriminator
            [D1_pred, D2_pred, ...] where D1 is finest scale
        """
        outputs = []

        for i, D in enumerate(self.discriminators):
            # Get prediction at current scale
            outputs.append(D(x, y))

            # Downsample for next scale (except last)
            if i < self.num_scales - 1:
                x = self.downsample(x)
                y = self.downsample(y)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_scales": self.num_scales,
        })
        return config


# ═══════════════════════════════════════════════════════════════════════════════
# DISCRIMINATOR REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

DISCRIMINATOR_REGISTRY = {
    # Standard PatchGAN (70x70 receptive field)
    "patch": PatchDiscriminator,

    # Smaller PatchGAN (34x34 receptive field) - faster, less memory
    "patch_small": lambda **kwargs: PatchDiscriminator(n_layers=2, **kwargs),

    # Larger PatchGAN (142x142 receptive field) - better global coherence
    "patch_large": lambda **kwargs: PatchDiscriminator(n_layers=4, **kwargs),

    # Multi-scale discriminator - best quality, more compute
    "multiscale": MultiScaleDiscriminator,
}


def get_discriminator(name: str, **kwargs) -> BaseModel:
    """
    Get a discriminator by name.

    Args:
        name: Discriminator name
            - "patch": Standard 70x70 RF PatchGAN (recommended)
            - "patch_small": Smaller 34x34 RF (faster)
            - "patch_large": Larger 142x142 RF (better global)
            - "multiscale": Multiple scales (best quality)
        **kwargs: Additional arguments passed to discriminator

    Returns:
        Instantiated discriminator
    """
    if name not in DISCRIMINATOR_REGISTRY:
        raise ValueError(
            f"Unknown discriminator: {name}. "
            f"Available: {list(DISCRIMINATOR_REGISTRY.keys())}"
        )
    return DISCRIMINATOR_REGISTRY[name](**kwargs)

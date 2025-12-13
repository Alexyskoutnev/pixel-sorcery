# U-Net & Encoder-Decoder Architecture Deep Dive

## 1. The Core Idea: Encoder-Decoder

### The Problem
We need to transform an image while:
- Understanding **global context** (what's in the scene, lighting conditions)
- Preserving **local details** (edges, textures, exact pixel positions)

### Basic Encoder-Decoder

```
INPUT (512x512x3)
    │
    ▼
┌─────────────────────────────────────┐
│         ENCODER (Downsampling)       │
│                                      │
│  512x512 → 256x256 → 128x128 → 64x64 → 32x32
│                                      │
│  "Compress" spatial info into        │
│  rich feature representations        │
└─────────────────────────────────────┘
    │
    ▼
   BOTTLENECK (32x32 with many channels)
   "Understands" the whole image
    │
    ▼
┌─────────────────────────────────────┐
│         DECODER (Upsampling)         │
│                                      │
│  32x32 → 64x64 → 128x128 → 256x256 → 512x512
│                                      │
│  "Expand" back to full resolution    │
└─────────────────────────────────────┘
    │
    ▼
OUTPUT (512x512x3)
```

### The Problem with Basic Encoder-Decoder

**Information bottleneck!**

When you compress 512×512×3 = 786,432 values down to 32×32×256 = 262,144 values, you lose fine details. The decoder has to "guess" where edges were.

---

## 2. U-Net: Skip Connections to the Rescue

U-Net was invented for medical image segmentation (2015) but works brilliantly for any image-to-image task.

### The Key Innovation: Skip Connections

```
INPUT ─────────────────────────────────────────────────────► COPY ─────┐
  │                                                                     │
  ▼                                                                     ▼
[Conv Block] 512x512x64 ──────────────────────────────► COPY ───► [Concat + Conv] → OUTPUT
  │                                                                     ▲
  ▼                                                                     │
[Downsample] 256x256x128 ────────────────────────► COPY ───► [Concat + Conv]
  │                                                                     ▲
  ▼                                                                     │
[Downsample] 128x128x256 ──────────────────► COPY ───► [Concat + Conv]
  │                                                                     ▲
  ▼                                                                     │
[Downsample] 64x64x512 ────────────► COPY ───► [Concat + Conv]
  │                                                                     ▲
  ▼                                                                     │
[Downsample] 32x32x512 ─────► [Bottleneck] ─────► [Upsample]
```

### Why Skip Connections Work

| Without Skips | With Skips |
|---------------|------------|
| Decoder must reconstruct edges from compressed features | Decoder gets exact edge locations from encoder |
| Gradients must flow through bottleneck | Gradients have "shortcut" paths |
| Hard to preserve fine details | Fine details preserved perfectly |

**Analogy:** Imagine describing a photo to someone over the phone (encoder), then they draw it (decoder). Now imagine you can also send them a blurry version of the photo (skip connections). Much easier!

---

## 3. The Building Blocks

### Convolutional Block

```python
class ConvBlock(nn.Module):
    """Two convolutions with normalization and activation"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),      # Stabilizes training
            nn.ReLU(inplace=True),        # Non-linearity
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
```

**Why two convolutions?**
- First conv: changes channel dimension
- Second conv: refines features
- More expressive than single conv

### Downsampling (Encoder)

**Option 1: Max Pooling** (classic U-Net)
```python
nn.MaxPool2d(2)  # 256x256 → 128x128
```
- Keeps strongest activations
- Loses exact spatial info

**Option 2: Strided Convolution** (modern preference)
```python
nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
```
- Learnable downsampling
- Often works better

### Upsampling (Decoder)

**Option 1: Bilinear + Conv** (stable)
```python
nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
```

**Option 2: Transposed Convolution** (learnable)
```python
nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
```
- Can cause "checkerboard artifacts" if not careful

**Option 3: Pixel Shuffle** (efficient for super-resolution)
```python
nn.PixelShuffle(upscale_factor=2)  # Rearranges channels into spatial
```

---

## 4. Loss Functions (Critical for Quality!)

### L1 Loss (Mean Absolute Error)
```python
loss = torch.abs(output - target).mean()
```
- Simple, stable
- Produces slightly blurry results
- **Good baseline**

### L2 Loss (Mean Squared Error)
```python
loss = ((output - target) ** 2).mean()
```
- Penalizes large errors more
- Even blurrier than L1
- **Not recommended alone**

### Perceptual Loss (VGG Loss)
```python
# Extract features from pretrained VGG
vgg_output = vgg(output)      # Features of our output
vgg_target = vgg(target)      # Features of ground truth
loss = torch.abs(vgg_output - vgg_target).mean()
```

**Why it works:**
- VGG was trained on ImageNet to recognize objects
- Its intermediate features capture "perceptual" similarity
- Two images can look similar to humans but have high pixel-wise error
- Perceptual loss captures what humans care about

### SSIM Loss (Structural Similarity)
```python
from pytorch_msssim import ssim
loss = 1 - ssim(output, target, data_range=1.0)
```
- Measures structural similarity
- Considers luminance, contrast, structure
- Good complement to L1

### Combined Loss (Best Practice)
```python
loss = l1_loss + 0.1 * perceptual_loss + 0.1 * (1 - ssim)
```

---

## 5. Training Techniques

### Learning Rate Schedule

```python
# Start high, decay over time
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs
)
```

### Data Augmentation

For image-to-image, **apply same augmentation to both input AND target**:

```python
# Random horizontal flip (apply to both!)
if random.random() > 0.5:
    input_img = TF.hflip(input_img)
    target_img = TF.hflip(target_img)

# Random crop (same crop for both!)
i, j, h, w = transforms.RandomCrop.get_params(input_img, (512, 512))
input_img = TF.crop(input_img, i, j, h, w)
target_img = TF.crop(target_img, i, j, h, w)
```

**Safe augmentations for color grading:**
- Horizontal flip ✓
- Random crop ✓
- Rotation (small angles) ✓

**Avoid for color grading:**
- Color jitter ✗ (would corrupt the color mapping!)
- Brightness/contrast changes ✗

### Batch Size Trade-offs

| Small Batch (4-8) | Large Batch (32+) |
|-------------------|-------------------|
| More noise in gradients | Stable gradients |
| Can escape local minima | May get stuck |
| Less memory | More memory |

**For this task:** Start with batch size 8-16 at 512px resolution.

---

## 6. Architecture Variations

### Residual U-Net (Learn the difference)

Instead of: `output = UNet(input)`

Do: `output = input + UNet(input)`

```python
class ResidualUNet(nn.Module):
    def forward(self, x):
        residual = self.unet(x)  # Learn the "edit"
        return x + residual       # Add to original
```

**Why?** For photo enhancement, the output is very similar to input. Learning the small difference is easier than learning the whole image.

### Attention U-Net

Add attention gates to skip connections to learn which parts of skip features are relevant.

### Multi-Scale / Feature Pyramid

Process at multiple resolutions and combine for better results.

---

## 7. Model Size Guidelines

| Resolution | Suggested Base Channels | Approx. Parameters |
|------------|------------------------|-------------------|
| 256px | 32 | ~2M |
| 512px | 64 | ~8M |
| 1024px | 64 | ~8M (use more levels) |

---

## Summary: What Makes a Good U-Net for This Task

1. **Skip connections** - preserve details from input
2. **Residual learning** - learn the edit, not the whole image
3. **Combined loss** - L1 + Perceptual + SSIM
4. **Appropriate depth** - 4-5 encoder levels for 512px
5. **Proper augmentation** - same transform on input AND target

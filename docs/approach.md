# AutoHDR Challenge: Engineering & Science Deep Dive

## Table of Contents
- [Research Summary](#research-summary)
- [SOTA Analysis](#sota-analysis)
- [Recommended Approaches](#recommended-approaches)
- [Approach 1: Efficient Diffusion (DiffIR)](#approach-1-efficient-diffusion-diffir)
- [Approach 2: Fine-tuned InstructPix2Pix](#approach-2-fine-tuned-instructpix2pix)
- [Approach 3: Neural 3D LUT (NILUT)](#approach-3-neural-3d-lut-nilut)
- [Approach 4: Direct Regression (NAFNet)](#approach-4-direct-regression-nafnet)
- [Loss Functions](#loss-functions)
- [Training Strategy](#training-strategy)
- [Deployment & Optimization](#deployment--optimization)
- [Final Recommendation](#final-recommendation)

---

## Research Summary

### Key Finding: Diffusion > Regression for Perceptual Quality

Research consistently shows that **diffusion models outperform regression approaches** for image-to-image tasks:

> "Researchers avoid pixel-level metrics like PSNR and SSIM as they are not reliable measures of sample quality for difficult tasks that require hallucination. PSNR and SSIM tend to prefer blurry regression outputs, unlike human perception."
> — [Palette: Image-to-Image Diffusion Models](https://arxiv.org/abs/2111.05826)

**However**, the challenge is inference cost. Recent work has solved this:

| Method | Inference Steps | Quality | Speed |
|--------|----------------|---------|-------|
| Traditional DDPM | 1000 | Excellent | Very Slow |
| [DiffIR](https://arxiv.org/abs/2303.09472) | 4-8 | Excellent | Fast |
| [ResShift](https://arxiv.org/abs/2403.07319) | 4 | SOTA | Fast |
| NAFNet | 1 | Good | Very Fast |
| [NILUT](https://github.com/mv-lab/nilut) | 1 | Good | Extremely Fast |

---

## SOTA Analysis

### MIT-Adobe 5K Benchmark (Photo Retouching)

Current leaderboard for the standard photo enhancement benchmark:

| Rank | Method | Year | Notes |
|------|--------|------|-------|
| 1 | [MAXIM](https://paperswithcode.com/sota/photo-retouching-on-mit-adobe-5k) | 2022 | Multi-axis MLP architecture |
| 2 | [IAC](https://arxiv.org/html/2501.06448v1) | 2025 | Image-adaptive coordinate system |
| 3 | [eLIR-Net](https://openaccess.thecvf.com/content/WACV2025/papers/Zhao_eLIR-Net_An_Efficient_AI_Solution_for_Image_Retouching_WACV_2025_paper.pdf) | 2025 | Efficient 1×1 conv framework |

### Cutting-Edge Methods (2024-2025)

| Method | Key Innovation | Pros | Cons |
|--------|---------------|------|------|
| [INRetouch](https://arxiv.org/html/2412.03848) | One-shot learning, implicit neural rep | Real-time 4K, learns from single example | Complex architecture |
| [InstructIR](https://arxiv.org/html/2401.16468v3) | Human instruction following | Multi-task, flexible | Slower inference |
| [NILUT](https://mv-lab.github.io/nilut/) | Neural implicit 3D LUT | Extremely fast, mobile-friendly | Limited to color transforms |
| [DiffIR](https://github.com/Zj-BinXia/DiffIR) | Compact prior extraction | 1000× faster than RePaint | Still slower than regression |

---

## Recommended Approaches

Based on hackathon constraints (70% quality / 30% efficiency, 577 pairs, limited time):

### Tier 1: Best Balance (Recommended)

**[DiffIR](https://github.com/Zj-BinXia/DiffIR) or [ResShift](https://arxiv.org/abs/2403.07319)**
- Diffusion quality with only 4-8 steps
- SOTA perceptual quality
- Reasonable inference cost

### Tier 2: Maximum Efficiency

**[NILUT](https://github.com/mv-lab/nilut) or [Image-Adaptive 3D LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT)**
- Extremely fast inference (<10ms)
- Great for the 30% efficiency score
- May not capture complex local edits

### Tier 3: Creative/Hybrid

**Fine-tuned [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix)**
- Can use natural language guidance
- Strong generalization
- Straightforward fine-tuning pipeline

---

## Approach 1: Efficient Diffusion (DiffIR)

### Why DiffIR Works

Traditional diffusion runs 1000 steps to denoise from pure noise. DiffIR's insight:

```
Traditional Diffusion:
  Pure Noise → [1000 steps] → Clean Image

DiffIR:
  LQ Image → CPEN extracts compact prior → [4-8 steps] → Clean Image
```

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DiffIR Pipeline                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Low-Quality Image                                       │
│        ↓                                                │
│  ┌──────────────┐                                       │
│  │    CPEN      │  Compact Prior Extraction Network     │
│  │  (Stage 2)   │  Extracts IR Prior Representation     │
│  └──────────────┘                                       │
│        ↓                                                │
│  ┌──────────────┐     ┌──────────────┐                  │
│  │  Denoising   │ ←── │  Diffusion   │  Only 4-8 steps! │
│  │   Network    │     │   Process    │                  │
│  └──────────────┘     └──────────────┘                  │
│        ↓                                                │
│  ┌──────────────┐                                       │
│  │  DIRformer   │  Dynamic IR Transformer               │
│  └──────────────┘                                       │
│        ↓                                                │
│  Enhanced Image                                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Key Innovation: Compact IR Prior

Instead of diffusing the entire image, DiffIR diffuses a **compact vector** representing the enhancement:

```python
# Traditional: Diffuse full image (H×W×3) - expensive!
# DiffIR: Diffuse compact prior (256-dim vector) - cheap!

# The prior captures "what enhancement is needed"
# The transformer applies it to the image
```

### Performance

> "DiffIR achieves SOTA performance consuming much less computation than other DM-based methods. In particular, DiffIR is **1000× more efficient than RePaint**."

---

## Approach 2: Fine-tuned InstructPix2Pix

### Why Consider This

[InstructPix2Pix](https://www.timothybrooks.com/instruct-pix2pix) was trained on millions of synthetic edit pairs. Fine-tuning on your 577 real estate pairs could work well.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│              InstructPix2Pix Architecture                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input Image ──→ VAE Encoder ──→ Latent (64×64×4)       │
│                                      ↓                   │
│  Text: "enhance" ──→ CLIP ──→ Text Embedding            │
│                                      ↓                   │
│                            ┌─────────────────┐          │
│                            │   Conditional   │          │
│                            │     U-Net       │ ← Fine-tune this │
│                            └─────────────────┘          │
│                                      ↓                   │
│                              VAE Decoder                 │
│                                      ↓                   │
│                              Enhanced Image              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Fine-tuning Strategy

```python
# From HuggingFace blog on instruction-tuning SD
# https://huggingface.co/blog/instruction-tuning-sd

# Freeze VAE and CLIP, only train U-Net
# Use your 577 pairs with instruction like:
instruction = "enhance this real estate photo professionally"

# Training config
learning_rate = 1e-5
batch_size = 4
epochs = 100
```

### Two CFG Parameters

InstructPix2Pix has unique dual conditioning:

| Parameter | Controls | Recommended |
|-----------|----------|-------------|
| Image CFG | How similar to input | 1.5 |
| Text CFG | How much to follow instruction | 7.5 |

For photo enhancement, you want **high Image CFG** (preserve structure) and **moderate Text CFG** (apply enhancement).

---

## Approach 3: Neural 3D LUT (NILUT)

### Why 3D LUTs Are Clever

Real estate photo editing is primarily **color grading** - exactly what 3D LUTs excel at.

> "NILUTs are memory-efficient and can run on mobile devices, therefore easy to integrate into modern deep learning ISPs."
> — [NILUT (AAAI 2024)](https://mv-lab.github.io/nilut/)

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│                   3D LUT Concept                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  A 3D LUT maps every possible color to a new color:     │
│                                                          │
│     (R, G, B) ──→ LUT[R][G][B] ──→ (R', G', B')         │
│                                                          │
│  Traditional: Fixed LUT for all images                   │
│  Neural: Predict image-specific LUT                      │
│                                                          │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐       │
│  │  Input   │ ──→  │   CNN    │ ──→  │ 33³×3    │       │
│  │  Image   │      │ (small)  │      │   LUT    │       │
│  └──────────┘      └──────────┘      └──────────┘       │
│                                            ↓             │
│                                     Trilinear Interp     │
│                                            ↓             │
│                                      Enhanced Image      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### NILUT: Neural Implicit LUT

Instead of predicting a discrete 33×33×33 grid, NILUT uses a neural network AS the LUT:

```python
# Traditional 3D LUT
output_color = LUT[r_idx, g_idx, b_idx]  # Discrete lookup

# NILUT (continuous, differentiable)
output_color = MLP(r, g, b)  # Neural network query
```

### Advantages

| Feature | Benefit |
|---------|---------|
| ~500KB model | Runs on mobile |
| <10ms inference | Great for efficiency score |
| Style blending | Can interpolate between looks |
| Differentiable | End-to-end trainable |

### Limitation

3D LUTs apply **global** color transforms. They can't do:
- Local adjustments (brighten just the window)
- Spatial-aware edits (different treatment for sky vs interior)

**However**, looking at your source/target pairs, most edits ARE global color/tone adjustments, so this might work well.

---

## Approach 4: Direct Regression (NAFNet)

### When to Use

If inference speed is critical and you need single-pass prediction:

```
Input → NAFNet → Output (one forward pass, ~50ms at 1024px)
```

### Architecture

```
Input Image (H×W×3)
       ↓
┌──────────────────────────────────────┐
│           ENCODER                     │
│  Conv → Down → Conv → Down → Conv    │
│  [3]    [64]   [128]  [256]  [512]   │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│           BOTTLENECK                  │
│     SimpleGate + Channel Attention    │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│           DECODER                     │
│  Up → Conv → Up → Conv → Up → Conv   │
│  [256]  [128]  [64]   [3]            │
│         ↑ Skip Connections ↑          │
└──────────────────────────────────────┘
       ↓
Output Image (H×W×3)
```

### The "Nonlinear Activation Free" Insight

```python
# Traditional
x = ReLU(Conv(x))  # Kills negative values

# NAFNet SimpleGate
x1, x2 = split(x)
x = x1 * x2  # Element-wise product, preserves info
```

### Benchmark Results

> "NAFNet achieves 33.69 dB PSNR on GoPro, exceeding the previous SOTA by 0.38 dB with only **8.4% of its computational costs**"

**Caveat**: NAFNet optimizes for PSNR/SSIM, which can produce blurry results. For perceptual quality, diffusion is better.

---

## Loss Functions

### The Critical Choice

| Loss | Optimizes For | Result |
|------|--------------|--------|
| L1/L2 | Pixel accuracy | Sharp but may miss color nuance |
| Perceptual (VGG) | Feature similarity | Better textures, can shift colors |
| SSIM | Structural similarity | Preserves edges and contrast |
| Adversarial (GAN) | Realism | Sharp but can hallucinate |
| LPIPS | Perceptual distance | Best correlation with human judgment |

### Recommended Combination

```python
# For regression models (NAFNet, etc.)
loss = 1.0 * L1_loss +
       0.1 * perceptual_loss +
       0.05 * (1 - SSIM_loss)

# For diffusion models
# Standard diffusion loss (predict noise) +
# Optional perceptual guidance
```

### Perceptual Loss Details

```python
# Extract features from pretrained VGG
vgg_layers = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']

def perceptual_loss(pred, target):
    loss = 0
    for layer in vgg_layers:
        feat_pred = vgg(pred, layer)
        feat_target = vgg(target, layer)
        loss += F.l1_loss(feat_pred, feat_target)
    return loss
```

---

## Training Strategy

### Data Efficiency for 577 Images

| Technique | Purpose |
|-----------|---------|
| Heavy augmentation | Flips, crops (NOT color jitter!) |
| Progressive training | 256px → 512px → 1024px |
| Pretrained weights | ImageNet or larger photo datasets |
| Regularization | Dropout, weight decay |

### Augmentation Rules

```python
# GOOD - preserves edit relationship
transforms.RandomHorizontalFlip()
transforms.RandomCrop(512)
transforms.RandomRotation(degrees=2)

# BAD - destroys what we're trying to learn
transforms.ColorJitter()  # NO! We're learning the color change
transforms.GaussianBlur()  # NO! Destroys detail
```

### Training Schedule

```python
# For diffusion (DiffIR/InstructPix2Pix)
optimizer = AdamW(lr=1e-5, weight_decay=0.01)
scheduler = CosineAnnealingLR(T_max=100)
epochs = 100-200

# For regression (NAFNet)
optimizer = AdamW(lr=1e-3, weight_decay=0.01)
scheduler = CosineAnnealingLR(T_max=200)
epochs = 200-400
```

---

## Deployment & Optimization

### TensorRT Pipeline

```
PyTorch Model
     ↓
Export to ONNX
     ↓
TensorRT optimization:
├── Layer fusion (Conv + BN + ReLU → single kernel)
├── FP16 precision (2x memory bandwidth)
├── Kernel auto-tuning (best CUDA kernels)
     ↓
Optimized Engine (2-5x speedup)
```

### DGX Spark Advantages

| Feature | How to Leverage |
|---------|-----------------|
| 128GB Unified Memory | Load entire dataset + model in memory |
| Grace CPU | Fast data preprocessing |
| TensorRT | Optimized inference |
| CUDA Graphs | Reduce kernel launch overhead |

### Inference Cost Estimation

| Method | Time @ 1024px | VRAM | Steps |
|--------|--------------|------|-------|
| NAFNet | ~50ms | ~2GB | 1 |
| DiffIR | ~200ms | ~4GB | 4 |
| InstructPix2Pix | ~500ms | ~6GB | 20 |
| NILUT | ~5ms | ~200MB | 1 |

---

## Final Recommendation

### For This Hackathon

Given **70% quality / 30% efficiency** scoring:

```
┌─────────────────────────────────────────────────────────┐
│              RECOMMENDED: DiffIR                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  WHY:                                                    │
│  ✓ Diffusion-level perceptual quality                   │
│  ✓ Only 4-8 inference steps (fast enough)               │
│  ✓ Published code available                             │
│  ✓ Can fine-tune on your 577 pairs                      │
│                                                          │
│  BACKUP: NILUT (if time constrained)                    │
│  ✓ Extremely fast inference                             │
│  ✓ Good for global color/tone edits                     │
│  ✓ Simple to implement                                  │
│                                                          │
│  CREATIVE ANGLE: Fine-tuned InstructPix2Pix             │
│  ✓ Novel approach for real estate                       │
│  ✓ Could win "frontier" points                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Implementation Priority

1. **Day 1 AM**: Set up DiffIR, run baseline on your data
2. **Day 1 PM**: Fine-tune on 577 pairs, validate
3. **Day 2 AM**: Export to TensorRT, benchmark
4. **Day 2 PM**: Prepare submission, run on test set

---

## Sources

### Diffusion Models
- [Palette: Image-to-Image Diffusion Models](https://arxiv.org/abs/2111.05826)
- [DiffIR: Efficient Diffusion Model for Image Restoration](https://arxiv.org/abs/2303.09472) | [GitHub](https://github.com/Zj-BinXia/DiffIR)
- [ResShift: Efficient Diffusion by Residual Shifting](https://arxiv.org/abs/2403.07319)
- [Taming Diffusion Models for Image Restoration: A Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12201591/)

### InstructPix2Pix
- [InstructPix2Pix Official](https://www.timothybrooks.com/instruct-pix2pix) | [GitHub](https://github.com/timothybrooks/instruct-pix2pix)
- [HuggingFace: Instruction-tuning SD](https://huggingface.co/blog/instruction-tuning-sd)
- [Fine-Tune InstructPix2Pix Notebook](https://github.com/arnabd64/Fine-Tune-Instruct-Pix2Pix)

### 3D LUT Methods
- [NILUT: Neural Implicit 3D LUTs (AAAI 2024)](https://mv-lab.github.io/nilut/) | [GitHub](https://github.com/mv-lab/nilut)
- [Image-Adaptive 3D LUT](https://arxiv.org/abs/2009.14468) | [GitHub](https://github.com/HuiZeng/Image-Adaptive-3DLUT)

### Photo Retouching SOTA
- [MIT-Adobe 5K Benchmark](https://paperswithcode.com/sota/photo-retouching-on-mit-adobe-5k)
- [INRetouch: Context Aware Implicit Neural Representation](https://arxiv.org/html/2412.03848)
- [InstructIR: High-Quality Image Restoration](https://arxiv.org/html/2401.16468v3)
- [eLIR-Net: Efficient AI for Image Retouching (WACV 2025)](https://openaccess.thecvf.com/content/WACV2025/papers/Zhao_eLIR-Net_An_Efficient_AI_Solution_for_Image_Retouching_WACV_2025_paper.pdf)

### Regression Baselines
- [NAFNet: Simple Baselines for Image Restoration](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670017.pdf)
- [A Comparative Study of NAFNet Baselines](https://arxiv.org/html/2506.19845)
- [Comparative Study of Image Restoration Networks](https://arxiv.org/html/2310.11881v4)

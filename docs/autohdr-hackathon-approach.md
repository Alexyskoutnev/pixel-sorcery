# AutoHDR Real Estate Photo Editing Hackathon - Technical Approach

## Challenge Overview

| Parameter | Value |
|-----------|-------|
| **Dataset** | 577 paired images (input → professionally edited) |
| **Task** | Image-to-image translation for real estate photo enhancement |
| **Scoring** | 70% Image Quality + 30% Inference Efficiency |
| **Resolution** | ~1024px / 1MP acceptable, higher encouraged |

---

## Latest 2025 State-of-the-Art Methods

### Tier 0: HYPIR - TOP RECOMMENDATION

#### HYPIR: Harnessing Diffusion-Yielded Score Priors for Image Restoration
**[SIGGRAPH 2025]** | [GitHub](https://github.com/XPixelGroup/HYPIR) | [Paper](https://arxiv.org/abs/HYPIR)

```
Architecture:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Input LQ Image                                                         │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              Stable Diffusion 2.1 (Frozen Base)                 │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │                    U-Net Backbone                        │   │   │
│  │  │  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐       │   │   │
│  │  │  │Down │──►│Down │──►│ Mid │──►│ Up  │──►│ Up  │       │   │   │
│  │  │  │Block│   │Block│   │Block│   │Block│   │Block│       │   │   │
│  │  │  └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘       │   │   │
│  │  │     │         │         │         │         │           │   │   │
│  │  │  ┌──▼──┐   ┌──▼──┐   ┌──▼──┐   ┌──▼──┐   ┌──▼──┐       │   │   │
│  │  │  │LoRA │   │LoRA │   │LoRA │   │LoRA │   │LoRA │       │   │   │
│  │  │  │r=256│   │r=256│   │r=256│   │r=256│   │r=256│       │   │   │
│  │  │  └─────┘   └─────┘   └─────┘   └─────┘   └─────┘       │   │   │
│  │  │           (Trainable LoRA adapters)                     │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  Score Prior s_θ(x_t, t) ──► Denoising ──► Output HQ Image             │
│                                                                         │
│  Key: model_t=200, coeff_t=200 (low timestep = preserves structure)    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why HYPIR is ideal for this hackathon:**

| Factor | HYPIR Advantage |
|--------|-----------------|
| **Venue** | SIGGRAPH 2025 - premier graphics conference |
| **Base Model** | SD 2.1 (~865M params) - lightweight, fast |
| **Training** | LoRA rank 256 - prevents overfitting on 577 samples |
| **Inference** | Tiled inference for arbitrary resolution |
| **Code Quality** | Clean repo, Gradio demo, Colab notebook |
| **Pretrained** | Weights on HuggingFace - can fine-tune from here |

**Core Innovation - Score Prior Approach:**
```python
# HYPIR uses diffusion score priors differently than standard diffusion
# Instead of full denoising from T=1000, it uses LOW timesteps (t=200)
# This preserves input structure while enhancing details

# Standard diffusion: x_T (noise) → x_0 (image)
# HYPIR: x_input + small noise (t=200) → x_enhanced

# The score function s_θ(x_t, t) guides enhancement, not generation
```

**Training with RealESRGAN Degradation Pipeline:**
```python
# HYPIR's degradation model (from their code)
class RealESRGANDataset:
    def __init__(self, ...):
        # Applies realistic degradations:
        # 1. Blur (Gaussian, motion, etc.)
        # 2. Resize (down + up with various interpolations)
        # 3. Noise (Gaussian, Poisson, JPEG artifacts)
        # 4. JPEG compression
        pass

    def degrade(self, hq_image):
        # First degradation
        x = random_blur(hq_image)
        x = random_resize(x)
        x = add_noise(x)
        x = jpeg_compress(x)

        # Second degradation (more realistic)
        x = random_blur(x)
        x = random_resize(x)
        x = add_noise(x)
        x = jpeg_compress(x)

        return x  # Degraded LQ image
```

**Quick Start with HYPIR:**
```bash
# Clone and setup
git clone https://github.com/XPixelGroup/HYPIR.git
cd HYPIR
conda create -n hypir python=3.10
conda activate hypir
pip install -r requirements.txt

# Download pretrained weights
# From HuggingFace: HYPIR_sd2.pth

# Run inference on your images
LORA_MODULES=(to_k to_q to_v to_out.0 conv conv1 conv2 conv_shortcut conv_out proj_in proj_out ff.net.2 ff.net.0.proj)
IFS=','
LORA_MODULES_STR="${LORA_MODULES[*]}"
unset IFS

python test.py \
  --base_model_type sd2 \
  --base_model_path stabilityai/stable-diffusion-2-1-base \
  --model_t 200 \
  --coeff_t 200 \
  --lora_rank 256 \
  --lora_modules $LORA_MODULES_STR \
  --weight_path path/to/HYPIR_sd2.pth \
  --patch_size 512 \
  --stride 256 \
  --lq_dir ./your_input_images \
  --scale_by factor \
  --upscale 1 \  # Set to 1 for enhancement only, no upscaling
  --output_dir ./results \
  --seed 231 \
  --device cuda
```

**Fine-tuning HYPIR for Real Estate (Recommended Approach):**
```python
# 1. Start from pretrained HYPIR weights
# 2. Fine-tune LoRA on your 577 real estate pairs
# 3. Use their degradation pipeline OR create custom one

# Custom degradation for real estate photos:
class RealEstateDegradation:
    """
    Real estate photos typically have:
    - Poor lighting / underexposure
    - Color cast (yellow from indoor lighting)
    - Low dynamic range
    - Slight blur from handheld cameras
    """
    def __call__(self, hq_image):
        # Simulate common real estate photo issues
        x = reduce_dynamic_range(hq_image)
        x = add_color_cast(x, yellow_strength=0.2)
        x = reduce_exposure(x, factor=0.7)
        x = add_slight_blur(x, sigma=0.5)
        return x
```

**Training Configuration for 577 Samples:**
```yaml
# configs/sd2_real_estate.yaml
output_dir: ./checkpoints/real_estate

model_config:
  base_model: stabilityai/stable-diffusion-2-1-base
  lora_rank: 256  # High rank for quality
  model_t: 200    # Low timestep preserves structure
  coeff_t: 200

data_config:
  train:
    batch_size: 4  # Adjust based on VRAM
    dataset:
      target: HYPIR.dataset.paired.PairedDataset
      params:
        input_dir: ./data/input
        target_dir: ./data/target
        size: 512

training:
  epochs: 100
  learning_rate: 1e-4
  warmup_steps: 500
```

**Expected Performance with HYPIR:**

| Metric | Value | Notes |
|--------|-------|-------|
| Training time | 2-4 hours | Fine-tuning from pretrained |
| Inference (512px) | ~1-2s | Single image |
| Inference (1024px) | ~3-5s | With tiling |
| VRAM | 8-12 GB | Depends on patch size |
| Quality | Very High | SIGGRAPH 2025 level |

---

### Tier 1: One-Step Diffusion (Best for Efficiency)

#### InstaRevive: One-Step Image Enhancement via Dynamic Score Matching
**[ICLR 2025]**

```
Architecture:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Input X ──► Encoder ──► Dynamic Score Network ──► Output Y │
│                              │                              │
│                    Single forward pass                      │
│                    (no iterative sampling)                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Innovation:**
- Replaces iterative diffusion sampling with single-step score matching
- Learns dynamic score functions conditioned on degradation level
- **1 step inference** - extremely efficient

**Why it fits this challenge:**
- Real estate editing is deterministic (one correct output per input)
- Single-step = lowest possible inference cost
- Score matching is robust to limited training data

---

#### Compression-Aware One-Step Diffusion Model
**[ICCV 2025]**

- Distills multi-step diffusion into single forward pass
- Compression-aware design handles JPEG artifacts common in real estate photos
- Adversarial training for perceptual quality

---

#### SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training
**[2025]**

```
Training Pipeline:
┌────────────────────────────────────────────────────────────┐
│  Stage 1: Train multi-step diffusion model                 │
│           ↓                                                │
│  Stage 2: Adversarial distillation to one-step            │
│           ↓                                                │
│  Result: Single-step model with diffusion quality          │
└────────────────────────────────────────────────────────────┘
```

**Adversarial Post-Training:**
```python
# Distillation objective
L_distill = ||one_step_model(x) - multi_step_model(x, steps=50)||²

# Adversarial objective
L_adv = -log(D(one_step_model(x)))

# Combined
L_total = L_distill + λ * L_adv
```

---

### Tier 2: Flow-Based Methods (Best Quality/Speed Balance)

#### Reversing Flow for Image Restoration
**[CVPR 2025]**

```
Flow Matching Framework:
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  Degraded X ─────────────────────────────────► Clean Y       │
│      │                                            ▲          │
│      │         Learned Flow Field v_θ(x,t)        │          │
│      └────────────────────────────────────────────┘          │
│                                                              │
│  ODE: dx/dt = v_θ(x, t)                                      │
│  Solve from t=0 to t=1 in 4-8 steps                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Why Flow Matching > Diffusion for this task:**
1. **Straight trajectories** - Optimal transport from X to Y
2. **Fewer steps** - 4-8 steps vs 20-50 for diffusion
3. **Deterministic** - No stochastic sampling variance
4. **Paired data friendly** - Directly learns X→Y mapping

**Implementation:**
```python
# Training
def flow_matching_loss(model, x_input, x_target):
    t = torch.rand(batch_size)
    # Interpolate between input and target
    x_t = (1 - t) * x_input + t * x_target
    # Add small noise for regularization
    x_t = x_t + sigma * torch.randn_like(x_t)

    # Model predicts velocity field
    v_pred = model(x_t, t, condition=x_input)
    v_target = x_target - x_input

    return F.mse_loss(v_pred, v_target)

# Inference (4 steps)
def inference(model, x_input, steps=4):
    x = x_input
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.tensor([i * dt])
        v = model(x, t, condition=x_input)
        x = x + v * dt
    return x
```

---

#### FlowIE: Efficient Image Enhancement via Rectified Flow
**[CVPR 2024 Oral]**

- Rectified flow with learned optimal transport
- Specifically designed for image enhancement
- 4-step inference with high quality

---

### Tier 3: Diffusion with Efficient Inference

#### Dual Prompting Image Restoration with Diffusion Transformers
**[CVPR 2025]**

```
Architecture:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌──────────────┐     ┌──────────────────────────┐         │
│  │ Visual Prompt│     │ Semantic Prompt          │         │
│  │ (Input Image)│     │ ("real estate, HDR,      │         │
│  └──────┬───────┘     │  professional editing")  │         │
│         │             └──────────┬───────────────┘         │
│         │                        │                          │
│         └────────┬───────────────┘                          │
│                  ▼                                          │
│         ┌────────────────┐                                  │
│         │ DiT (Diffusion │                                  │
│         │ Transformer)   │                                  │
│         └────────┬───────┘                                  │
│                  ▼                                          │
│            Output Image                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**
- Diffusion Transformer (DiT) backbone - better scaling than U-Net
- Dual conditioning: visual (input image) + semantic (text prompt)
- Can guide style with text: "professional real estate photo, HDR, warm tones"

---

#### Visual-Instructed Degradation Diffusion for All-in-One Image Restoration
**[CVPR 2025]**

- Learns degradation-aware representations
- Single model handles multiple restoration tasks
- Instruction-following capability

---

#### Acquire and then Adapt: Squeezing out Text-to-Image Model for Image Restoration
**[CVPR 2025]**

```
Two-Stage Approach:
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Acquire                                           │
│  ├── Extract restoration priors from pretrained T2I model  │
│  └── Freeze T2I weights                                     │
│                                                             │
│  Stage 2: Adapt                                             │
│  ├── Train lightweight adapter layers                       │
│  ├── LoRA fine-tuning (4-64 rank)                          │
│  └── Task-specific head for restoration                    │
└─────────────────────────────────────────────────────────────┘
```

**Why this approach is ideal for 577 samples:**
- Leverages billions of images from T2I pretraining
- Only trains adapter layers (prevents overfitting)
- LoRA requires minimal compute and data

---

### Tier 4: Specialized Enhancement Models

#### GenDR: Lightning Generative Detail Restorator
**[2025]**

- "Lightning" fast inference
- Generative approach for detail enhancement
- Good for real estate detail preservation (textures, edges)

---

#### LucidFlux: Caption-Free Universal Image Restoration via Large-Scale Diffusion Transformer
**[2025]**

- No text captions needed (pure image-to-image)
- Large-scale DiT architecture
- Universal restoration capabilities

---

## Recommended Architecture for Hackathon

### Primary Recommendation: Residual Flow Matching with Pretrained Encoder

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Input Image X                                                  │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────┐                                           │
│  │ Pretrained VAE  │  (Frozen - from SDXL)                     │
│  │ Encoder         │                                           │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│       Latent z_x ──────────────────┐                           │
│           │                        │                           │
│           ▼                        │ (Skip connection)         │
│  ┌─────────────────┐               │                           │
│  │ Flow Network    │               │                           │
│  │ (Trainable)     │               │                           │
│  │                 │               │                           │
│  │ DiT or U-Net    │◄──────────────┘                           │
│  │ backbone        │                                           │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│     Δz (residual)                                              │
│           │                                                     │
│           ▼                                                     │
│     z_y = z_x + Δz                                             │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ Pretrained VAE  │  (Frozen)                                 │
│  │ Decoder         │                                           │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│     Output Image Y                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Architecture

| Design Choice | Reasoning |
|--------------|-----------|
| **Pretrained VAE** | Leverage SDXL's learned image compression (trained on billions of images) |
| **Residual learning** | Edit X→Y is small; learning Δ is easier than learning Y from scratch |
| **Flow matching** | Straight trajectories, 4-step inference, deterministic outputs |
| **Latent space** | 8x compression reduces compute; 1024px → 128x128 latent |
| **Skip connection** | Preserves input structure; model only learns the enhancement |

---

## Implementation Plan

### Phase 1: Data Preparation (1-2 hours)

```python
# Dataset structure
class RealEstateDataset(Dataset):
    def __init__(self, input_dir, target_dir, size=512):
        self.inputs = sorted(glob(f"{input_dir}/*.jpg"))
        self.targets = sorted(glob(f"{target_dir}/*.jpg"))
        self.size = size
        self.transform = A.Compose([
            A.RandomCrop(size, size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, p=0.3),  # Only on input
            A.Normalize(mean=[0.5]*3, std=[0.5]*3),
            ToTensorV2()
        ])

    def __getitem__(self, idx):
        input_img = cv2.imread(self.inputs[idx])
        target_img = cv2.imread(self.targets[idx])

        # Apply same spatial transform to both
        transformed = self.transform(image=input_img, target=target_img)
        return transformed['image'], transformed['target']
```

### Phase 2: Model Setup (2-3 hours)

```python
import torch
from diffusers import AutoencoderKL

class ResidualFlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Frozen pretrained VAE
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sdxl-vae"
        ).requires_grad_(False)

        # Trainable flow network (U-Net or DiT)
        self.flow_net = UNet2DConditionModel(
            in_channels=8,  # z_x concat with z_noisy
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            attention_head_dim=8,
        )

    def encode(self, x):
        with torch.no_grad():
            return self.vae.encode(x).latent_dist.sample() * 0.18215

    def decode(self, z):
        with torch.no_grad():
            return self.vae.decode(z / 0.18215).sample

    def forward(self, z_t, t, z_condition):
        # Concatenate noisy latent with condition
        z_input = torch.cat([z_t, z_condition], dim=1)
        # Predict velocity/residual
        return self.flow_net(z_input, t).sample
```

### Phase 3: Training (4-8 hours)

```python
def train_flow_matching(model, dataloader, epochs=100):
    optimizer = torch.optim.AdamW(model.flow_net.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for x_input, x_target in dataloader:
            # Encode to latent space
            z_input = model.encode(x_input)
            z_target = model.encode(x_target)

            # Sample timestep
            t = torch.rand(x_input.shape[0], device=x_input.device)

            # Interpolate (flow matching)
            z_t = (1 - t.view(-1,1,1,1)) * z_input + t.view(-1,1,1,1) * z_target

            # Add small noise for regularization
            z_t = z_t + 0.01 * torch.randn_like(z_t)

            # Predict velocity
            v_pred = model(z_t, t, z_input)
            v_target = z_target - z_input

            # Loss
            loss = F.mse_loss(v_pred, v_target)

            # Optional: Add perceptual loss
            if epoch > 50:
                x_pred = model.decode(z_input + v_pred)
                loss += 0.1 * lpips_loss(x_pred, x_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Phase 4: Inference

```python
@torch.no_grad()
def inference(model, x_input, steps=4):
    z = model.encode(x_input)
    z_condition = z.clone()

    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((z.shape[0],), i * dt, device=z.device)
        v = model(z, t, z_condition)
        z = z + v * dt

    return model.decode(z)
```

---

## Loss Function Design

```python
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips = lpips.LPIPS(net='vgg')
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred, target, pred_latent=None, target_latent=None):
        # Pixel-space losses (decoded)
        l1_loss = self.l1(pred, target)
        lpips_loss = self.lpips(pred, target).mean()

        # Latent-space loss (if provided)
        latent_loss = 0
        if pred_latent is not None:
            latent_loss = self.mse(pred_latent, target_latent)

        # Color histogram loss (important for HDR editing)
        color_loss = histogram_loss(pred, target)

        return {
            'total': l1_loss + 0.5 * lpips_loss + 0.1 * latent_loss + 0.1 * color_loss,
            'l1': l1_loss,
            'lpips': lpips_loss,
            'latent': latent_loss,
            'color': color_loss
        }

def histogram_loss(pred, target, bins=64):
    """Encourage matching color distributions"""
    loss = 0
    for c in range(3):
        pred_hist = torch.histc(pred[:,c], bins=bins, min=-1, max=1)
        target_hist = torch.histc(target[:,c], bins=bins, min=-1, max=1)
        pred_hist = pred_hist / pred_hist.sum()
        target_hist = target_hist / target_hist.sum()
        loss += F.l1_loss(pred_hist, target_hist)
    return loss / 3
```

---

## Efficiency Optimizations

### 1. Mixed Precision Training
```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    v_pred = model(z_t, t, z_input)
    loss = F.mse_loss(v_pred, v_target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. Gradient Checkpointing
```python
model.flow_net.enable_gradient_checkpointing()
```

### 3. Compile for Inference (PyTorch 2.0+)
```python
model = torch.compile(model, mode="reduce-overhead")
```

### 4. TensorRT Deployment
```python
# Export to ONNX, then TensorRT for fastest inference
torch.onnx.export(model, dummy_input, "model.onnx")
# trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
```

---

## Expected Performance

| Metric | Target | Notes |
|--------|--------|-------|
| **Training time** | 4-8 hours | Single A100/H100 GPU |
| **Inference time** | 200-500ms | 1024px, 4 steps, FP16 |
| **VRAM (inference)** | 6-10 GB | Depends on batch size |
| **LPIPS** | < 0.15 | Lower is better |
| **PSNR** | > 25 dB | Higher is better |

---

## Alternative Approaches (Backup Plans)

### Backup 1: Pure LoRA Fine-tuning (Fastest to implement)

```bash
# Using diffusers library
accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --instance_data_dir="./data/input" \
  --output_dir="./lora_weights" \
  --instance_prompt="professional real estate photo" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --max_train_steps=2000 \
  --rank=32
```

### Backup 2: Pix2Pix HD (Simplest, fastest inference)

```python
# GAN-based, single forward pass
# Lower quality but 10-50ms inference
class Pix2PixHD(nn.Module):
    def __init__(self):
        self.encoder = ResNetEncoder()
        self.decoder = ResNetDecoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))
```

### Backup 3: NAFNet / Restormer (Non-diffusion SOTA)

- Transformer-based image restoration
- Single forward pass (~50-100ms)
- Good quality, very fast inference
- Less creative/generative capability

---

## Key Papers to Reference (2025)

| Paper | Venue | Key Contribution | Code |
|-------|-------|-----------------|------|
| **HYPIR** | **SIGGRAPH 2025** | **Score priors + LoRA for restoration** | **[GitHub](https://github.com/XPixelGroup/HYPIR)** |
| InstaRevive | ICLR 2025 | One-step enhancement via dynamic score matching | - |
| Reversing Flow | CVPR 2025 | Flow matching for restoration | - |
| Dual Prompting | CVPR 2025 | DiT with visual+text conditioning | - |
| Acquire and Adapt | CVPR 2025 | Efficient T2I adaptation for restoration | - |
| Visual-Instructed Degradation | CVPR 2025 | All-in-one restoration | - |
| Compression-Aware One-Step | ICCV 2025 | One-step distilled diffusion | - |

---

## Comparison: HYPIR vs Other Approaches

| Criteria | HYPIR | Flow Matching | One-Step Diffusion | Pix2Pix |
|----------|-------|---------------|-------------------|---------|
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Inference Speed** | ⭐⭐⭐ (1-5s) | ⭐⭐⭐⭐ (0.2-0.5s) | ⭐⭐⭐⭐⭐ (50ms) | ⭐⭐⭐⭐⭐ (30ms) |
| **Training Data Efficiency** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **Code Availability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Pretrained Weights** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **High-Res Support** | ✅ Tiled | ✅ Possible | ⚠️ Limited | ✅ Yes |
| **Hackathon Readiness** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

### Recommendation Matrix

| Your Priority | Best Choice | Reason |
|--------------|-------------|--------|
| **Maximum Quality** | HYPIR | SIGGRAPH 2025, pretrained, proven results |
| **Balance Quality + Speed** | HYPIR + fewer steps | Reduce model_t from 200 to 100 |
| **Maximum Speed** | One-Step or Pix2Pix | Single forward pass |
| **Minimum Engineering** | HYPIR | Just clone, fine-tune, done |
| **Research Novelty** | Flow Matching | Newer paradigm, less explored |

---

## Submission Checklist

- [ ] Model trained on 577 pairs
- [ ] Inference time measured (target: <1s at 1024px)
- [ ] Sample outputs generated
- [ ] ZIP file prepared with outputs
- [ ] Technical writeup completed:
  - [ ] Model architecture
  - [ ] Why this approach
  - [ ] Training time
  - [ ] Inference time
  - [ ] VRAM usage
  - [ ] Notable optimizations

---

## Quick Start Commands

```bash
# Clone necessary repos
git clone https://github.com/huggingface/diffusers
pip install diffusers transformers accelerate

# Download dataset
aws s3 cp s3://hackathon-dec12/autohdr-real-estate-577/ ./data/ --recursive

# Start training
python train.py \
  --data_dir ./data \
  --output_dir ./checkpoints \
  --resolution 512 \
  --batch_size 4 \
  --epochs 100 \
  --learning_rate 1e-4

# Run inference
python inference.py \
  --checkpoint ./checkpoints/best.pt \
  --input_dir ./test_images \
  --output_dir ./outputs \
  --resolution 1024
```

# Pixel Sorcery

> AI-powered image enhancement using GANs and UNet models for HDR-style processing

Transform dark, underexposed images into vibrant, well-lit photos using deep learning.

## Features

- **GAN-based Enhancement**: Adversarial training for photorealistic results
- **Fast Training**: 512px resolution trains 4-8x faster than 1024px
- **Mobile Deployment**: Export to ONNX with INT8 quantization (75% size reduction)
- **Production Ready**: Complete inference, benchmarking, and testing tools

## Quick Start

### Installation

```bash
git clone git@github.com:Alexyskoutnev/pixel-sorcery.git
cd pixel-sorcery
uv sync
```

### Dataset Structure

```
autohdr-real-estate-577/images/
├── input/      # Dark images
└── target/     # Enhanced targets
```

### Training

**Recommended: Aggressive GAN** (best color quality)

```bash
nohup uv run python train.py \
    --data_dir autohdr-real-estate-577/images \
    --loss gan --model unet --image_size 512 --batch_size 8 \
    --base_channels 32 --depth 4 --epochs 100 --lr 1e-4 \
    --l1_weight 1.0 --perceptual_weight 20.0 --ssim_weight 0.3 --adv_weight 3.5 \
    --discriminator patch_small --save_every 5 --num_workers 4 \
    > training_gan_512_aggressive.log 2>&1 &
```

**Why these weights?**
- Low L1 (1.0) prevents color desaturation
- High Perceptual (20.0) ensures natural colors
- High Adversarial (3.5) pushes photorealism

### Inference

```bash
# Single image
uv run python inference.py \
    --checkpoint checkpoints/gan_unet_512px_bs8_<timestamp>/best_model.pt \
    --input test_image.jpg \
    --output enhanced.jpg

# Batch processing
uv run python inference.py \
    --checkpoint checkpoints/gan_unet_512px_bs8_<timestamp>/best_model.pt \
    --input_dir test_images/ \
    --output_dir enhanced_images/
```

### Export to ONNX

```bash
# Export model
uv run python scripts/export_onnx.py \
    -c checkpoints/gan_unet_512px_bs8_<timestamp>/best_model.pt \
    -o bin/model_gan_512.onnx -s 512

# Quantize to INT8 (12MB → 3.1MB)
uv run python scripts/quantize_onnx.py \
    --model bin/model_gan_512.onnx \
    --output bin/model_gan_512_int8.onnx \
    --method dynamic

# Test ONNX model
uv run python scripts/test_onnx.py \
    --model bin/model_gan_512_int8.onnx \
    --input test_image.jpg \
    --output test_output.jpg

# Benchmark speed
uv run python scripts/benchmark_onnx.py \
    --model bin/model_gan_512_int8.onnx --runs 20
```

## How It Works

### Architecture

**Generator (UNet)**: Encoder-decoder with skip connections
- Downsamples to capture context
- Upsamples to original resolution
- Skip connections preserve details

**Discriminator (PatchGAN)**: Classifies image patches as real/fake
- Guides generator to produce realistic textures
- Prevents washed-out colors

### Loss Functions

| Loss | Weight | Purpose |
|------|--------|---------|
| **L1** | 1.0 (low) | Pixel accuracy (high values wash out colors) |
| **Perceptual** | 20.0 (high) | Natural appearance via VGG features |
| **SSIM** | 0.3 | Structural similarity |
| **Adversarial** | 3.5 (high) | Photorealism from discriminator |

**Golden Rule**: Minimize L1, maximize Perceptual + Adversarial for vibrant colors.

## Project Structure

```
pixel-sorcery/
├── train.py                     # Main training script
├── inference.py                 # Batch inference
├── scripts/
│   ├── export_onnx.py           # PyTorch → ONNX export
│   ├── quantize_pytorch.py      # PyTorch INT8 quantization
│   ├── quantize_onnx.py         # ONNX INT8 quantization
│   ├── benchmark_onnx.py        # Speed benchmarking
│   ├── test_onnx.py             # Test ONNX on images
│   ├── info_onnx.py             # Model metadata
│   └── compare_pytorch_onnx.py  # Verify PyTorch ↔ ONNX
├── src/
│   ├── models/                  # UNet + discriminator
│   ├── training/                # GAN trainer + losses
│   └── utils/                   # Data loading
└── checkpoints/                 # Training outputs
```

## Model Configurations

| Configuration | Resolution | Batch Size | Speed | Size (ONNX) | Use Case |
|--------------|-----------|------------|-------|-------------|----------|
| **Aggressive GAN** | 512px | 8 | ~16-18s/epoch | 12MB → 3MB | Production (best color) |
| **Basic UNet** | 512px | 12 | ~10-12s/epoch | 12MB → 3MB | Fast baseline |
| **High-Res GAN** | 1024px | 4 | ~3 min/epoch | 12MB → 3MB | Maximum quality |

### Alternative Architectures

```bash
# Small UNet (faster, lower memory)
--model unet_small --base_channels 16 --depth 3

# Standard UNet (recommended)
--model unet --base_channels 32 --depth 4

# Large UNet (maximum capacity)
--model unet_large --base_channels 64 --depth 5
```

## Monitoring

```bash
# View training log
tail -f training_gan_512_aggressive.log

# Check GPU
watch -n 1 nvidia-smi

# List checkpoints
ls -lh checkpoints/gan_unet_512px_bs8_*/
```

## Common Issues

| Issue | Solution |
|-------|----------|
| Washed out colors | Decrease `--l1_weight` to 1.0 |
| Unrealistic textures | Increase `--perceptual_weight` to 15-20 |
| Out of memory | Decrease `--batch_size` or `--base_channels` |
| Slow training | Use `--image_size 512` instead of 1024 |

## Performance

**Training** (512px, bs=8, RTX 3080):
- GAN: ~16-18s/epoch (~30 min for 100 epochs)

**Inference** (512px image):
- PyTorch: 30-50ms
- ONNX INT8: 15-25ms (mobile CPU)

**Model Size**:
- PyTorch: ~12MB
- ONNX: ~12MB
- ONNX INT8: ~3.1MB

## Requirements

- Python 3.8+
- CUDA GPU (6GB+ VRAM recommended)
- UV package manager

## License

MIT License

## Acknowledgments

- UNet: [Ronneberger et al.](https://arxiv.org/abs/1505.04597)
- PatchGAN: [Isola et al.](https://arxiv.org/abs/1611.07004)
- Perceptual Loss: [Johnson et al.](https://arxiv.org/abs/1603.08155)

---

**Questions?** Open an issue or check `notes.txt` for detailed examples.

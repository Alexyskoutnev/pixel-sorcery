# Pixel Sorcery

AI-powered image enhancement using GANs and UNet models for HDR-style image processing.

## Overview

This project trains deep learning models to enhance images with HDR-style processing. Supports both standard UNet and GAN-based training for higher quality results. Models can be exported to ONNX for mobile deployment.

## Quick Start

### Training

Train a GAN model for high-quality results:

```bash
# Fast training at 512px (4-8x faster)
nohup uv run python train.py \
    --data_dir autohdr-real-estate-577/images \
    --loss gan \
    --model unet \
    --image_size 512 \
    --batch_size 8 \
    --base_channels 32 \
    --depth 4 \
    --epochs 100 \
    --lr 1e-4 \
    --l1_weight 10.0 \
    --perceptual_weight 5.0 \
    --ssim_weight 1.0 \
    --adv_weight 1.0 \
    --discriminator patch_small \
    --save_every 5 \
    --num_workers 4 > training_gan_512.log 2>&1 &
```

### Inference (Testing Your Model)

Test your trained GAN model on images:

```bash
# Single image
uv run python inference.py \
    --checkpoint checkpoints/gan_unet_512px_bs8_<timestamp>/best_model.pt \
    --input test_image.jpg \
    --output enhanced_output.jpg

# Batch process a directory
uv run python inference.py \
    --checkpoint checkpoints/gan_unet_512px_bs8_<timestamp>/best_model.pt \
    --input_dir test_images/ \
    --output_dir enhanced_images/

# Interactive visualization (shows before/after)
uv run python scripts/inference.py \
    --image test_image.jpg \
    --checkpoint checkpoints/gan_unet_512px_bs8_<timestamp>/best_model.pt \
    --output enhanced.jpg \
    --size 512
```

### Export to ONNX

Export your trained model for mobile deployment:

```bash
# Export GAN model (automatically exports only the generator)
uv run python scripts/export_onnx.py \
    -c checkpoints/gan_unet_512px_bs8_<timestamp>/best_model.pt \
    -o bin/model_gan_512.onnx \
    -s 512
```

### Test ONNX Model

Benchmark and test your ONNX model using dedicated scripts:

```bash
# Check model info (size, opset, inputs/outputs)
uv run python scripts/info_onnx.py --model bin/model_gan_512.onnx

# Benchmark inference speed
uv run python scripts/benchmark_onnx.py --model bin/model_gan_512.onnx --runs 20

# Test on real image
uv run python scripts/test_onnx.py \
    --model bin/model_gan_512.onnx \
    --input test_image.jpg \
    --output test_onnx_output.jpg

# Test on batch of images
uv run python scripts/test_onnx.py \
    --model bin/model_gan_512.onnx \
    --input_dir test_images/ \
    --output_dir onnx_outputs/

# Compare PyTorch vs ONNX (verify correctness)
uv run python scripts/compare_pytorch_onnx.py \
    --checkpoint checkpoints/gan_unet_512px_bs8_<timestamp>/best_model.pt \
    --onnx bin/model_gan_512.onnx \
    --image test_image.jpg
```

## Key Features

- **GAN Training**: Higher quality results with adversarial training
- **Fast Training**: 512px models train 4-8x faster than 1024px
- **Mobile-Ready**: Export to ONNX for mobile deployment
- **Flexible**: Supports multiple model sizes and configurations

## Model Configurations

### Recommended: GAN 512px (Fast Training)
- **Training Speed**: ~6-12s per batch
- **Model Size**: ~15-20MB (ONNX)
- **Inference**: ~150-250ms on mobile
- **Best for**: Quick iterations, testing, production

### High Quality: GAN 1024px
- **Training Speed**: ~50s per batch
- **Model Size**: ~15-20MB (ONNX)
- **Inference**: ~150-250ms on mobile
- **Best for**: Maximum quality, final models

## Project Structure

```
pixel-sorcery/
├── train.py                     # Main training script
├── inference.py                 # Batch inference (PyTorch)
├── scripts/
│   ├── inference.py             # Interactive inference with visualization
│   ├── export_onnx.py           # Export PyTorch models to ONNX
│   ├── benchmark_onnx.py        # Benchmark ONNX inference speed
│   ├── test_onnx.py             # Test ONNX models on images
│   ├── info_onnx.py             # Display ONNX model information
│   └── compare_pytorch_onnx.py  # Compare PyTorch vs ONNX outputs
├── src/
│   ├── models/                  # Model architectures (UNet, discriminators)
│   ├── training/                # Training logic (GAN trainer, loss functions)
│   └── utils/                   # Data loading, logging
├── checkpoints/                 # Saved models (created during training)
└── bin/                         # Exported ONNX models

```

## Notes

- The GAN model exports **only the generator** to ONNX (discriminator is discarded)
- Use 512px for fast training, then optionally fine-tune at 1024px
- All inference scripts automatically detect GAN vs regular checkpoints
- See `notes.txt` for detailed configuration options

## Monitoring Training

```bash
# View training log
tail -f training_gan_512.log

# Check GPU usage
watch -n 1 nvidia-smi

# List checkpoints
ls -lh checkpoints/<run_name>/
```

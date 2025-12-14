# Model Quantization Guide

Quantization reduces model size and speeds up inference by converting FP32 weights to INT8.

## Quick Start - Quantize Your Model

### Option 1: PyTorch Dynamic Quantization (Easiest)

```bash
uv run python scripts/quantize_pytorch.py \
    --checkpoint /mnt/c/Users/alexy/Documents/Projects/pixel-sorcery/checkpoints/gan_unet_512px_bs8_20251214_082125/best_model.pt \
    --output checkpoints/best_model_quantized.pt \
    --method dynamic \
    --benchmark
```

**Benefits:**
- No calibration data needed
- ~4x smaller model size
- 1.5-2x faster inference
- Very easy to use

---

### Option 2: Export to ONNX, Then Quantize (Best for Mobile)

**Step 1: Export to ONNX**
```bash
uv run python scripts/export_onnx.py \
    -c /mnt/c/Users/alexy/Documents/Projects/pixel-sorcery/checkpoints/gan_unet_512px_bs8_20251214_082125/best_model.pt \
    -o bin/model_gan_512.onnx \
    -s 512
```

**Step 2: Quantize ONNX (Dynamic)**
```bash
uv run python scripts/quantize_onnx.py \
    --model bin/model_gan_512.onnx \
    --output bin/model_gan_512_quantized.onnx \
    --method dynamic \
    --benchmark
```

**Benefits:**
- Best for mobile deployment
- Works with ONNX Runtime
- ~4x smaller size
- 1.5-2x faster

---

### Option 3: Static Quantization (Best Quality/Speed)

Static quantization uses calibration data for better accuracy.

**For PyTorch:**
```bash
uv run python scripts/quantize_pytorch.py \
    --checkpoint /mnt/c/Users/alexy/Documents/Projects/pixel-sorcery/checkpoints/gan_unet_512px_bs8_20251214_082125/best_model.pt \
    --output checkpoints/best_model_static_quantized.pt \
    --method static \
    --calib_images autohdr-real-estate-577/images/input \
    --calib_samples 100 \
    --image_size 512 \
    --benchmark
```

**For ONNX:**
```bash
uv run python scripts/quantize_onnx.py \
    --model bin/model_gan_512.onnx \
    --output bin/model_gan_512_static_quantized.onnx \
    --method static \
    --calib_images autohdr-real-estate-577/images/input \
    --calib_samples 100 \
    --image_size 512 \
    --benchmark
```

**Benefits:**
- Best performance (2-4x faster)
- Best size reduction (~4x smaller)
- Both weights and activations quantized
- Minimal quality loss

---

## Comparison: Dynamic vs Static

| Method | Size Reduction | Speed Improvement | Calibration Data | Setup Difficulty |
|--------|----------------|-------------------|------------------|------------------|
| **Dynamic** | ~4x | 1.5-2x | Not needed | Easy ✓ |
| **Static** | ~4x | 2-4x | Required | Medium |

---

## Testing Quantized Models

### Test PyTorch Quantized Model

```bash
# Use regular inference script - it handles quantized models
uv run python inference.py \
    --checkpoint checkpoints/best_model_quantized.pt \
    --input test_image.jpg \
    --output test_quantized_output.jpg
```

### Test ONNX Quantized Model

```bash
# Use ONNX test script
uv run python scripts/test_onnx.py \
    --model bin/model_gan_512_quantized.onnx \
    --input test_image.jpg \
    --output test_quantized_output.jpg

# Benchmark speed
uv run python scripts/benchmark_onnx.py \
    --model bin/model_gan_512_quantized.onnx \
    --runs 50
```

### Compare Original vs Quantized

```bash
# Compare outputs side-by-side
uv run python scripts/compare_pytorch_onnx.py \
    --checkpoint checkpoints/best_model.pt \
    --onnx bin/model_gan_512_quantized.onnx \
    --image test_image.jpg \
    --output_dir comparison_quantized
```

---

## Expected Results

### Model Size
- **Original**: ~44 MB (PyTorch) or ~12 MB (ONNX FP32)
- **Quantized**: ~11 MB (PyTorch) or ~3 MB (ONNX INT8)
- **Reduction**: ~75% smaller

### Inference Speed
- **Dynamic Quantization**: 1.5-2x faster
- **Static Quantization**: 2-4x faster
- **Platform dependent**: CPU sees biggest gains

### Quality
- **Dynamic**: Minimal quality loss (<1% difference)
- **Static**: Very small quality loss (1-2% difference with good calibration)

---

## Workflow Recommendation

### For Development/Testing
1. Train model
2. Export to ONNX
3. Apply dynamic quantization (fast, no calibration needed)
4. Test quality

### For Production/Mobile
1. Train model
2. Export to ONNX
3. Apply static quantization with 100-200 calibration images
4. Test quality and speed
5. Deploy quantized ONNX model

---

## Troubleshooting

### Quality Loss Too High

If quantized model quality is poor:

1. **Use more calibration samples** (200-500 images)
   ```bash
   --calib_samples 500
   ```

2. **Use diverse calibration data** (cover different scenes)

3. **Try dynamic quantization** (if static is problematic)

4. **Check quantization precision**
   - Some layers may be sensitive to quantization
   - Consider mixed precision (some layers FP32, some INT8)

### Quantization Fails

Common issues:
- **Out of memory**: Reduce `--calib_samples`
- **Model too complex**: Try dynamic instead of static
- **ONNX opset issues**: Re-export with `--opset_version 13`

### No Speed Improvement

If quantized model isn't faster:
- Ensure running on **CPU** (quantization optimizes CPU inference)
- GPU inference may not benefit from INT8
- Try ONNX Runtime with quantized model
- Check if hardware supports INT8 instructions

---

## Advanced Options

### Custom Calibration Dataset

Create a representative calibration dataset:

```bash
# Use validation set or specific scenarios
mkdir calibration_images
cp autohdr-real-estate-577/images/input/*.jpg calibration_images/

# Run static quantization
uv run python scripts/quantize_onnx.py \
    --model bin/model_gan_512.onnx \
    --output bin/model_quantized.onnx \
    --method static \
    --calib_images calibration_images \
    --calib_samples 200
```

### Benchmark All Variants

```bash
# Original ONNX
uv run python scripts/benchmark_onnx.py --model bin/model_gan_512.onnx

# Dynamic quantized
uv run python scripts/benchmark_onnx.py --model bin/model_gan_512_dynamic.onnx

# Static quantized
uv run python scripts/benchmark_onnx.py --model bin/model_gan_512_static.onnx
```

---

## Summary

**For your model:**
```bash
# Start with this (easiest):
uv run python scripts/quantize_pytorch.py \
    --checkpoint /mnt/c/Users/alexy/Documents/Projects/pixel-sorcery/checkpoints/gan_unet_512px_bs8_20251214_082125/best_model.pt \
    --output checkpoints/best_model_quantized.pt \
    --method dynamic \
    --benchmark
```

This will give you:
- ✓ ~4x smaller model (44MB → 11MB)
- ✓ 1.5-2x faster inference
- ✓ Minimal quality loss
- ✓ No calibration data needed

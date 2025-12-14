# NAFNet Real Estate Photo Enhancement

Fine-tuned NAFNet model for enhancing real estate photography. Improves lighting, color balance, and overall image quality.

## Models

**Download from Hugging Face:** [SebRincon/nafnet-realestate](https://huggingface.co/SebRincon/nafnet-realestate)

| Format | File | Size | Use Case |
|--------|------|------|----------|
| PyTorch | [`nafnet_realestate.pth`](https://huggingface.co/SebRincon/nafnet-realestate/blob/main/nafnet_realestate.pth) | 117 MB | Training, inference |
| ONNX | [`nafnet_realestate.onnx`](https://huggingface.co/SebRincon/nafnet-realestate/blob/main/nafnet_realestate.onnx) | 117 MB | Mobile/cross-platform |

## Quick Start

### Download Model
```bash
# Using Hugging Face CLI
pip install huggingface_hub
huggingface-cli download SebRincon/nafnet-realestate nafnet_realestate.pth

# Or direct download
wget https://huggingface.co/SebRincon/nafnet-realestate/resolve/main/nafnet_realestate.pth
```

### Run Inference
```bash
# Install dependencies
pip install torch torchvision opencv-python

# Run on single image
python inference.py --input photo.jpg --output enhanced.jpg

# Run on folder
python inference.py --input ./photos/ --output ./enhanced/
```

### Validate Model Output vs Dataset
Creates triptychs `(LQ | PRED | GT)` across `datasets/realestate/{train,val}` and logs per-image RAM/time.

```bash
conda activate nafnet  # or your env with onnxruntime + opencv + psutil installed
python create_dataset_triptychs.py --backend torch --device cuda
```

Outputs land in `nafnet-realestate/dataset_triptychs/onnx/<timestamp>/` (`metrics.csv`, `summary.json`, plus images).
Outputs land in `nafnet-realestate/dataset_triptychs/<backend>/<timestamp>/` (`metrics.csv`, `summary.json`, plus images).

## Performance

| Metric | Value |
|--------|-------|
| **PSNR** | 21.69 dB |
| **SSIM** | 0.8968 |
| **Speed** | 4s per 7MP image |
| **RAM** | 581 MB |
| **GPU** | 8.3 GB VRAM |

See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for detailed benchmarks.

## Mobile Deployment (iOS)

The model fits comfortably on mobile devices:

| Resolution | RAM | Time |
|------------|-----|------|
| 1080p | 150-250 MB | ~1-2s |
| 1440p | 250-400 MB | ~2-3s |
| 3K+ | 500-800 MB | ~3-5s |

### Convert to Core ML
```bash
# Generate ONNX (if not downloaded)
python convert_to_coreml.py

# On macOS: convert to Core ML
cd mobile_models
pip install coremltools
python convert_on_mac.py
```

See [mobile_models/README.md](mobile_models/README.md) for iOS integration guide.

## Project Structure

```
nafnet-realestate/
├── inference.py              # Run model on images
├── benchmark_inference.py    # Detailed performance testing
├── convert_to_coreml.py      # Export to ONNX
├── create_comparisons.py     # Side-by-side comparisons
├── create_dataset_triptychs.py  # (LQ | PRED | GT) dataset triptychs + RAM/time logs
├── BENCHMARK_RESULTS.md      # Full benchmark data
├── MODEL_CARD.md             # Hugging Face model card
├── mobile_models/
│   ├── README.md             # iOS integration guide
│   └── convert_on_mac.py     # ONNX → Core ML
├── real_estate_test/
│   ├── input/                # 100 test images
│   ├── output/               # Enhanced results
│   └── comparison/           # Side-by-side views
└── test_input/               # Quick test images
```

## Training

Trained on 577 before/after real estate photo pairs using:
- **Architecture**: NAFNet (width=32, 29.2M params)
- **Base**: NAFNet-SIDD-width32 pretrained weights
- **Loss**: L1 + Perceptual (VGG19)
- **Time**: ~5 hours on NVIDIA GB10

## Links

- **Models**: [Hugging Face](https://huggingface.co/SebRincon/nafnet-realestate)
- **Original NAFNet**: [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet)
- **Paper**: [Simple Baselines for Image Restoration](https://arxiv.org/abs/2204.04676)

## License

Apache 2.0

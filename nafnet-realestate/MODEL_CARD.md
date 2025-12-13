---
license: apache-2.0
language:
- en
tags:
- image-enhancement
- real-estate
- photo-enhancement
- nafnet
- image-restoration
- pytorch
- onnx
- coreml
- ios
pipeline_tag: image-to-image
library_name: pytorch
datasets:
- custom
metrics:
- psnr
- ssim
model-index:
- name: NAFNet Real Estate Enhancement
  results:
  - task:
      type: image-enhancement
      name: Image Enhancement
    metrics:
    - type: psnr
      value: 21.69
      name: PSNR
    - type: ssim
      value: 0.8968
      name: SSIM
---

# NAFNet Real Estate Enhancement

A fine-tuned NAFNet model for enhancing real estate photography. Trained on 577 before/after image pairs to improve lighting, color, and overall image quality.

## Model Details

| Metric | Value |
|--------|-------|
| **Architecture** | NAFNet (width=32) |
| **Parameters** | 29.2 million |
| **Model Size** | 111 MB (FP32) / 56 MB (FP16) |
| **Training Time** | 5 hours |
| **Training Images** | 577 pairs |
| **Final PSNR** | 21.69 dB |
| **Final SSIM** | 0.8968 |

## Available Formats

| Format | File | Size | Use Case |
|--------|------|------|----------|
| PyTorch | `nafnet_realestate.pth` | 117 MB | Training, fine-tuning |
| ONNX | `nafnet_realestate.onnx` | 117 MB | Cross-platform deployment |
| Core ML | Convert from ONNX | ~56 MB | iOS/macOS apps |

## Performance Benchmarks

Tested on 100 high-resolution real estate images (avg 7.25 megapixels):

### Timing
| Metric | Value |
|--------|-------|
| Average per image | 4.0 seconds |
| Throughput | 0.25 images/second |
| Megapixels/second | 1.81 MP/s |

### Memory Usage
| Resource | Usage |
|----------|-------|
| **RAM** | 581 MB total |
| **GPU VRAM** | 8.3 GB peak |

### Scaling by Resolution
| Resolution | RAM | GPU | Time |
|------------|-----|-----|------|
| 1080p (2.1 MP) | 150-250 MB | ~2.5 GB | ~1.2s |
| 1440p (3.7 MP) | 250-400 MB | ~4.3 GB | ~2.0s |
| 3K (7.3 MP) | 500-800 MB | ~8.3 GB | ~4.0s |
| 4K (8.3 MP) | 600-900 MB | ~9.5 GB | ~4.6s |

## Usage

### PyTorch
```python
import torch
from PIL import Image
import numpy as np

# Load model
model = NAFNet(img_channel=3, width=32, middle_blk_num=12,
               enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
checkpoint = torch.load("nafnet_realestate.pth", map_location="cpu")
model.load_state_dict(checkpoint["params"])
model.eval()

# Process image
img = Image.open("input.jpg")
img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

with torch.no_grad():
    output = model(img_tensor)

output_img = (output.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
Image.fromarray(output_img).save("enhanced.jpg")
```

### ONNX Runtime
```python
import onnxruntime as ort
import numpy as np
from PIL import Image

sess = ort.InferenceSession("nafnet_realestate.onnx")
img = np.array(Image.open("input.jpg")).astype(np.float32) / 255.0
img = img.transpose(2, 0, 1)[np.newaxis, ...]

output = sess.run(None, {"input": img})[0]
output_img = (output[0].transpose(1, 2, 0) * 255).astype(np.uint8)
Image.fromarray(output_img).save("enhanced.jpg")
```

## Mobile Deployment (iOS)

All resolutions fit within typical mobile RAM budgets (3-4 GB):

1. Convert ONNX to Core ML on macOS:
```bash
pip install coremltools
python convert_on_mac.py
```

2. Add `.mlpackage` to Xcode project
3. Use Vision framework for inference

## Training

- **Framework**: BasicSR + PyTorch
- **Base Model**: NAFNet-SIDD-width32 (pretrained on denoising)
- **Loss**: L1 + Perceptual (VGG19)
- **Optimizer**: AdamW (lr=1e-3)
- **Iterations**: 12,000

## License

Apache 2.0

## Citation

```bibtex
@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
```

## Links

- **GitHub**: [SebRincon/pixel-sorcery](https://github.com/SebRincon/pixel-sorcery/tree/sebastian/nafnet-realestate)
- **Original NAFNet**: [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet)

# NAFNet Real Estate Enhancement - Benchmark Results

## Model Information

| Metric | Value |
|--------|-------|
| **Architecture** | NAFNet (width=32) |
| **Parameters** | 29.2 million |
| **Model Size** | 111 MB (FP32) |
| **Training Time** | 5 hours |
| **Training Images** | 577 pairs |
| **Final PSNR** | 21.69 dB |
| **Final SSIM** | 0.8968 |

## Benchmark Configuration

- **Test Images**: 100 high-resolution real estate photos
- **Average Resolution**: 3300x2200 (7.25 megapixels)
- **Hardware**: NVIDIA GB10 (128GB unified memory)
- **Framework**: PyTorch 2.9 + BasicSR

## Performance Results

### Timing

| Metric | Value |
|--------|-------|
| **Total time (100 images)** | 6.7 minutes |
| **Average per image** | 4.0 seconds |
| **Minimum** | 3.25 seconds |
| **Maximum** | 4.53 seconds |
| **Throughput** | 0.25 images/second |
| **Megapixels/second** | 1.81 MP/s |

### Memory Usage - RAM

| Stage | RAM Usage |
|-------|-----------|
| **Baseline (Python)** | 736 MB |
| **After model load** | 905 MB |
| **Peak during inference** | 1,317 MB |
| **Model overhead** | +169 MB |
| **Inference overhead** | +412 MB |
| **Total net usage** | **581 MB** |

### Memory Usage - GPU

| Metric | Value |
|--------|-------|
| **Model size** | 111 MB |
| **Peak allocated** | 8,258 MB |
| **Peak reserved** | 12,050 MB |
| **Net for inference** | 8,147 MB |

## Scaling by Resolution

| Resolution | Megapixels | Est. RAM | Est. GPU | Est. Time |
|------------|------------|----------|----------|-----------|
| 1080p | 2.1 MP | 150-250 MB | ~2.5 GB | ~1.2s |
| 1440p | 3.7 MP | 250-400 MB | ~4.3 GB | ~2.0s |
| 3K | 7.3 MP | 500-800 MB | ~8.3 GB | ~4.0s |
| 4K | 8.3 MP | 600-900 MB | ~9.5 GB | ~4.6s |

## Mobile Deployment (iOS)

| Format | Size | Precision |
|--------|------|-----------|
| PyTorch | 111 MB | FP32 |
| ONNX | 112 MB | FP32 |
| Core ML | ~56 MB | FP16 |

### iOS Memory Requirements

All resolutions fit within 3-4 GB RAM budget:
- 1080p: 150-250 MB ✅
- 1440p: 250-400 MB ✅
- 3K: 500-800 MB ✅
- 4K: 600-900 MB ✅

## Raw Benchmark Output

```
============================================================
NAFNet INFERENCE BENCHMARK
============================================================

[BASELINE MEMORY]
  Process RAM: 735.6 MB
  GPU Memory:  0.0 MB

[LOADING MODEL]
  Path: BasicSR/experiments/NAFNet_RealEstate_Fast/models/net_g_12000.pth
  Load time:   0.47s
  Model RAM:   169.1 MB (process now: 904.7 MB)
  Model GPU:   111.3 MB (allocated now: 111.3 MB)

[PROCESSING 100 IMAGES]
------------------------------------------------------------
  [  1/100] 0000_val.jpg: 3301x2199 (7.26MP) | 4.525s | RAM: 1254MB | GPU: 111MB
  [  2/100] 0001_val.jpg: 3299x2200 (7.26MP) | 3.975s | RAM: 1254MB | GPU: 111MB
  ...
  [100/100] 0519_train.jpg: 3300x2199 (7.26MP) | 4.114s | RAM: 1295MB | GPU: 111MB

============================================================
BENCHMARK RESULTS
============================================================

[IMAGES PROCESSED]
  Total:       100 images
  Megapixels:  724.6 MP total (7.25 MP avg)

[TIMING]
  Total time:  399.88s
  Avg/image:   3.999s
  Min:         3.252s
  Max:         4.525s
  Throughput:  0.25 img/s
  MP/second:   1.81 MP/s

[MEMORY - RAM]
  Baseline:    735.6 MB
  After model: 904.7 MB (+169.1 MB for model)
  Peak:        1316.6 MB
  Net usage:   581.0 MB (model + inference)

[MEMORY - GPU]
  Model size:      111.3 MB
  Peak allocated:  8258.4 MB
  Peak reserved:   12050.0 MB
  Net for inference: 8147.1 MB

============================================================
```

## Files

Models available on Hugging Face: `SebRincon/pixel-sorcery`
- `nafnet_realestate.pth` - Trained PyTorch model
- `nafnet_realestate.onnx` - ONNX format for deployment

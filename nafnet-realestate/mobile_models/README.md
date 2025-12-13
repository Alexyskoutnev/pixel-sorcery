# NAFNet Mobile Deployment

## Download Models

**Hugging Face:** [SebRincon/nafnet-realestate](https://huggingface.co/SebRincon/nafnet-realestate)

```bash
# Download ONNX model directly
wget https://huggingface.co/SebRincon/nafnet-realestate/resolve/main/nafnet_realestate.onnx
```

## Model Stats
- **Parameters**: 29.2 million
- **ONNX size**: ~112 MB (FP32)
- **Core ML size**: ~56 MB (FP16)

## Memory Requirements by Resolution

| Resolution | Megapixels | RAM Usage |
|------------|------------|-----------|
| 1080p | 2.1 MP | 150-250 MB |
| 1440p | 3.7 MP | 250-400 MB |
| 3K | 7.3 MP | 500-800 MB |
| 4K | 8.3 MP | 600-900 MB |

## Get ONNX Model

**Option 1: Download from Hugging Face** (recommended)
```bash
wget https://huggingface.co/SebRincon/nafnet-realestate/resolve/main/nafnet_realestate.onnx
```

**Option 2: Generate locally**
```bash
# From nafnet-realestate directory
python convert_to_coreml.py
```

This creates:
- `mobile_models/nafnet_realestate.onnx` (~112 MB)

## Convert to Core ML (macOS only)

```bash
cd mobile_models
pip install coremltools
python convert_on_mac.py
```

This creates:
- `NAFNetRealEstate.mlpackage` (~56 MB, FP16)

## iOS Integration

```swift
import CoreML
import Vision

// Load model
let config = MLModelConfiguration()
config.computeUnits = .all  // Use Neural Engine
let model = try! NAFNetRealEstate(configuration: config)

// Create request
let visionModel = try! VNCoreMLModel(for: model.model)
let request = VNCoreMLRequest(model: visionModel) { request, error in
    guard let results = request.results as? [VNPixelBufferObservation],
          let output = results.first?.pixelBuffer else { return }
    // Use enhanced image
}

// Process image
let handler = VNImageRequestHandler(cgImage: inputImage, options: [:])
try! handler.perform([request])
```

## Pricing Tiers Suggestion

| Tier | Resolution | Est. Time (ANE) | Use Case |
|------|------------|-----------------|----------|
| Free/Preview | 1080p | ~1-2s | Quick preview |
| Standard | 1440p | ~2-3s | Social media |
| Premium | 3K+ | ~3-5s | Print/professional |

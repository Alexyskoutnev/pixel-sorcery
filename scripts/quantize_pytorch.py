#!/usr/bin/env python3
"""
Quantize PyTorch models for faster inference and smaller size.

Usage:
    # Dynamic quantization (easiest, no calibration data needed)
    python scripts/quantize_pytorch.py \
        --checkpoint checkpoints/gan_unet_512px_bs8_20251214_082125/best_model.pt \
        --output checkpoints/best_model_quantized.pt \
        --method dynamic

    # Static quantization (best quality, requires calibration data)
    python scripts/quantize_pytorch.py \
        --checkpoint checkpoints/gan_unet_512px_bs8_20251214_082125/best_model.pt \
        --output checkpoints/best_model_quantized.pt \
        --method static \
        --calib_images autohdr-real-estate-577/images/input
"""

import argparse
import sys
from pathlib import Path
import time

import torch
import torch.quantization
from torch.quantization import quantize_dynamic, get_default_qconfig
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import get_model


def load_model(checkpoint_path: str, device: torch.device):
    """Load PyTorch model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = checkpoint.get("model_config", {})
    model_name = model_config.get("name", "unet")

    # Handle name mapping
    name_mapping = {"UNetSmall": "unet_small", "UNet": "unet", "UNetLarge": "unet_large"}
    model_name = name_mapping.get(model_name, model_name)

    # Create model with correct architecture parameters
    model_kwargs = {}
    if "base_channels" in model_config:
        model_kwargs["base_channels"] = model_config["base_channels"]
        print(f"Using base_channels={model_config['base_channels']} from checkpoint")
    if "depth" in model_config:
        model_kwargs["depth"] = model_config["depth"]
        print(f"Using depth={model_config['depth']} from checkpoint")

    model = get_model(model_name, **model_kwargs)

    # Load weights (handle both GAN and regular checkpoints)
    if "generator_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["generator_state_dict"])
        print("Loaded GAN generator weights")
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded model weights")
    else:
        raise ValueError("No model weights found in checkpoint")

    model.to(device)
    model.eval()

    return model, model_config


def get_model_size(model):
    """Calculate model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


def dynamic_quantization(model):
    """
    Apply dynamic quantization.

    Pros:
    - No calibration data needed
    - Easy to apply
    - Reduces model size

    Cons:
    - Less speedup than static quantization
    - Weights quantized, activations computed in fp32
    """
    print("\nApplying dynamic quantization...")
    print("This quantizes weights to int8, activations stay fp32")

    # Quantize Conv2d and Linear layers
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Conv2d, torch.nn.Linear},
        dtype=torch.qint8
    )

    return quantized_model


def prepare_calibration_data(calib_dir: str, num_samples: int = 100, image_size: int = 512):
    """Load calibration images for static quantization."""
    calib_dir = Path(calib_dir)

    if not calib_dir.exists():
        raise ValueError(f"Calibration directory not found: {calib_dir}")

    # Find images
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = [f for f in calib_dir.iterdir() if f.suffix.lower() in extensions]

    if not image_files:
        raise ValueError(f"No images found in {calib_dir}")

    # Limit to num_samples
    image_files = image_files[:num_samples]
    print(f"Loading {len(image_files)} calibration images...")

    calib_data = []
    for img_path in tqdm(image_files, desc="Loading calibration data"):
        img = Image.open(img_path).convert("RGB")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        img_tensor = TF.to_tensor(img).unsqueeze(0)
        calib_data.append(img_tensor)

    return calib_data


def static_quantization(model, calib_data):
    """
    Apply static quantization with calibration.

    Pros:
    - Best performance (speed + size)
    - Both weights and activations quantized to int8

    Cons:
    - Requires calibration data
    - More complex to set up
    - May have slight quality loss
    """
    print("\nApplying static quantization...")
    print("This quantizes both weights and activations to int8")

    # Set quantization config
    model.qconfig = get_default_qconfig('fbgemm')

    # Prepare model for quantization
    model_prepared = torch.quantization.prepare(model, inplace=False)

    # Calibrate with sample data
    print("Calibrating model...")
    model_prepared.eval()
    with torch.no_grad():
        for data in tqdm(calib_data, desc="Calibration"):
            model_prepared(data)

    # Convert to quantized model
    print("Converting to quantized model...")
    quantized_model = torch.quantization.convert(model_prepared, inplace=False)

    return quantized_model


def benchmark_model(model, input_size=(1, 3, 512, 512), num_runs=20):
    """Benchmark inference speed."""
    dummy_input = torch.randn(input_size)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            times.append(time.time() - start)

    import numpy as np
    times = np.array(times) * 1000  # Convert to ms

    return {
        "mean": np.mean(times),
        "median": np.median(times),
        "min": np.min(times),
        "max": np.max(times)
    }


def save_quantized_model(model, output_path: str, model_config: dict, metadata: dict):
    """Save quantized model with metadata."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "quantization_metadata": metadata
    }

    torch.save(checkpoint, output_path)
    print(f"\nSaved quantized model to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Quantize PyTorch models")

    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to save quantized model"
    )
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=["dynamic", "static"],
        default="dynamic",
        help="Quantization method (default: dynamic)"
    )
    parser.add_argument(
        "--calib_images",
        type=str,
        default=None,
        help="Path to calibration images (required for static quantization)"
    )
    parser.add_argument(
        "--calib_samples",
        type=int,
        default=100,
        help="Number of calibration samples (default: 100)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Image size for calibration (default: 512)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark inference speed before/after quantization"
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return

    if args.method == "static" and not args.calib_images:
        print("Error: --calib_images required for static quantization")
        return

    print(f"{'='*70}")
    print(f"PYTORCH MODEL QUANTIZATION")
    print(f"{'='*70}")
    print(f"Method: {args.method}")
    print(f"Input: {args.checkpoint}")
    print(f"Output: {args.output}")

    # Load model
    device = torch.device("cpu")  # Quantization requires CPU
    model, model_config = load_model(args.checkpoint, device)

    # Get original model size
    original_size = get_model_size(model)
    print(f"\nOriginal model size: {original_size:.2f} MB")

    # Benchmark original model
    if args.benchmark:
        print("\nBenchmarking original model...")
        original_speed = benchmark_model(
            model,
            input_size=(1, 3, args.image_size, args.image_size)
        )
        print(f"Original inference time: {original_speed['mean']:.2f}ms (avg)")

    # Apply quantization
    if args.method == "dynamic":
        quantized_model = dynamic_quantization(model)
        metadata = {
            "method": "dynamic",
            "dtype": "qint8",
            "layers": "Conv2d, Linear"
        }

    elif args.method == "static":
        # Load calibration data
        calib_data = prepare_calibration_data(
            args.calib_images,
            num_samples=args.calib_samples,
            image_size=args.image_size
        )

        quantized_model = static_quantization(model, calib_data)
        metadata = {
            "method": "static",
            "dtype": "qint8",
            "calib_samples": len(calib_data),
            "calib_source": args.calib_images
        }

    # Get quantized model size
    quantized_size = get_model_size(quantized_model)
    size_reduction = (1 - quantized_size / original_size) * 100

    print(f"\n{'='*70}")
    print(f"QUANTIZATION RESULTS")
    print(f"{'='*70}")
    print(f"Original size:   {original_size:.2f} MB")
    print(f"Quantized size:  {quantized_size:.2f} MB")
    print(f"Size reduction:  {size_reduction:.1f}%")

    # Benchmark quantized model
    if args.benchmark:
        print("\nBenchmarking quantized model...")
        quantized_speed = benchmark_model(
            quantized_model,
            input_size=(1, 3, args.image_size, args.image_size)
        )
        speedup = original_speed['mean'] / quantized_speed['mean']

        print(f"\n{'='*70}")
        print(f"SPEED COMPARISON")
        print(f"{'='*70}")
        print(f"Original:   {original_speed['mean']:.2f}ms")
        print(f"Quantized:  {quantized_speed['mean']:.2f}ms")
        print(f"Speedup:    {speedup:.2f}x")

    # Save quantized model
    save_quantized_model(quantized_model, args.output, model_config, metadata)

    print(f"\n{'='*70}")
    print("✓ Quantization complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

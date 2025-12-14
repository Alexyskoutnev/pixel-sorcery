#!/usr/bin/env python3
"""
Quantize ONNX models for faster inference and smaller size.

Usage:
    # Dynamic quantization (easiest)
    python scripts/quantize_onnx.py \
        --model bin/model_gan_512.onnx \
        --output bin/model_gan_512_quantized.onnx \
        --method dynamic

    # Static quantization (best quality, requires calibration)
    python scripts/quantize_onnx.py \
        --model bin/model_gan_512.onnx \
        --output bin/model_gan_512_quantized.onnx \
        --method static \
        --calib_images autohdr-real-estate-577/images/input
"""

import argparse
from pathlib import Path
import time

import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, quantize_static, CalibrationDataReader, QuantType
import numpy as np
from PIL import Image
from tqdm import tqdm


class ImageCalibrationDataReader(CalibrationDataReader):
    """Calibration data reader for image models."""

    def __init__(self, image_dir: str, input_name: str, num_samples: int = 100, image_size: int = 512):
        self.image_dir = Path(image_dir)
        self.input_name = input_name
        self.image_size = image_size

        # Find images
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        self.image_files = [f for f in self.image_dir.iterdir() if f.suffix.lower() in extensions]
        self.image_files = self.image_files[:num_samples]

        print(f"Found {len(self.image_files)} calibration images")

        self.index = 0

    def get_next(self):
        """Get next calibration sample."""
        if self.index >= len(self.image_files):
            return None

        # Load and preprocess image
        img_path = self.image_files[self.index]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Convert to numpy array
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
        img_batch = np.expand_dims(img_tensor, axis=0)   # Add batch dimension

        self.index += 1

        return {self.input_name: img_batch}


def get_model_size(model_path: str):
    """Get model file size in MB."""
    size_bytes = Path(model_path).stat().st_size
    size_mb = size_bytes / (1024 ** 2)
    return size_mb


def benchmark_onnx_model(model_path: str, input_size=(1, 3, 512, 512), num_runs=20):
    """Benchmark ONNX model inference speed."""
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name

    # Create dummy input
    dummy_input = np.random.randn(*input_size).astype(np.float32)

    # Warmup
    for _ in range(5):
        sess.run(None, {input_name: dummy_input})

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        sess.run(None, {input_name: dummy_input})
        times.append(time.time() - start)

    times = np.array(times) * 1000  # Convert to ms

    return {
        "mean": np.mean(times),
        "median": np.median(times),
        "min": np.min(times),
        "max": np.max(times)
    }


def dynamic_quantize_onnx(model_path: str, output_path: str):
    """
    Apply dynamic quantization to ONNX model.

    Pros:
    - No calibration data needed
    - Easy to apply
    - Reduces model size ~4x

    Cons:
    - Activations still computed in fp32
    - Less speedup than static quantization
    """
    print("\nApplying dynamic quantization...")
    print("This quantizes weights to int8, activations stay fp32")

    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8
    )

    print(f"Saved quantized model to: {output_path}")


def static_quantize_onnx(model_path: str, output_path: str, calib_reader: CalibrationDataReader):
    """
    Apply static quantization to ONNX model.

    Pros:
    - Best performance (speed + size)
    - Both weights and activations quantized

    Cons:
    - Requires calibration data
    - Takes longer to quantize
    """
    print("\nApplying static quantization...")
    print("This quantizes both weights and activations to int8")

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calib_reader,
        quant_format=QuantType.QUInt8
    )

    print(f"Saved quantized model to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX models")

    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to save quantized model"
    )
    parser.add_argument(
        "--method",
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
        help="Image size (default: 512)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark inference speed before/after quantization"
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        return

    if args.method == "static" and not args.calib_images:
        print("Error: --calib_images required for static quantization")
        return

    print(f"{'='*70}")
    print(f"ONNX MODEL QUANTIZATION")
    print(f"{'='*70}")
    print(f"Method: {args.method}")
    print(f"Input: {args.model}")
    print(f"Output: {args.output}")

    # Get original model size
    original_size = get_model_size(args.model)
    print(f"\nOriginal model size: {original_size:.2f} MB")

    # Benchmark original model
    if args.benchmark:
        print("\nBenchmarking original model...")
        original_speed = benchmark_onnx_model(
            args.model,
            input_size=(1, 3, args.image_size, args.image_size)
        )
        print(f"Original inference time: {original_speed['mean']:.2f}ms (avg)")

    # Apply quantization
    if args.method == "dynamic":
        dynamic_quantize_onnx(args.model, args.output)

    elif args.method == "static":
        # Get input name from model
        sess = ort.InferenceSession(args.model)
        input_name = sess.get_inputs()[0].name

        # Create calibration data reader
        print(f"\nLoading calibration data from: {args.calib_images}")
        calib_reader = ImageCalibrationDataReader(
            args.calib_images,
            input_name,
            num_samples=args.calib_samples,
            image_size=args.image_size
        )

        print("\nCalibrating model (this may take a few minutes)...")
        static_quantize_onnx(args.model, args.output, calib_reader)

    # Get quantized model size
    quantized_size = get_model_size(args.output)
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
        quantized_speed = benchmark_onnx_model(
            args.output,
            input_size=(1, 3, args.image_size, args.image_size)
        )
        speedup = original_speed['mean'] / quantized_speed['mean']

        print(f"\n{'='*70}")
        print(f"SPEED COMPARISON")
        print(f"{'='*70}")
        print(f"Original:   {original_speed['mean']:.2f}ms")
        print(f"Quantized:  {quantized_speed['mean']:.2f}ms")
        print(f"Speedup:    {speedup:.2f}x")

    print(f"\n{'='*70}")
    print("✓ Quantization complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

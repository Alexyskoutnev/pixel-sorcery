#!/usr/bin/env python3
"""
Benchmark ONNX model inference speed.

Usage:
    python scripts/benchmark_onnx.py --model bin/model_gan_512.onnx
    python scripts/benchmark_onnx.py --model bin/model_gan_512.onnx --runs 50
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def benchmark_onnx(model_path: str, runs: int = 20, warmup: int = 5):
    """
    Benchmark ONNX model inference speed.

    Args:
        model_path: Path to ONNX model
        runs: Number of benchmark runs
        warmup: Number of warmup runs
    """
    print(f"Loading model: {model_path}")

    # Check if file exists
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return

    # Load ONNX model
    sess = ort.InferenceSession(model_path)

    # Get input/output info
    input_info = sess.get_inputs()[0]
    output_info = sess.get_outputs()[0]
    input_name = input_info.name
    output_name = output_info.name
    input_shape = input_info.shape

    print(f"\nModel Info:")
    print(f"  Input name: {input_name}")
    print(f"  Input shape: {input_shape}")
    print(f"  Output name: {output_name}")
    print(f"  Output shape: {output_info.shape}")

    # Determine input size from shape
    # Shape is typically [batch, channels, height, width]
    if len(input_shape) == 4:
        batch_size = 1
        channels = input_shape[1] if isinstance(input_shape[1], int) else 3
        height = input_shape[2] if isinstance(input_shape[2], int) else 512
        width = input_shape[3] if isinstance(input_shape[3], int) else 512
    else:
        print("Warning: Unexpected input shape, using defaults")
        batch_size, channels, height, width = 1, 3, 512, 512

    print(f"\nUsing input size: [{batch_size}, {channels}, {height}, {width}]")

    # Create dummy input
    dummy_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)

    # Warmup runs
    print(f"\nWarming up ({warmup} runs)...")
    for i in range(warmup):
        sess.run([output_name], {input_name: dummy_input})
        print(f"  Warmup {i+1}/{warmup}", end="\r")
    print()

    # Benchmark runs
    print(f"\nBenchmarking ({runs} runs)...")
    times = []
    for i in range(runs):
        start = time.time()
        output = sess.run([output_name], {input_name: dummy_input})
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}/{runs}: {elapsed*1000:.2f}ms", end="\r")
    print()

    # Calculate statistics
    times_ms = np.array(times) * 1000

    print(f"\n{'='*50}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Average:  {np.mean(times_ms):.2f}ms")
    print(f"Median:   {np.median(times_ms):.2f}ms")
    print(f"Min:      {np.min(times_ms):.2f}ms")
    print(f"Max:      {np.max(times_ms):.2f}ms")
    print(f"Std Dev:  {np.std(times_ms):.2f}ms")
    print(f"{'='*50}")
    print(f"\nOutput shape: {output[0].shape}")
    print(f"Output dtype: {output[0].dtype}")
    print(f"Output range: [{output[0].min():.3f}, {output[0].max():.3f}]")


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX model inference speed")
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=20,
        help="Number of benchmark runs (default: 20)"
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=5,
        help="Number of warmup runs (default: 5)"
    )

    args = parser.parse_args()

    benchmark_onnx(args.model, args.runs, args.warmup)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convert NAFNet ONNX model to FP16 for mobile deployment.
Run: python quantize_model.py

Requires: pip install onnx onnxconverter-common
"""

import os
import sys
from pathlib import Path

try:
    import onnx
    from onnxconverter_common import float16
except ImportError:
    print("Missing packages. Install with:")
    print("  pip install onnx onnxconverter-common")
    sys.exit(1)

# Paths
MODEL_PATH = Path("flutter_app/assets/models/nafnet_realestate.onnx")
FP16_OUTPUT = Path("flutter_app/assets/models/nafnet_realestate_fp16.onnx")


def convert_to_fp16():
    """Convert model to FP16 for reduced memory usage with no quality loss"""
    print("NAFNet FP16 Conversion")
    print("=" * 40)
    print(f"Input:  {MODEL_PATH}")

    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return False

    original_size = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"Original size: {original_size:.1f} MB")

    print("\nConverting to FP16...")
    model = onnx.load(str(MODEL_PATH))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(FP16_OUTPUT))

    new_size = FP16_OUTPUT.stat().st_size / (1024 * 1024)

    print(f"\nOutput: {FP16_OUTPUT}")
    print(f"FP16 size: {new_size:.1f} MB")
    print(f"Reduction: {(1 - new_size/original_size) * 100:.1f}%")
    print("\nDone! The app will automatically use the FP16 model.")
    return True


if __name__ == "__main__":
    convert_to_fp16()

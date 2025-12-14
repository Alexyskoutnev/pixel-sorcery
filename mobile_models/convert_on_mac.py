#!/usr/bin/env python3
"""
Convert ONNX to Core ML - RUN THIS ON macOS
============================================

Requirements:
    pip install coremltools

Usage:
    python convert_on_mac.py
"""

import coremltools as ct
import os

ONNX_PATH = "nafnet_realestate.onnx"
COREML_PATH = "NAFNetRealEstate.mlpackage"


def main():
    print("=" * 60)
    print("NAFNet ONNX → Core ML Conversion")
    print("=" * 60)

    if not os.path.exists(ONNX_PATH):
        print(f"Error: {ONNX_PATH} not found!")
        print("Make sure you're running this in the mobile_models directory.")
        return

    print(f"\nConverting: {ONNX_PATH}")
    print(f"Output:     {COREML_PATH}")

    # Convert with FP16 precision for Neural Engine
    model = ct.convert(
        ONNX_PATH,
        source="onnx",
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )

    # Add metadata
    model.author = "NAFNet Real Estate Enhancement"
    model.short_description = "Enhances real estate photos using NAFNet neural network"
    model.version = "1.0"

    # Input/output specs
    model.input_description["input"] = "Input image tensor (1, 3, H, W) normalized to 0-1 range"
    model.output_description["output"] = "Enhanced image tensor (1, 3, H, W) normalized to 0-1 range"

    # Save
    model.save(COREML_PATH)

    # Get size
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, fn in os.walk(COREML_PATH)
        for f in fn
    ) / (1024 * 1024)

    print(f"\n✅ Core ML model saved!")
    print(f"   Size: {size_mb:.1f} MB")
    print(f"   Path: {COREML_PATH}")

    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Drag NAFNetRealEstate.mlpackage into Xcode project")
    print("2. Use Vision framework to run inference")
    print("3. Test on device with Neural Engine acceleration")

    print("""
Sample Swift code:

    import CoreML
    import Vision

    let model = try! NAFNetRealEstate(configuration: .init())
    let request = VNCoreMLRequest(model: try! VNCoreMLModel(for: model.model))

    // Process image...
""")


if __name__ == "__main__":
    main()

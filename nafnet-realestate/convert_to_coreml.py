#!/usr/bin/env python3
"""
Convert NAFNet model to Core ML format for iOS deployment.
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'BasicSR'))

from basicsr.archs.NAFNet_arch import NAFNet

MODEL_PATH = "BasicSR/experiments/NAFNet_RealEstate_Fast/models/net_g_12000.pth"
OUTPUT_DIR = "mobile_models"


def load_model():
    """Load trained NAFNet model."""
    model = NAFNet(
        img_channel=3,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2]
    )

    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    if 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'])
    elif 'params_ema' in checkpoint:
        model.load_state_dict(checkpoint['params_ema'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def export_to_onnx(model, onnx_path):
    """Export PyTorch model to ONNX format."""
    print(f"Exporting to ONNX: {onnx_path}")

    # Use flexible input size (batch=1, channels=3, height/width variable)
    # Start with 1080p as default
    dummy_input = torch.randn(1, 3, 1080, 1920)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {2: 'height', 3: 'width'},
            'output': {2: 'height', 3: 'width'}
        }
    )

    # Simplify ONNX model
    try:
        import onnx
        from onnxsim import simplify

        print("Simplifying ONNX model...")
        onnx_model = onnx.load(onnx_path)
        simplified_model, check = simplify(onnx_model)
        if check:
            onnx.save(simplified_model, onnx_path)
            print("ONNX simplification successful")
        else:
            print("ONNX simplification failed, using original")
    except Exception as e:
        print(f"ONNX simplification skipped: {e}")

    return onnx_path


def convert_to_coreml(onnx_path, coreml_path):
    """Convert ONNX model to Core ML."""
    import coremltools as ct

    print(f"Converting to Core ML: {coreml_path}")

    # Load ONNX model - use unified convert API
    model = ct.convert(
        onnx_path,
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,  # Use FP16 for Neural Engine
        convert_to="mlprogram",
    )

    # Set metadata
    model.author = "NAFNet Real Estate Enhancement"
    model.short_description = "Enhances real estate photos using NAFNet"
    model.version = "1.0"

    # Add input/output descriptions
    model.input_description['input'] = "Input image (RGB, normalized 0-1)"
    model.output_description['output'] = "Enhanced image (RGB, normalized 0-1)"

    # Save
    model.save(coreml_path)
    print(f"Core ML model saved: {coreml_path}")

    return coreml_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("NAFNet to Core ML Conversion")
    print("=" * 60)

    # Load PyTorch model
    print("\n[1/3] Loading PyTorch model...")
    model = load_model()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Export to ONNX
    print("\n[2/3] Exporting to ONNX...")
    onnx_path = os.path.join(OUTPUT_DIR, "nafnet_realestate.onnx")
    export_to_onnx(model, onnx_path)
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  ONNX size: {onnx_size:.1f} MB")

    # Convert to Core ML
    print("\n[3/3] Converting to Core ML...")
    coreml_path = os.path.join(OUTPUT_DIR, "NAFNetRealEstate.mlpackage")
    convert_to_coreml(onnx_path, coreml_path)

    # Get Core ML size
    import subprocess
    result = subprocess.run(['du', '-sh', coreml_path], capture_output=True, text=True)
    coreml_size = result.stdout.split()[0]

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  ONNX:    {onnx_path} ({onnx_size:.1f} MB)")
    print(f"  Core ML: {coreml_path} ({coreml_size})")
    print(f"\nNext steps:")
    print(f"  1. Copy {coreml_path} to your Xcode project")
    print(f"  2. Import and use with Vision framework")
    print(f"  3. Test on device with Neural Engine acceleration")


if __name__ == "__main__":
    main()

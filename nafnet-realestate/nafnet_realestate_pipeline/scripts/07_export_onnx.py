#!/usr/bin/env python3
"""
07_export_onnx.py - Export trained model to ONNX format for production

ONNX enables:
- Faster inference with TensorRT
- Deployment without PyTorch
- Cross-platform compatibility

Usage:
    python scripts/07_export_onnx.py \
        --model experiments/NAFNet_RealEstate_Fast/models/net_g_30000.pth \
        --output nafnet_realestate.onnx
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.onnx


def export_to_onnx(
    model_path: str,
    output_path: str,
    width: int = 32,
    input_size: tuple = (1, 3, 512, 512),
    opset_version: int = 17,
    dynamic_axes: bool = True,
    fp16: bool = False,
):
    """Export NAFNet model to ONNX format."""

    # Add BasicSR to path (local checkout under `nafnet-realestate/BasicSR`)
    project_root = Path(__file__).resolve().parents[2]
    basicsr_root = project_root / "BasicSR"
    if str(basicsr_root) not in sys.path:
        sys.path.insert(0, str(basicsr_root))

    from basicsr.archs.NAFNet_arch import NAFNet

    print(f"Loading model: {model_path}")

    # Create model
    model = NAFNet(
        img_channel=3,
        width=width,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2]
    )

    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'])
    elif 'params_ema' in checkpoint:
        model.load_state_dict(checkpoint['params_ema'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    if fp16:
        model = model.half()

    # Create dummy input
    dummy_input = torch.randn(*input_size)
    if fp16:
        dummy_input = dummy_input.half()

    # Configure dynamic axes
    if dynamic_axes:
        dynamic_axes_config = {
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'}
        }
    else:
        dynamic_axes_config = None

    print(f"Exporting to ONNX (opset {opset_version})...")
    print(f"  Input shape: {input_size}")
    print(f"  Dynamic axes: {dynamic_axes}")
    print(f"  DType: {'fp16' if fp16 else 'fp32'}")

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes_config
    )
    
    print(f"\n✅ Exported to: {output_path}")
    
    # Verify the model
    try:
        import onnx
        print("\nVerifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model is valid!")
        
        # Print model info
        file_size = os.path.getsize(output_path) / 1e6
        print(f"   File size: {file_size:.1f} MB")
        
    except ImportError:
        print("⚠️  Install 'onnx' package to verify: pip install onnx")
    except Exception as e:
        print(f"⚠️  Verification warning: {e}")
    
    return output_path


def convert_to_tensorrt(onnx_path: str, trt_path: str = None):
    """Convert ONNX model to TensorRT (optional)."""
    
    if trt_path is None:
        trt_path = onnx_path.replace('.onnx', '.trt')
    
    print(f"\nConverting to TensorRT: {trt_path}")
    print("Run this command manually:")
    print(f"""
    trtexec --onnx={onnx_path} \\
        --saveEngine={trt_path} \\
        --minShapes=input:1x3x256x256 \\
        --optShapes=input:1x3x512x512 \\
        --maxShapes=input:1x3x1024x1024 \\
        --fp16
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Export NAFNet model to ONNX format"
    )

    parser.add_argument('--model', '-m', required=True, help='Model checkpoint path')
    parser.add_argument('--output', '-o', default='nafnet_realestate.onnx', help='Output ONNX path')
    parser.add_argument('--width', type=int, default=32, help='NAFNet width (default: 32)')
    parser.add_argument('--input-size', type=int, nargs=4, default=[1, 3, 512, 512],
                        help='Input size: batch channels height width')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    parser.add_argument('--static', action='store_true', help='Use static input size (no dynamic axes)')
    parser.add_argument('--fp16', action='store_true', help='Export FP16 ONNX weights/ops (experimental)')
    parser.add_argument('--tensorrt', action='store_true', help='Print TensorRT conversion command')

    args = parser.parse_args()

    print("=" * 60)
    print("  NAFNet ONNX Export")
    print("=" * 60)
    
    # Export
    onnx_path = export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        width=args.width,
        input_size=tuple(args.input_size),
        opset_version=args.opset,
        dynamic_axes=not args.static,
        fp16=args.fp16,
    )
    
    # TensorRT instructions
    if args.tensorrt:
        convert_to_tensorrt(onnx_path)
    
    print("\n" + "=" * 60)
    print("  Export Complete!")
    print("=" * 60)
    print(f"""
    ONNX model: {onnx_path}
    
    Use with ONNX Runtime:
        import onnxruntime as ort
        session = ort.InferenceSession("{onnx_path}")
        output = session.run(None, {{"input": image_tensor}})
    
    Convert to TensorRT (optional, for faster inference):
        trtexec --onnx={onnx_path} --saveEngine=nafnet.trt --fp16
    """)


if __name__ == "__main__":
    main()

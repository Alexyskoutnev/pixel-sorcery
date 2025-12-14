#!/usr/bin/env python3
"""
Compare PyTorch and ONNX model outputs to verify correctness.

Usage:
    python scripts/compare_pytorch_onnx.py \
        --checkpoint checkpoints/gan_unet_512px/best_model.pt \
        --onnx bin/model_gan_512.onnx \
        --image test_image.jpg
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
import torchvision.transforms.functional as TF

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import get_model


def load_pytorch_model(checkpoint_path: str, device: torch.device):
    """Load PyTorch model from checkpoint."""
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

    return model


def process_with_pytorch(model, image_path: str, size: int, device: torch.device):
    """Process image with PyTorch model."""
    # Load and preprocess
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)

    # Convert to tensor
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(img_tensor)

    # Convert to numpy
    output_np = output.cpu().numpy()

    return output_np, img_tensor.cpu().numpy()


def process_with_onnx(session, input_name: str, image_path: str, size: int):
    """Process image with ONNX model."""
    # Load and preprocess
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)

    # Convert to numpy array
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
    img_batch = np.expand_dims(img_tensor, axis=0)   # Add batch dimension

    # Run inference
    output = session.run(None, {input_name: img_batch})[0]

    return output, img_batch


def compare_outputs(pytorch_out: np.ndarray, onnx_out: np.ndarray, tolerance: float = 1e-3):
    """Compare PyTorch and ONNX outputs."""
    print(f"\n{'='*70}")
    print("OUTPUT COMPARISON")
    print(f"{'='*70}")

    # Shape comparison
    print(f"PyTorch output shape: {pytorch_out.shape}")
    print(f"ONNX output shape: {onnx_out.shape}")

    if pytorch_out.shape != onnx_out.shape:
        print("ERROR: Output shapes do not match!")
        return False

    # Value comparison
    print(f"\nPyTorch output range: [{pytorch_out.min():.6f}, {pytorch_out.max():.6f}]")
    print(f"ONNX output range: [{onnx_out.min():.6f}, {onnx_out.max():.6f}]")

    # Calculate differences
    abs_diff = np.abs(pytorch_out - onnx_out)
    rel_diff = abs_diff / (np.abs(pytorch_out) + 1e-8)

    print(f"\n{'='*70}")
    print("DIFFERENCE METRICS")
    print(f"{'='*70}")
    print(f"Max absolute difference: {abs_diff.max():.6f}")
    print(f"Mean absolute difference: {abs_diff.mean():.6f}")
    print(f"Max relative difference: {rel_diff.max():.6f}")
    print(f"Mean relative difference: {rel_diff.mean():.6f}")

    # Tolerance check
    max_diff = abs_diff.max()
    print(f"\nTolerance threshold: {tolerance}")

    if max_diff < tolerance:
        print(f"✓ PASS: Outputs match within tolerance ({max_diff:.6f} < {tolerance})")
        status = True
    else:
        print(f"✗ FAIL: Outputs differ beyond tolerance ({max_diff:.6f} >= {tolerance})")
        status = False

    # Percentage of pixels within tolerance
    within_tolerance = np.sum(abs_diff < tolerance)
    total_pixels = abs_diff.size
    pct = (within_tolerance / total_pixels) * 100
    print(f"Pixels within tolerance: {within_tolerance}/{total_pixels} ({pct:.2f}%)")

    return status


def save_comparison_images(pytorch_out: np.ndarray, onnx_out: np.ndarray, output_dir: str):
    """Save side-by-side comparison images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def to_image(tensor):
        """Convert tensor to PIL Image."""
        output = np.clip(tensor[0], 0, 1)  # Remove batch, clip
        output = np.transpose(output, (1, 2, 0))  # CHW to HWC
        output = (output * 255).astype(np.uint8)
        return Image.fromarray(output)

    # Save individual outputs
    pytorch_img = to_image(pytorch_out)
    onnx_img = to_image(onnx_out)

    pytorch_img.save(output_dir / "pytorch_output.png")
    onnx_img.save(output_dir / "onnx_output.png")

    # Save difference map
    diff = np.abs(pytorch_out - onnx_out)[0]  # Remove batch
    diff = np.transpose(diff, (1, 2, 0))  # CHW to HWC

    # Amplify differences for visualization
    diff_amplified = np.clip(diff * 10, 0, 1)
    diff_img = (diff_amplified * 255).astype(np.uint8)

    # Convert to RGB if grayscale
    if diff_img.shape[2] == 1:
        diff_img = np.repeat(diff_img, 3, axis=2)

    Image.fromarray(diff_img).save(output_dir / "difference_map.png")

    print(f"\nComparison images saved to: {output_dir}")
    print(f"  - pytorch_output.png")
    print(f"  - onnx_output.png")
    print(f"  - difference_map.png (10x amplified)")


def main():
    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX model outputs")

    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint"
    )
    parser.add_argument(
        "--onnx", "-x",
        type=str,
        required=True,
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to test image"
    )
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=512,
        help="Image size (default: 512)"
    )
    parser.add_argument(
        "--tolerance", "-t",
        type=float,
        default=1e-3,
        help="Tolerance for comparison (default: 0.001)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="comparison_outputs",
        help="Directory to save comparison images (default: comparison_outputs)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for PyTorch model (default: cpu)"
    )

    args = parser.parse_args()

    # Check inputs exist
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    if not Path(args.onnx).exists():
        print(f"Error: ONNX model not found: {args.onnx}")
        return
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return

    print(f"{'='*70}")
    print("PYTORCH vs ONNX COMPARISON")
    print(f"{'='*70}")
    print(f"PyTorch checkpoint: {args.checkpoint}")
    print(f"ONNX model: {args.onnx}")
    print(f"Test image: {args.image}")
    print(f"Image size: {args.size}x{args.size}")

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load models
    print("\nLoading PyTorch model...")
    pytorch_model = load_pytorch_model(args.checkpoint, device)

    print("Loading ONNX model...")
    onnx_session = ort.InferenceSession(args.onnx)
    input_name = onnx_session.get_inputs()[0].name

    # Process with PyTorch
    print("\nRunning PyTorch inference...")
    pytorch_out, pytorch_input = process_with_pytorch(
        pytorch_model, args.image, args.size, device
    )

    # Process with ONNX
    print("Running ONNX inference...")
    onnx_out, onnx_input = process_with_onnx(
        onnx_session, input_name, args.image, args.size
    )

    # Compare inputs (should be identical)
    input_diff = np.abs(pytorch_input - onnx_input).max()
    print(f"\nInput difference: {input_diff:.10f}")
    if input_diff > 1e-6:
        print("WARNING: Inputs differ! This may affect comparison.")

    # Compare outputs
    passed = compare_outputs(pytorch_out, onnx_out, args.tolerance)

    # Save comparison images
    save_comparison_images(pytorch_out, onnx_out, args.output_dir)

    # Final result
    print(f"\n{'='*70}")
    if passed:
        print("✓ VERIFICATION PASSED")
        print("PyTorch and ONNX models produce matching outputs.")
    else:
        print("✗ VERIFICATION FAILED")
        print("PyTorch and ONNX models produce different outputs.")
        print("Check the difference map and consider re-exporting the ONNX model.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

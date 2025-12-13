#!/usr/bin/env python3
"""
Simple inference script for image enhancement.

Usage:
    python scripts/inference.py \
        --image path/to/input.jpg \
        --checkpoint checkpoints/best_model.pt \
        --output output.jpg

    # Or just view without saving:
    python scripts/inference.py \
        --image path/to/input.jpg \
        --checkpoint checkpoints/best_model.pt
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import get_model


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = checkpoint.get("model_config", {})
    model_name = model_config.get("name", "unet_small")

    # Map class names to registry keys
    name_mapping = {
        "UNetSmall": "unet_small",
        "UNet": "unet",
        "UNetLarge": "unet_large",
    }
    model_name = name_mapping.get(model_name, model_name)

    model = get_model(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def process_image(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    image_size: int = 512,
) -> tuple[Image.Image, Image.Image]:
    """
    Process a single image through the model.

    Returns:
        (input_image, output_image) as PIL Images
    """
    # Load and preprocess
    input_img = Image.open(image_path).convert("RGB")
    original_size = input_img.size  # (W, H)

    # Resize for model
    img_resized = TF.resize(input_img, (image_size, image_size))
    img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output_tensor = model(img_tensor)

    # Convert back to PIL
    output_tensor = output_tensor.squeeze(0).cpu().clamp(0, 1)
    output_img = TF.to_pil_image(output_tensor)

    # Resize back to original size
    output_img = output_img.resize(original_size, Image.LANCZOS)

    return input_img, output_img


def show_comparison(input_img: Image.Image, output_img: Image.Image, save_path: str = None):
    """Display side-by-side comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(input_img)
    axes[0].set_title("Input (Original)", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(output_img)
    axes[1].set_title("Output (Enhanced)", fontsize=14)
    axes[1].axis("off")

    plt.tight_layout()

    if save_path:
        # Save just the output image
        output_img.save(save_path, quality=95)
        print(f"Saved output to: {save_path}")

        # Also save comparison
        comparison_path = save_path.replace(".", "_comparison.")
        plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison to: {comparison_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--image", "-i", type=str, required=True, help="Input image path")
    parser.add_argument("--checkpoint", "-c", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output image path (optional)")
    parser.add_argument("--size", "-s", type=int, default=512, help="Processing size (default: 512)")
    parser.add_argument("--no-show", action="store_true", help="Don't display the result")
    args = parser.parse_args()

    # Check inputs exist
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    # Process image
    print(f"Processing: {args.image}")
    input_img, output_img = process_image(args.image, model, device, args.size)

    # Show/save results
    if args.no_show and args.output:
        output_img.save(args.output, quality=95)
        print(f"Saved output to: {args.output}")
    else:
        show_comparison(input_img, output_img, args.output)

    print("Done!")


if __name__ == "__main__":
    main()

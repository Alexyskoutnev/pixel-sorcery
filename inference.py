#!/usr/bin/env python3
"""
Inference script for running trained models on images.

Usage:
    # Single image
    python inference.py \
        --checkpoint checkpoints/best_model.pt \
        --input image.jpg \
        --output output.jpg

    # Directory of images
    python inference.py \
        --checkpoint checkpoints/best_model.pt \
        --input_dir input_images/ \
        --output_dir output_images/

    # With specific model (if not saved in checkpoint)
    python inference.py \
        --checkpoint checkpoints/best_model.pt \
        --model unet \
        --input image.jpg
"""

import argparse
from pathlib import Path
from typing import Optional
import time

import torch
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

from src.models import get_model, ResidualWrapper
from src.utils.logging import logger


def load_model(
    checkpoint_path: str,
    model_name: Optional[str] = None,
    device: str = "auto",
) -> tuple[torch.nn.Module, torch.device]:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model_name: Model name (if not in checkpoint)
        device: Device to load model on

    Returns:
        (model, device)
    """
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    logger.info(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config from checkpoint
    model_config = checkpoint.get("model_config", {})
    saved_model_name = model_config.get("name", "").lower()

    # Determine if residual wrapper was used
    is_residual = "residual" in saved_model_name.lower()
    if is_residual:
        saved_model_name = saved_model_name.replace("residual", "").strip()

    # Use provided model name or infer from checkpoint
    if model_name:
        actual_model_name = model_name
    elif saved_model_name:
        actual_model_name = saved_model_name
    else:
        actual_model_name = "unet"

    logger.info(f"Loading model: {actual_model_name}")

    # Create model
    model_kwargs = {}
    if "base_channels" in model_config:
        model_kwargs["base_channels"] = model_config["base_channels"]
    if "depth" in model_config:
        model_kwargs["depth"] = model_config["depth"]

    model = get_model(actual_model_name, **model_kwargs)

    if is_residual:
        model = ResidualWrapper(model)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded: {model.get_num_params():,} parameters")

    return model, device


def process_image(
    model: torch.nn.Module,
    image_path: str,
    output_path: str,
    device: torch.device,
    image_size: Optional[int] = None,
) -> float:
    """
    Process a single image.

    Args:
        model: Trained model
        image_path: Path to input image
        output_path: Path to save output
        device: Device to run on
        image_size: Resize to this size (None = keep original)

    Returns:
        Inference time in seconds
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # (width, height)

    # Resize if specified
    if image_size:
        img = TF.resize(img, (image_size, image_size))

    # Convert to tensor
    input_tensor = TF.to_tensor(img).unsqueeze(0).to(device)

    # Run inference
    start_time = time.time()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    inference_time = time.time() - start_time

    # Convert back to image
    output_tensor = output_tensor.squeeze(0).cpu()
    output_tensor = torch.clamp(output_tensor, 0, 1)
    output_img = TF.to_pil_image(output_tensor)

    # Resize back to original if we resized
    if image_size and output_img.size != original_size:
        output_img = output_img.resize(original_size, Image.BILINEAR)

    # Save
    output_img.save(output_path, quality=95)

    return inference_time


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained model")

    # Model
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (if not in checkpoint)",
    )

    # Input/Output
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to single input image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for output image",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory of input images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for output images",
    )

    # Processing
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Process at this size (None = original size)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.input and args.input_dir:
        raise ValueError("Specify either --input or --input_dir, not both")
    if not args.input and not args.input_dir:
        raise ValueError("Must specify --input or --input_dir")

    # Load model
    model, device = load_model(
        args.checkpoint,
        model_name=args.model,
        device=args.device,
    )

    # Process single image
    if args.input:
        input_path = Path(args.input)
        output_path = args.output or f"{input_path.stem}_enhanced{input_path.suffix}"

        logger.info(f"Processing: {input_path}")
        inference_time = process_image(
            model, str(input_path), output_path, device, args.image_size
        )
        logger.info(f"Saved: {output_path}")
        logger.info(f"Inference time: {inference_time*1000:.1f}ms")

    # Process directory
    else:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir or f"{input_dir}_enhanced")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        images = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]

        logger.info(f"Processing {len(images)} images...")
        total_time = 0

        for img_path in tqdm(images):
            output_path = output_dir / img_path.name
            inference_time = process_image(
                model, str(img_path), str(output_path), device, args.image_size
            )
            total_time += inference_time

        logger.info(f"Processed {len(images)} images")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Average: {total_time/len(images)*1000:.1f}ms per image")
        logger.info(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()

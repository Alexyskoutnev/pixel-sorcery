#!/usr/bin/env python3
"""
Test ONNX model on real images.

Usage:
    # Single image
    python scripts/test_onnx.py \
        --model bin/model_gan_512.onnx \
        --input test_image.jpg \
        --output test_onnx_output.jpg

    # Batch processing
    python scripts/test_onnx.py \
        --model bin/model_gan_512.onnx \
        --input_dir test_images/ \
        --output_dir onnx_outputs/
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm


def preprocess_image(image_path: str, target_size: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Load and preprocess image for ONNX model.

    Args:
        image_path: Path to input image
        target_size: (height, width) to resize to

    Returns:
        (preprocessed_tensor, original_size)
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # (width, height)

    # Resize to model input size
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)

    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0

    # Convert from HWC to CHW (channels first)
    img_tensor = np.transpose(img_array, (2, 0, 1))

    # Add batch dimension: [1, C, H, W]
    img_batch = np.expand_dims(img_tensor, axis=0)

    return img_batch, original_size


def postprocess_output(output_tensor: np.ndarray, target_size: tuple[int, int]) -> Image.Image:
    """
    Convert model output back to PIL Image.

    Args:
        output_tensor: Model output [1, C, H, W]
        target_size: (width, height) to resize to

    Returns:
        PIL Image
    """
    # Remove batch dimension
    output = output_tensor[0]

    # Clip values to [0, 1]
    output = np.clip(output, 0, 1)

    # Convert from CHW to HWC
    output = np.transpose(output, (1, 2, 0))

    # Convert to uint8
    output = (output * 255).astype(np.uint8)

    # Convert to PIL Image
    output_img = Image.fromarray(output)

    # Resize to target size if needed
    if output_img.size != target_size:
        output_img = output_img.resize(target_size, Image.BILINEAR)

    return output_img


def process_single_image(
    sess: ort.InferenceSession,
    input_name: str,
    image_path: str,
    output_path: str,
    model_size: tuple[int, int],
    keep_original_size: bool = True,
) -> float:
    """
    Process a single image through ONNX model.

    Args:
        sess: ONNX Runtime session
        input_name: Input tensor name
        image_path: Path to input image
        output_path: Path to save output
        model_size: (height, width) model expects
        keep_original_size: Resize output to match input

    Returns:
        Inference time in seconds
    """
    # Preprocess
    input_tensor, original_size = preprocess_image(image_path, model_size)

    # Run inference
    start_time = time.time()
    output = sess.run(None, {input_name: input_tensor})[0]
    inference_time = time.time() - start_time

    # Postprocess
    target_size = original_size if keep_original_size else (model_size[1], model_size[0])
    output_img = postprocess_output(output, target_size)

    # Save
    output_img.save(output_path, quality=95)

    return inference_time


def main():
    parser = argparse.ArgumentParser(description="Test ONNX model on images")

    # Model
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to ONNX model"
    )

    # Input/Output
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to input image"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save output image"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory of input images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for output images"
    )

    # Options
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=None,
        help="Model input size (default: auto-detect from model)"
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Don't resize output back to original size"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.input and args.input_dir:
        raise ValueError("Specify either --input or --input_dir, not both")
    if not args.input and not args.input_dir:
        raise ValueError("Must specify --input or --input_dir")

    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        return

    # Load ONNX model
    print(f"Loading model: {args.model}")
    sess = ort.InferenceSession(args.model)

    # Get input info
    input_info = sess.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape

    # Determine model size
    if args.size:
        model_size = (args.size, args.size)
    else:
        # Auto-detect from model shape [batch, channels, height, width]
        if len(input_shape) >= 4:
            h = input_shape[2] if isinstance(input_shape[2], int) else 512
            w = input_shape[3] if isinstance(input_shape[3], int) else 512
            model_size = (h, w)
        else:
            model_size = (512, 512)

    print(f"Model input size: {model_size[0]}x{model_size[1]}")

    # Process single image
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input image not found: {args.input}")
            return

        output_path = args.output or f"{input_path.stem}_onnx{input_path.suffix}"

        print(f"Processing: {input_path}")
        inference_time = process_single_image(
            sess,
            input_name,
            str(input_path),
            output_path,
            model_size,
            keep_original_size=not args.no_resize,
        )

        print(f"Saved: {output_path}")
        print(f"Inference time: {inference_time*1000:.1f}ms")

    # Process directory
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory not found: {args.input_dir}")
            return

        output_dir = Path(args.output_dir or f"{input_dir}_onnx")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        images = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]

        if not images:
            print(f"No images found in {input_dir}")
            return

        print(f"Processing {len(images)} images...")
        total_time = 0

        for img_path in tqdm(images, desc="Processing"):
            output_path = output_dir / img_path.name
            inference_time = process_single_image(
                sess,
                input_name,
                str(img_path),
                str(output_path),
                model_size,
                keep_original_size=not args.no_resize,
            )
            total_time += inference_time

        print(f"\nProcessed {len(images)} images")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average: {total_time/len(images)*1000:.1f}ms per image")
        print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()

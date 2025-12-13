#!/usr/bin/env python3
"""
05_inference.py - Run inference with trained NAFNet model

Supports:
- Single image inference
- Automatic tiled processing for large images (3300x2200+)
- Multiple output formats

Usage:
    # Single image
    python scripts/05_inference.py --input photo.png --output enhanced.png
    
    # Specify model path
    python scripts/05_inference.py --input photo.png --output enhanced.png \
        --model experiments/NAFNet_RealEstate_Fast/models/net_g_30000.pth
    
    # Process large image with custom tile size
    python scripts/05_inference.py --input large_photo.png --output enhanced.png \
        --tile-size 512 --tile-pad 32
"""

import argparse
import os
import sys
import time
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_nafnet_model(model_path: str, width: int = 32, device: str = 'cuda'):
    """Load trained NAFNet model."""
    
    # Add NAFNet to path
    nafnet_path = Path(__file__).parent.parent / "NAFNet"
    if str(nafnet_path) not in sys.path:
        sys.path.insert(0, str(nafnet_path))
    
    from basicsr.archs.NAFNet_arch import NAFNet
    
    # Create model with same architecture as training
    model = NAFNet(
        img_channel=3,
        width=width,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2]
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'])
    elif 'params_ema' in checkpoint:
        model.load_state_dict(checkpoint['params_ema'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    
    return model


def process_image_simple(model, img: np.ndarray, device: str = 'cuda') -> np.ndarray:
    """Process a single image without tiling (for smaller images)."""
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
    
    # Convert back to numpy
    output = output.squeeze().cpu().numpy()
    output = np.clip(output * 255, 0, 255).astype(np.uint8)
    output = output.transpose(1, 2, 0)
    
    return output


def process_image_tiled(
    model, 
    img: np.ndarray, 
    tile_size: int = 512, 
    tile_pad: int = 32,
    device: str = 'cuda'
) -> np.ndarray:
    """Process a large image using tiles with overlap blending."""
    
    h, w, c = img.shape
    
    # Pad image to be divisible by tile_size
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    
    img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    h_pad, w_pad = img_padded.shape[:2]
    
    # Calculate number of tiles
    tiles_x = math.ceil(w_pad / tile_size)
    tiles_y = math.ceil(h_pad / tile_size)
    
    # Output image
    output = np.zeros_like(img_padded, dtype=np.float32)
    weight_map = np.zeros((h_pad, w_pad, 1), dtype=np.float32)
    
    # Create blending weight (cosine falloff at edges)
    def create_weight(size, pad):
        weight = np.ones(size, dtype=np.float32)
        if pad > 0:
            # Cosine ramp for smooth blending
            ramp = np.linspace(0, np.pi/2, pad)
            weight[:pad] = np.sin(ramp) ** 2
            weight[-pad:] = np.cos(ramp) ** 2
        return weight
    
    total_tiles = tiles_x * tiles_y
    pbar = tqdm(total=total_tiles, desc="Processing tiles")
    
    for y in range(tiles_y):
        for x in range(tiles_x):
            # Calculate tile coordinates with padding
            x1 = max(x * tile_size - tile_pad, 0)
            x2 = min((x + 1) * tile_size + tile_pad, w_pad)
            y1 = max(y * tile_size - tile_pad, 0)
            y2 = min((y + 1) * tile_size + tile_pad, h_pad)
            
            # Extract tile
            tile = img_padded[y1:y2, x1:x2]
            
            # Process tile
            tile_output = process_image_simple(model, tile, device)
            
            # Calculate output position (without padding)
            out_x1 = x * tile_size
            out_x2 = min((x + 1) * tile_size, w_pad)
            out_y1 = y * tile_size
            out_y2 = min((y + 1) * tile_size, h_pad)
            
            # Calculate tile crop (remove padding)
            tile_x1 = out_x1 - x1
            tile_x2 = tile_x1 + (out_x2 - out_x1)
            tile_y1 = out_y1 - y1
            tile_y2 = tile_y1 + (out_y2 - out_y1)
            
            # Create weight for this tile
            tile_h, tile_w = out_y2 - out_y1, out_x2 - out_x1
            weight_y = create_weight(tile_h, min(tile_pad, tile_h // 4))
            weight_x = create_weight(tile_w, min(tile_pad, tile_w // 4))
            tile_weight = np.outer(weight_y, weight_x)[:, :, np.newaxis]
            
            # Blend into output
            output[out_y1:out_y2, out_x1:out_x2] += tile_output[tile_y1:tile_y2, tile_x1:tile_x2].astype(np.float32) * tile_weight
            weight_map[out_y1:out_y2, out_x1:out_x2] += tile_weight
            
            pbar.update(1)
    
    pbar.close()
    
    # Normalize by weights
    output = output / (weight_map + 1e-8)
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    # Remove padding
    output = output[:h, :w]
    
    return output


def enhance_image(
    model,
    input_path: str,
    output_path: str,
    tile_size: int = 512,
    tile_pad: int = 32,
    device: str = 'cuda'
):
    """Main function to enhance a single image."""
    
    # Load image
    print(f"Loading: {input_path}")
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError(f"Could not load image: {input_path}")
    
    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Decide whether to use tiling
    start_time = time.time()
    
    if h * w > 1024 * 1024:  # More than 1 megapixel
        print(f"Using tiled processing (tile_size={tile_size}, overlap={tile_pad})")
        output = process_image_tiled(model, img, tile_size, tile_pad, device)
    else:
        print("Processing full image...")
        output = process_image_simple(model, img, device)
    
    elapsed = time.time() - start_time
    print(f"Processing time: {elapsed:.2f} seconds")
    
    # Save output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(output_path, output)
    print(f"Saved: {output_path}")
    
    return output


def find_latest_model(experiments_dir: str = "experiments") -> str:
    """Find the most recent trained model."""
    
    exp_path = Path(experiments_dir)
    
    # Look for NAFNet experiment folders
    for exp_name in ["NAFNet_RealEstate_Fast", "NAFNet_RealEstate_Enhancement"]:
        model_dir = exp_path / exp_name / "models"
        if model_dir.exists():
            # Find latest model
            models = list(model_dir.glob("net_g_*.pth"))
            if models:
                # Sort by iteration number
                models.sort(key=lambda x: int(x.stem.split('_')[-1]) if x.stem.split('_')[-1].isdigit() else 0)
                return str(models[-1])
    
    # Check for net_g_latest.pth
    for exp_name in ["NAFNet_RealEstate_Fast", "NAFNet_RealEstate_Enhancement"]:
        latest = exp_path / exp_name / "models" / "net_g_latest.pth"
        if latest.exists():
            return str(latest)
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Enhance real estate photos with trained NAFNet model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (auto-finds latest model)
    python 05_inference.py --input photo.png --output enhanced.png
    
    # Specify model
    python 05_inference.py --input photo.png --output enhanced.png \\
        --model experiments/NAFNet_RealEstate_Fast/models/net_g_30000.pth
    
    # Process large image with custom tile settings
    python 05_inference.py --input 4k_photo.png --output enhanced.png \\
        --tile-size 768 --tile-pad 64
        """
    )
    
    parser.add_argument('--input', '-i', required=True, help='Input image path')
    parser.add_argument('--output', '-o', required=True, help='Output image path')
    parser.add_argument('--model', '-m', help='Model checkpoint path (auto-detected if not specified)')
    parser.add_argument('--width', type=int, default=32, help='NAFNet width (default: 32)')
    parser.add_argument('--tile-size', type=int, default=512, help='Tile size for large images (default: 512)')
    parser.add_argument('--tile-pad', type=int, default=32, help='Tile overlap padding (default: 32)')
    parser.add_argument('--device', default='cuda', help='Device: cuda or cpu')
    
    args = parser.parse_args()
    
    # Find model
    if args.model:
        model_path = args.model
    else:
        model_path = find_latest_model()
        if model_path is None:
            print("❌ No trained model found!")
            print("   Train a model first: ./scripts/04_train.sh")
            print("   Or specify a model: --model path/to/model.pth")
            sys.exit(1)
    
    print(f"Using model: {model_path}")
    
    # Check input exists
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model
    print("Loading model...")
    model = load_nafnet_model(model_path, width=args.width, device=args.device)
    print("Model loaded!")
    
    # Process image
    enhance_image(
        model=model,
        input_path=args.input,
        output_path=args.output,
        tile_size=args.tile_size,
        tile_pad=args.tile_pad,
        device=args.device
    )
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
06_batch_inference.py - Process multiple images with trained model

Usage:
    # Process all images in a folder
    python scripts/06_batch_inference.py \
        --input-dir /path/to/input_photos \
        --output-dir /path/to/enhanced_photos
    
    # With specific model and parallel processing
    python scripts/06_batch_inference.py \
        --input-dir ./input \
        --output-dir ./output \
        --model experiments/NAFNet_RealEstate_Fast/models/net_g_30000.pth \
        --workers 4
"""

import argparse
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import cv2
import torch
from tqdm import tqdm

# Import from inference script
sys.path.insert(0, str(Path(__file__).parent))
from inference import load_nafnet_model, enhance_image, find_latest_model


def get_image_files(input_dir: str) -> List[Path]:
    """Get all image files from directory."""
    
    extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    input_path = Path(input_dir)
    
    files = []
    for ext in extensions:
        files.extend(input_path.glob(f'*{ext}'))
        files.extend(input_path.glob(f'*{ext.upper()}'))
    
    return sorted(files)


def process_batch(
    model,
    input_files: List[Path],
    output_dir: str,
    tile_size: int = 512,
    tile_pad: int = 32,
    device: str = 'cuda',
    output_format: str = 'png'
) -> Tuple[int, int, List[str]]:
    """
    Process a batch of images.
    
    Returns:
        Tuple of (successful_count, failed_count, error_messages)
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    errors = []
    
    for input_file in tqdm(input_files, desc="Processing"):
        try:
            output_file = output_path / f"{input_file.stem}.{output_format}"
            
            enhance_image(
                model=model,
                input_path=str(input_file),
                output_path=str(output_file),
                tile_size=tile_size,
                tile_pad=tile_pad,
                device=device
            )
            
            successful += 1
            
        except Exception as e:
            failed += 1
            errors.append(f"{input_file.name}: {str(e)}")
    
    return successful, failed, errors


def main():
    parser = argparse.ArgumentParser(
        description="Batch process images with trained NAFNet model"
    )
    
    parser.add_argument('--input-dir', '-i', required=True, help='Input directory')
    parser.add_argument('--output-dir', '-o', required=True, help='Output directory')
    parser.add_argument('--model', '-m', help='Model path (auto-detected if not specified)')
    parser.add_argument('--width', type=int, default=32, help='NAFNet width (default: 32)')
    parser.add_argument('--tile-size', type=int, default=512, help='Tile size (default: 512)')
    parser.add_argument('--tile-pad', type=int, default=32, help='Tile padding (default: 32)')
    parser.add_argument('--format', default='png', choices=['png', 'jpg'], help='Output format')
    parser.add_argument('--device', default='cuda', help='Device: cuda or cpu')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  NAFNet Batch Image Enhancement")
    print("=" * 60)
    
    # Find model
    if args.model:
        model_path = args.model
    else:
        model_path = find_latest_model()
        if model_path is None:
            print("❌ No trained model found!")
            sys.exit(1)
    
    print(f"\nModel: {model_path}")
    
    # Get input files
    input_files = get_image_files(args.input_dir)
    print(f"Found {len(input_files)} images to process")
    
    if len(input_files) == 0:
        print("❌ No images found in input directory")
        sys.exit(1)
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model
    print("\nLoading model...")
    model = load_nafnet_model(model_path, width=args.width, device=args.device)
    
    # Process images
    print("\nProcessing images...")
    start_time = time.time()
    
    successful, failed, errors = process_batch(
        model=model,
        input_files=input_files,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        tile_pad=args.tile_pad,
        device=args.device,
        output_format=args.format
    )
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("  Batch Processing Complete!")
    print("=" * 60)
    print(f"""
    ✅ Successful: {successful}
    ❌ Failed: {failed}
    ⏱️  Total time: {elapsed:.1f} seconds
    📊 Average: {elapsed/len(input_files):.2f} seconds/image
    
    Output saved to: {args.output_dir}
    """)
    
    if errors:
        print("Errors:")
        for error in errors[:10]:
            print(f"  - {error}")


if __name__ == "__main__":
    main()

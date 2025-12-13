#!/usr/bin/env python3
"""
02_prepare_dataset.py - Prepare paired images for NAFNet training

This script:
1. Validates your before/after image pairs
2. Creates train/val split (90/10)
3. Organizes into BasicSR folder structure
4. Generates dataset statistics

Usage:
    python scripts/02_prepare_dataset.py \
        --before /path/to/before_photos \
        --after /path/to/after_photos \
        --output ./datasets/realestate
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np
from tqdm import tqdm


def find_paired_images(before_folder: str, after_folder: str) -> List[Tuple[Path, Path]]:
    """Find matching image pairs between before and after folders."""
    
    before_path = Path(before_folder)
    after_path = Path(after_folder)
    
    # Supported image extensions
    extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.tif', '.tiff'}
    
    # Get all images
    before_images = {f.stem: f for f in before_path.iterdir() if f.suffix in extensions}
    after_images = {f.stem: f for f in after_path.iterdir() if f.suffix in extensions}
    
    # Find pairs (matching filenames)
    common_names = set(before_images.keys()) & set(after_images.keys())
    
    pairs = [(before_images[name], after_images[name]) for name in sorted(common_names)]
    
    return pairs


def validate_pair(before_path: Path, after_path: Path) -> Tuple[bool, str]:
    """Validate a single image pair."""
    
    # Check files exist
    if not before_path.exists():
        return False, f"Before image not found: {before_path}"
    if not after_path.exists():
        return False, f"After image not found: {after_path}"
    
    # Load images
    before_img = cv2.imread(str(before_path))
    after_img = cv2.imread(str(after_path))
    
    if before_img is None:
        return False, f"Could not read before image: {before_path}"
    if after_img is None:
        return False, f"Could not read after image: {after_path}"
    
    # Check dimensions match
    if before_img.shape != after_img.shape:
        return False, f"Dimension mismatch: {before_img.shape} vs {after_img.shape}"
    
    # Check minimum size (at least 256x256)
    h, w = before_img.shape[:2]
    if h < 256 or w < 256:
        return False, f"Image too small: {w}x{h} (minimum 256x256)"
    
    return True, "OK"


def copy_and_rename(src: Path, dst_folder: Path, new_name: str) -> Path:
    """Copy image to destination with new name, converting to PNG."""
    
    dst_folder.mkdir(parents=True, exist_ok=True)
    dst_path = dst_folder / f"{new_name}.png"
    
    # Read and write as PNG (lossless)
    img = cv2.imread(str(src))
    cv2.imwrite(str(dst_path), img)
    
    return dst_path


def prepare_dataset(
    before_folder: str,
    after_folder: str,
    output_folder: str,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Dict:
    """
    Prepare the complete dataset for NAFNet training.
    
    Returns:
        Dictionary with statistics about the prepared dataset
    """
    
    print("=" * 60)
    print("  Dataset Preparation for NAFNet Training")
    print("=" * 60)
    
    # Find pairs
    print("\n[1/4] Finding paired images...")
    pairs = find_paired_images(before_folder, after_folder)
    print(f"      Found {len(pairs)} potential pairs")
    
    if len(pairs) == 0:
        print("\n❌ ERROR: No paired images found!")
        print("   Make sure before and after images have matching filenames")
        print(f"   Before folder: {before_folder}")
        print(f"   After folder: {after_folder}")
        sys.exit(1)
    
    # Validate pairs
    print("\n[2/4] Validating image pairs...")
    valid_pairs = []
    invalid_pairs = []
    
    for before, after in tqdm(pairs, desc="      Validating"):
        is_valid, msg = validate_pair(before, after)
        if is_valid:
            valid_pairs.append((before, after))
        else:
            invalid_pairs.append((before, after, msg))
    
    print(f"      Valid pairs: {len(valid_pairs)}")
    print(f"      Invalid pairs: {len(invalid_pairs)}")
    
    if len(invalid_pairs) > 0:
        print("\n      Invalid pairs (first 5):")
        for before, after, msg in invalid_pairs[:5]:
            print(f"        - {before.name}: {msg}")
    
    if len(valid_pairs) == 0:
        print("\n❌ ERROR: No valid image pairs!")
        sys.exit(1)
    
    # Create train/val split
    print("\n[3/4] Creating train/val split...")
    random.seed(seed)
    random.shuffle(valid_pairs)
    
    n_val = max(1, int(len(valid_pairs) * val_ratio))
    val_pairs = valid_pairs[:n_val]
    train_pairs = valid_pairs[n_val:]
    
    print(f"      Training set: {len(train_pairs)} pairs")
    print(f"      Validation set: {len(val_pairs)} pairs")
    
    # Copy files to output structure
    print("\n[4/4] Copying files to output folder...")
    output_path = Path(output_folder)
    
    # Create directories
    for split in ['train', 'val']:
        for folder in ['lq', 'gt']:
            (output_path / split / folder).mkdir(parents=True, exist_ok=True)
    
    # Copy training pairs
    print("      Copying training set...")
    for i, (before, after) in enumerate(tqdm(train_pairs, desc="      Train")):
        name = f"{i:04d}"
        copy_and_rename(before, output_path / "train" / "lq", name)
        copy_and_rename(after, output_path / "train" / "gt", name)
    
    # Copy validation pairs
    print("      Copying validation set...")
    for i, (before, after) in enumerate(tqdm(val_pairs, desc="      Val")):
        name = f"{i:04d}"
        copy_and_rename(before, output_path / "val" / "lq", name)
        copy_and_rename(after, output_path / "val" / "gt", name)
    
    # Calculate statistics
    sample_img = cv2.imread(str(train_pairs[0][0]))
    h, w = sample_img.shape[:2]
    
    stats = {
        'total_pairs': len(valid_pairs),
        'train_pairs': len(train_pairs),
        'val_pairs': len(val_pairs),
        'image_height': h,
        'image_width': w,
        'invalid_pairs': len(invalid_pairs),
        'output_folder': str(output_path.absolute())
    }
    
    # Save stats
    stats_file = output_path / "dataset_stats.txt"
    with open(stats_file, 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    # Print summary
    print("\n" + "=" * 60)
    print("  ✅ Dataset Preparation Complete!")
    print("=" * 60)
    print(f"""
    📊 Dataset Statistics:
       - Total valid pairs: {stats['total_pairs']}
       - Training pairs: {stats['train_pairs']}
       - Validation pairs: {stats['val_pairs']}
       - Image size: {stats['image_width']}x{stats['image_height']}
    
    📁 Output Location:
       {stats['output_folder']}
       ├── train/
       │   ├── lq/  ({stats['train_pairs']} images)
       │   └── gt/  ({stats['train_pairs']} images)
       └── val/
           ├── lq/  ({stats['val_pairs']} images)
           └── gt/  ({stats['val_pairs']} images)
    
    ➡️  Next Step: python scripts/03_verify_setup.py
    """)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare paired images for NAFNet training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python 02_prepare_dataset.py --before ./raw/before --after ./raw/after --output ./datasets/realestate
    
    # Custom validation split
    python 02_prepare_dataset.py --before ./raw/before --after ./raw/after --output ./datasets/realestate --val-ratio 0.15
        """
    )
    
    parser.add_argument('--before', required=True, help='Folder containing before (input) images')
    parser.add_argument('--after', required=True, help='Folder containing after (ground truth) images')
    parser.add_argument('--output', required=True, help='Output folder for prepared dataset')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.isdir(args.before):
        print(f"❌ ERROR: Before folder not found: {args.before}")
        sys.exit(1)
    
    if not os.path.isdir(args.after):
        print(f"❌ ERROR: After folder not found: {args.after}")
        sys.exit(1)
    
    # Run preparation
    prepare_dataset(
        before_folder=args.before,
        after_folder=args.after,
        output_folder=args.output,
        val_ratio=args.val_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

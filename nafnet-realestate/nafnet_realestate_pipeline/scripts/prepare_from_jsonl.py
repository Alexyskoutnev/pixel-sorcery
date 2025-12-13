#!/usr/bin/env python3
"""
prepare_from_jsonl.py - Prepare dataset from JSONL format

Reads train.jsonl with {"src": "...", "tar": "..."} entries and creates
the BasicSR folder structure for NAFNet training.

Usage:
    python prepare_from_jsonl.py \
        --jsonl /path/to/train.jsonl \
        --images /path/to/images \
        --output ./datasets/realestate
"""

import os
import sys
import json
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm


def prepare_dataset(jsonl_path: str, images_root: str, output_folder: str,
                    val_ratio: float = 0.1, seed: int = 42):
    """Prepare dataset from JSONL format."""

    print("=" * 60)
    print("  Dataset Preparation from JSONL")
    print("=" * 60)

    jsonl_path = Path(jsonl_path)
    images_root = Path(images_root)
    output_path = Path(output_folder)

    # Read JSONL
    print(f"\n[1/4] Reading {jsonl_path}...")
    pairs = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            src_path = images_root.parent / entry['src']  # JSONL paths are relative to parent
            tar_path = images_root.parent / entry['tar']
            pairs.append((src_path, tar_path))

    print(f"      Found {len(pairs)} pairs in JSONL")

    # Validate pairs exist
    print("\n[2/4] Validating image pairs...")
    valid_pairs = []
    missing = []

    for src, tar in tqdm(pairs, desc="      Checking"):
        if src.exists() and tar.exists():
            valid_pairs.append((src, tar))
        else:
            missing.append((src, tar))

    print(f"      Valid pairs: {len(valid_pairs)}")
    if missing:
        print(f"      Missing pairs: {len(missing)}")
        for src, tar in missing[:3]:
            print(f"        - {src.name} / {tar.name}")

    if not valid_pairs:
        print("\n ERROR: No valid pairs found!")
        sys.exit(1)

    # Create train/val split
    print("\n[3/4] Creating train/val split...")
    random.seed(seed)
    shuffled = valid_pairs.copy()
    random.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_ratio))
    val_pairs = shuffled[:n_val]
    train_pairs = shuffled[n_val:]

    print(f"      Training: {len(train_pairs)} pairs")
    print(f"      Validation: {len(val_pairs)} pairs")

    # Create output directories
    for split in ['train', 'val']:
        for folder in ['lq', 'gt']:
            (output_path / split / folder).mkdir(parents=True, exist_ok=True)

    # Copy files
    print("\n[4/4] Copying files...")

    print("      Copying training set...")
    for i, (src, tar) in enumerate(tqdm(train_pairs, desc="      Train")):
        name = f"{i:04d}.jpg"
        shutil.copy2(src, output_path / "train" / "lq" / name)
        shutil.copy2(tar, output_path / "train" / "gt" / name)

    print("      Copying validation set...")
    for i, (src, tar) in enumerate(tqdm(val_pairs, desc="      Val")):
        name = f"{i:04d}.jpg"
        shutil.copy2(src, output_path / "val" / "lq" / name)
        shutil.copy2(tar, output_path / "val" / "gt" / name)

    # Print summary
    print("\n" + "=" * 60)
    print("  Dataset Preparation Complete!")
    print("=" * 60)
    print(f"""
    Dataset Statistics:
       - Total pairs: {len(valid_pairs)}
       - Training: {len(train_pairs)} pairs
       - Validation: {len(val_pairs)} pairs

    Output Location:
       {output_path.absolute()}
       |-- train/
       |   |-- lq/  ({len(train_pairs)} images - input/before)
       |   +-- gt/  ({len(train_pairs)} images - ground truth/after)
       +-- val/
           |-- lq/  ({len(val_pairs)} images)
           +-- gt/  ({len(val_pairs)} images)

    Next Step: Run training with BasicSR
    """)


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset from JSONL")
    parser.add_argument('--jsonl', required=True, help='Path to train.jsonl')
    parser.add_argument('--images', required=True, help='Path to images folder')
    parser.add_argument('--output', required=True, help='Output folder')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    prepare_dataset(
        jsonl_path=args.jsonl,
        images_root=args.images,
        output_folder=args.output,
        val_ratio=args.val_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

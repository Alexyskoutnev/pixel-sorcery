#!/usr/bin/env python3
"""
NAFNet Real Estate Photo Enhancement - Inference Script
========================================================
Enhance photos using the trained NAFNet model.

Usage:
    # Single image
    python inference.py --input photo.jpg --output enhanced.jpg

    # Folder of images
    python inference.py --input ./input_folder --output ./output_folder

    # With specific model checkpoint
    python inference.py --input photo.jpg --output enhanced.jpg --model path/to/model.pth
"""

import argparse
import os
import sys
import glob
import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add BasicSR to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'BasicSR'))

from basicsr.archs.NAFNet_arch import NAFNet


def load_model(model_path, device):
    """Load the trained NAFNet model."""
    # NAFNet architecture matching the training config (width=32)
    model = NAFNet(
        img_channel=3,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2]
    )

    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # Handle different checkpoint formats
    if 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'])
    elif 'params_ema' in checkpoint:
        model.load_state_dict(checkpoint['params_ema'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def process_image(model, img_path, device):
    """Process a single image through the model."""
    # Read image (BGR)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    # Store original size
    h, w = img.shape[:2]

    # Pad to multiple of 8 (required by NAFNet architecture)
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    # Convert to tensor: BGR -> RGB, HWC -> CHW, normalize to [0,1]
    img_tensor = img[:, :, ::-1].copy()  # BGR to RGB
    img_tensor = img_tensor.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)

    # Convert back to image: CHW -> HWC, RGB -> BGR, denormalize
    output = output.squeeze(0).cpu().clamp(0, 1).numpy()
    output = (output.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    output = output[:, :, ::-1]  # RGB to BGR

    # Remove padding
    if pad_h > 0 or pad_w > 0:
        output = output[:h, :w]

    return output


def main():
    parser = argparse.ArgumentParser(description='NAFNet Real Estate Photo Enhancement')
    parser.add_argument('--input', '-i', required=True,
                        help='Input image or folder path')
    parser.add_argument('--output', '-o', required=True,
                        help='Output image or folder path')
    parser.add_argument('--model', '-m',
                        default='BasicSR/experiments/NAFNet_RealEstate_Fast/models/net_g_12000.pth',
                        help='Path to trained model weights')
    parser.add_argument('--device', '-d', default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run inference on')
    args = parser.parse_args()

    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    device = torch.device(args.device)

    # Load model
    print(f"Loading model from: {args.model}")
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)
    model = load_model(args.model, device)
    print(f"Model loaded successfully on {device}")

    # Get input/output paths
    if os.path.isfile(args.input):
        # Single image mode
        input_paths = [args.input]
        if os.path.isdir(args.output):
            output_paths = [os.path.join(args.output, os.path.basename(args.input))]
        else:
            output_paths = [args.output]
    else:
        # Folder mode
        input_paths = sorted(glob.glob(os.path.join(args.input, '*.[jJpP][pPnN][gG]')))
        input_paths += sorted(glob.glob(os.path.join(args.input, '*.[jJ][pP][eE][gG]')))
        if not input_paths:
            print(f"No images found in {args.input}")
            sys.exit(1)
        os.makedirs(args.output, exist_ok=True)
        output_paths = [os.path.join(args.output, os.path.basename(p)) for p in input_paths]

    # Process images
    print(f"\nProcessing {len(input_paths)} image(s)...")
    for inp_path, out_path in tqdm(zip(input_paths, output_paths), total=len(input_paths)):
        try:
            result = process_image(model, inp_path, device)
            cv2.imwrite(out_path, result)
        except Exception as e:
            print(f"\nError processing {inp_path}: {e}")
            continue

    print(f"\nDone! Results saved to: {args.output}")


if __name__ == '__main__':
    main()

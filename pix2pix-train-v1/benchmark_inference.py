#!/usr/bin/env python3
"""
NAFNet Inference Benchmark - Precise RAM and Compute Tracking
==============================================================
Measures exact memory usage and processing time.
"""

import argparse
import os
import sys
import glob
import gc
import time
import cv2
import numpy as np
import psutil

# Add BasicSR to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'BasicSR'))

import torch
from basicsr.archs.NAFNet_arch import NAFNet


def get_process_memory_mb():
    """Get current process RAM usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0


def get_gpu_memory_reserved_mb():
    """Get reserved GPU memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / (1024 * 1024)
    return 0


def load_model(model_path, device):
    """Load the trained NAFNet model."""
    model = NAFNet(
        img_channel=3,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2]
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

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
    """Process a single image and return output + metrics."""
    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    h, w = img.shape[:2]
    megapixels = (h * w) / 1_000_000

    # Pad to multiple of 8
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    # Convert to tensor
    img_tensor = img[:, :, ::-1].copy()
    img_tensor = img_tensor.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    # Measure inference
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        output = model(img_tensor)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    inference_time = time.perf_counter() - start_time

    # Convert back to image
    output = output.squeeze(0).cpu().clamp(0, 1).numpy()
    output = (output.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    output = output[:, :, ::-1]

    if pad_h > 0 or pad_w > 0:
        output = output[:h, :w]

    return output, inference_time, (w, h), megapixels


def main():
    parser = argparse.ArgumentParser(description='NAFNet Benchmark')
    parser.add_argument('--input', '-i', required=True, help='Input folder')
    parser.add_argument('--output', '-o', required=True, help='Output folder')
    parser.add_argument('--model', '-m',
                        default='BasicSR/experiments/NAFNet_RealEstate_Fast/models/net_g_12000.pth')
    parser.add_argument('--device', '-d', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)

    # Force garbage collection before starting
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("=" * 60)
    print("NAFNet INFERENCE BENCHMARK")
    print("=" * 60)

    # Baseline memory
    baseline_ram = get_process_memory_mb()
    baseline_gpu = get_gpu_memory_mb()
    print(f"\n[BASELINE MEMORY]")
    print(f"  Process RAM: {baseline_ram:.1f} MB")
    print(f"  GPU Memory:  {baseline_gpu:.1f} MB")

    # Load model
    print(f"\n[LOADING MODEL]")
    print(f"  Path: {args.model}")

    model_load_start = time.perf_counter()
    model = load_model(args.model, device)
    model_load_time = time.perf_counter() - model_load_start

    after_model_ram = get_process_memory_mb()
    after_model_gpu = get_gpu_memory_mb()

    model_ram_usage = after_model_ram - baseline_ram
    model_gpu_usage = after_model_gpu - baseline_gpu

    print(f"  Load time:   {model_load_time:.2f}s")
    print(f"  Model RAM:   {model_ram_usage:.1f} MB (process now: {after_model_ram:.1f} MB)")
    print(f"  Model GPU:   {model_gpu_usage:.1f} MB (allocated now: {after_model_gpu:.1f} MB)")

    # Get input files
    input_paths = sorted(glob.glob(os.path.join(args.input, '*.[jJpP][pPnN][gG]')))
    input_paths += sorted(glob.glob(os.path.join(args.input, '*.[jJ][pP][eE][gG]')))

    if not input_paths:
        print(f"No images found in {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print(f"\n[PROCESSING {len(input_paths)} IMAGES]")
    print("-" * 60)

    # Track metrics
    times = []
    peak_ram = after_model_ram
    peak_gpu = after_model_gpu
    total_megapixels = 0

    for i, inp_path in enumerate(input_paths):
        filename = os.path.basename(inp_path)
        out_path = os.path.join(args.output, filename)

        # Memory before this image
        pre_ram = get_process_memory_mb()
        pre_gpu = get_gpu_memory_mb()

        try:
            result, inf_time, dims, mp = process_image(model, inp_path, device)
            cv2.imwrite(out_path, result)

            # Memory after this image
            post_ram = get_process_memory_mb()
            post_gpu = get_gpu_memory_mb()

            # Track peaks
            peak_ram = max(peak_ram, post_ram)
            peak_gpu = max(peak_gpu, post_gpu)

            times.append(inf_time)
            total_megapixels += mp

            # Print every 10 images or first 5
            if i < 5 or (i + 1) % 10 == 0 or i == len(input_paths) - 1:
                print(f"  [{i+1:3d}/{len(input_paths)}] {filename}: {dims[0]}x{dims[1]} ({mp:.2f}MP) "
                      f"| {inf_time:.3f}s | RAM: {post_ram:.0f}MB | GPU: {post_gpu:.0f}MB")

        except Exception as e:
            print(f"  [{i+1:3d}/{len(input_paths)}] {filename}: ERROR - {e}")
            continue

    # Final stats
    if device.type == 'cuda':
        peak_gpu_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
        peak_gpu_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\n[IMAGES PROCESSED]")
    print(f"  Total:       {len(times)} images")
    print(f"  Megapixels:  {total_megapixels:.1f} MP total ({total_megapixels/len(times):.2f} MP avg)")

    print(f"\n[TIMING]")
    print(f"  Total time:  {sum(times):.2f}s")
    print(f"  Avg/image:   {sum(times)/len(times):.3f}s")
    print(f"  Min:         {min(times):.3f}s")
    print(f"  Max:         {max(times):.3f}s")
    print(f"  Throughput:  {len(times)/sum(times):.2f} img/s")
    print(f"  MP/second:   {total_megapixels/sum(times):.2f} MP/s")

    print(f"\n[MEMORY - RAM]")
    print(f"  Baseline:    {baseline_ram:.1f} MB")
    print(f"  After model: {after_model_ram:.1f} MB (+{model_ram_usage:.1f} MB for model)")
    print(f"  Peak:        {peak_ram:.1f} MB")
    print(f"  Net usage:   {peak_ram - baseline_ram:.1f} MB (model + inference)")

    if device.type == 'cuda':
        print(f"\n[MEMORY - GPU]")
        print(f"  Model size:      {model_gpu_usage:.1f} MB")
        print(f"  Peak allocated:  {peak_gpu_allocated:.1f} MB")
        print(f"  Peak reserved:   {peak_gpu_reserved:.1f} MB")
        print(f"  Net for inference: {peak_gpu_allocated - model_gpu_usage:.1f} MB")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()

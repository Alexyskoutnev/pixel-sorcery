#!/usr/bin/env python3
"""
03_verify_setup.py - Verify everything is ready for training

This script checks:
1. Python environment and packages
2. GPU availability and memory
3. Dataset structure and validity
4. Config file correctness
5. Pretrained weights availability

Usage:
    python scripts/03_verify_setup.py
"""

import os
import sys
from pathlib import Path


def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_ok(text):
    print(f"  ✅ {text}")


def print_warning(text):
    print(f"  ⚠️  {text}")


def print_error(text):
    print(f"  ❌ {text}")


def check_python():
    print_header("1. Python Environment")
    
    errors = []
    
    # Python version
    py_version = sys.version_info
    print(f"  Python: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version.major != 3 or py_version.minor < 8:
        print_warning("Python 3.8+ recommended")
    else:
        print_ok("Python version OK")
    
    # Required packages
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
        'lpips': 'LPIPS',
    }
    
    for module, name in packages.items():
        try:
            __import__(module)
            print_ok(f"{name} installed")
        except ImportError:
            print_error(f"{name} NOT installed")
            errors.append(f"pip install {module.replace('PIL', 'pillow').replace('cv2', 'opencv-python')}")
    
    # BasicSR
    try:
        import basicsr
        print_ok(f"BasicSR installed (v{basicsr.__version__})")
    except ImportError:
        print_error("BasicSR NOT installed")
        errors.append("cd BasicSR && pip install -e .")
    
    return errors


def check_gpu():
    print_header("2. GPU Status")
    
    errors = []
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print_ok("CUDA available")
            
            device_name = torch.cuda.get_device_name(0)
            print(f"  GPU: {device_name}")
            
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  Memory: {total_mem:.1f} GB")
            
            if total_mem >= 100:
                print_ok("Memory sufficient for batch_size=12, patch=384")
            elif total_mem >= 40:
                print_ok("Memory sufficient for batch_size=8, patch=256")
            elif total_mem >= 16:
                print_warning("Limited memory - using batch_size=4, patch=256")
            else:
                print_error("Insufficient GPU memory (<16GB)")
                errors.append("Need at least 16GB GPU memory")
            
            # Test tensor operation
            x = torch.randn(100, 100, device='cuda')
            y = torch.matmul(x, x)
            print_ok("GPU tensor operations working")
            
        else:
            print_error("CUDA not available!")
            errors.append("Install CUDA-enabled PyTorch")
            
    except Exception as e:
        print_error(f"GPU check failed: {e}")
        errors.append("Check GPU drivers and CUDA installation")
    
    return errors


def check_dataset():
    print_header("3. Dataset Structure")
    
    errors = []
    dataset_path = Path("datasets/realestate")
    
    if not dataset_path.exists():
        print_error("Dataset folder not found: datasets/realestate")
        errors.append("Run: python scripts/02_prepare_dataset.py")
        return errors
    
    # Check required folders
    required = [
        ("train/lq", "Training input images"),
        ("train/gt", "Training ground truth images"),
        ("val/lq", "Validation input images"),
        ("val/gt", "Validation ground truth images"),
    ]
    
    for folder, desc in required:
        folder_path = dataset_path / folder
        if folder_path.exists():
            count = len(list(folder_path.glob("*.png")))
            if count > 0:
                print_ok(f"{desc}: {count} images")
            else:
                print_error(f"{desc}: Empty!")
                errors.append(f"Add images to {folder_path}")
        else:
            print_error(f"{desc}: Folder missing!")
            errors.append(f"Create folder: {folder_path}")
    
    # Verify pairing
    train_lq = set(f.stem for f in (dataset_path / "train/lq").glob("*.png"))
    train_gt = set(f.stem for f in (dataset_path / "train/gt").glob("*.png"))
    
    if train_lq and train_gt:
        paired = train_lq & train_gt
        unpaired_lq = train_lq - train_gt
        unpaired_gt = train_gt - train_lq
        
        if unpaired_lq or unpaired_gt:
            print_warning(f"Unpaired images found: {len(unpaired_lq)} lq, {len(unpaired_gt)} gt")
        else:
            print_ok(f"All {len(paired)} training pairs matched correctly")
    
    return errors


def check_config():
    print_header("4. Training Configuration")
    
    errors = []
    config_path = Path("configs/nafnet_fast.yml")
    
    if not config_path.exists():
        print_error("Config file not found: configs/nafnet_fast.yml")
        errors.append("Config file missing - will be created")
        return errors
    
    print_ok(f"Config file found: {config_path}")
    
    # Parse and validate config
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Check key settings
        checks = [
            ('name', config.get('name')),
            ('gt_size', config.get('datasets', {}).get('train', {}).get('gt_size')),
            ('batch_size', config.get('datasets', {}).get('train', {}).get('batch_size_per_gpu')),
            ('total_iter', config.get('train', {}).get('total_iter')),
        ]
        
        for name, value in checks:
            if value:
                print(f"  {name}: {value}")
            else:
                print_warning(f"{name}: Not set in config")
        
        print_ok("Config parsed successfully")
        
    except Exception as e:
        print_error(f"Config parse error: {e}")
        errors.append("Fix config file syntax")
    
    return errors


def check_pretrained():
    print_header("5. Pretrained Weights")
    
    errors = []
    weights_path = Path("pretrained/NAFNet-SIDD-width32.pth")
    
    if weights_path.exists():
        size_mb = weights_path.stat().st_size / 1e6
        print_ok(f"Pretrained weights found ({size_mb:.1f} MB)")
    else:
        print_warning("Pretrained weights not found (will train from scratch)")
        print("  Download from: https://github.com/megvii-research/NAFNet")
        print("  Place at: pretrained/NAFNet-SIDD-width32.pth")
    
    return errors


def estimate_training_time():
    print_header("6. Training Time Estimate")
    
    try:
        import torch
        
        # Get dataset size
        dataset_path = Path("datasets/realestate/train/gt")
        if dataset_path.exists():
            n_images = len(list(dataset_path.glob("*.png")))
        else:
            n_images = 500  # Estimate
        
        # Get GPU memory
        if torch.cuda.is_available():
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            mem_gb = 16
        
        # Estimate based on config
        iterations = 30000
        
        if mem_gb >= 100:
            batch_size = 12
            iters_per_hour = 7000
        elif mem_gb >= 40:
            batch_size = 8
            iters_per_hour = 5000
        else:
            batch_size = 4
            iters_per_hour = 3000
        
        hours = iterations / iters_per_hour
        
        print(f"  Dataset size: {n_images} pairs")
        print(f"  GPU memory: {mem_gb:.0f} GB")
        print(f"  Batch size: {batch_size}")
        print(f"  Total iterations: {iterations:,}")
        print(f"  ⏱️  Estimated time: {hours:.1f} hours")
        
        if hours <= 5:
            print_ok("Training will complete in under 5 hours!")
        else:
            print_warning(f"Training may take {hours:.1f} hours")
            
    except Exception as e:
        print_warning(f"Could not estimate time: {e}")
    
    return []


def main():
    print("\n" + "=" * 60)
    print("  NAFNet Training Setup Verification")
    print("=" * 60)
    
    all_errors = []
    
    all_errors.extend(check_python())
    all_errors.extend(check_gpu())
    all_errors.extend(check_dataset())
    all_errors.extend(check_config())
    all_errors.extend(check_pretrained())
    estimate_training_time()
    
    print_header("Summary")
    
    if all_errors:
        print(f"\n  Found {len(all_errors)} issue(s) to fix:\n")
        for i, error in enumerate(all_errors, 1):
            print(f"  {i}. {error}")
        print("\n  Fix these issues before training.")
        sys.exit(1)
    else:
        print("""
  ✅ All checks passed! Ready to train.
  
  Start training with:
    ./scripts/04_train.sh
  
  Or manually:
    cd NAFNet
    python basicsr/train.py -opt ../configs/nafnet_fast.yml
        """)
        sys.exit(0)


if __name__ == "__main__":
    main()

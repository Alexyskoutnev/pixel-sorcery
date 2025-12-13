#!/bin/bash
# ============================================================
# QUICK START CHEATSHEET - Copy & Paste These Commands
# ============================================================
# Total time: ~4-5 hours
# ============================================================

# ======================
# STEP 1: Setup (10 min)
# ======================

# Make setup script executable and run it
chmod +x scripts/01_setup_environment.sh
./scripts/01_setup_environment.sh


# ======================
# STEP 2: Prepare Data (5 min)
# ======================

# Replace these paths with your actual image folders!
python scripts/02_prepare_dataset.py \
    --before /path/to/your/BEFORE_photos \
    --after /path/to/your/AFTER_photos \
    --output ./datasets/realestate


# ======================
# STEP 3: Verify (1 min)
# ======================

python scripts/03_verify_setup.py


# ======================
# STEP 4: Train (~4-5 hours)
# ======================

# Start training (run this and wait ~4-5 hours)
chmod +x scripts/04_train.sh
./scripts/04_train.sh

# OPTIONAL: Monitor training in another terminal
tensorboard --logdir experiments/NAFNet_RealEstate_Fast/tb_logger --port 6006
# Then open http://localhost:6006 in browser


# ======================
# STEP 5: Test (2 min)
# ======================

# Test on a single image
python scripts/05_inference.py \
    --input /path/to/test_image.png \
    --output ./results/enhanced.png

# Process a folder of images
python scripts/06_batch_inference.py \
    --input-dir /path/to/input_folder \
    --output-dir ./results/batch_output


# ======================
# OPTIONAL: Export for Production
# ======================

# Export to ONNX
python scripts/07_export_onnx.py \
    --model experiments/NAFNet_RealEstate_Fast/models/net_g_30000.pth \
    --output nafnet_realestate.onnx


# ============================================================
# TROUBLESHOOTING
# ============================================================

# If training crashes with OOM (out of memory):
# Edit configs/nafnet_fast.yml and change:
#   batch_size_per_gpu: 8  (reduce from 12)
#   gt_size: 256           (reduce from 384)

# If training seems stuck:
# Check GPU is being used:
watch -n 1 nvidia-smi

# Resume training if it stopped:
./scripts/04_train.sh  # It auto-resumes from checkpoint

# ============================================================
# EXPECTED RESULTS
# ============================================================
# 
# After training, your model will be at:
#   experiments/NAFNet_RealEstate_Fast/models/net_g_30000.pth
#
# Quality metrics (approximate):
#   - PSNR: 25-30 dB (higher = better)
#   - SSIM: 0.85-0.95 (higher = better)
#   - Visual: ~80% of professional editing quality
#
# Inference speed:
#   - 512x512 image: ~0.05 seconds
#   - 3300x2200 image: ~2-5 seconds (tiled)
# ============================================================

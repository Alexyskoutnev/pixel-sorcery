#!/bin/bash
# ============================================================
# 04_train.sh - Launch NAFNet training
# Expected time: ~4-5 hours on GB10
# ============================================================

set -e

echo "=============================================="
echo "  NAFNet Real Estate Enhancement Training"
echo "  Expected duration: ~4-5 hours"
echo "=============================================="

# Configuration
CONFIG_FILE="configs/nafnet_fast.yml"
NAFNET_DIR="NAFNet"

# Check config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check NAFNet directory exists
if [ ! -d "$NAFNET_DIR" ]; then
    echo "❌ NAFNet directory not found. Run setup first:"
    echo "   ./scripts/01_setup_environment.sh"
    exit 1
fi

# Set environment variables for optimal performance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Print GPU info
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Start training
echo "Starting training..."
echo "Monitor progress with: tensorboard --logdir experiments/NAFNet_RealEstate_Fast/tb_logger"
echo ""

cd $NAFNET_DIR

# Check if resuming from checkpoint
if [ -f "../experiments/NAFNet_RealEstate_Fast/training_states/latest.state" ]; then
    echo "Found existing checkpoint - resuming training..."
    python -u basicsr/train.py -opt ../$CONFIG_FILE --auto_resume
else
    echo "Starting fresh training..."
    python -u basicsr/train.py -opt ../$CONFIG_FILE
fi

echo ""
echo "=============================================="
echo "  Training Complete!"
echo "=============================================="
echo ""
echo "Your trained model is at:"
echo "  experiments/NAFNet_RealEstate_Fast/models/net_g_latest.pth"
echo ""
echo "Test it with:"
echo "  python scripts/05_inference.py --input test.png --output enhanced.png"

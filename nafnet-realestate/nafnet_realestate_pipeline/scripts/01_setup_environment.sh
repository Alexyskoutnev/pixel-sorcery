#!/bin/bash
# ============================================================
# 01_setup_environment.sh - Complete environment setup for NAFNet
# Run time: ~10 minutes
# ============================================================

set -e  # Exit on any error

echo "=============================================="
echo "  NAFNet Real Estate Enhancement Setup"
echo "  Estimated time: ~10 minutes"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if conda is available
if command -v conda &> /dev/null; then
    print_status "Conda found"
else
    print_warning "Conda not found, using pip with system Python"
fi

# Create project directories
echo ""
echo "Creating directory structure..."
mkdir -p datasets/realestate/{train,val}/{lq,gt}
mkdir -p experiments
mkdir -p results
mkdir -p pretrained
print_status "Directories created"

# Install PyTorch (detect CUDA version)
echo ""
echo "Installing PyTorch..."

# Try to detect CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    print_status "Detected CUDA $CUDA_VERSION"
else
    CUDA_VERSION="12.1"
    print_warning "CUDA version not detected, assuming 12.1"
fi

# Install appropriate PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

print_status "PyTorch installed"

# Install core dependencies
echo ""
echo "Installing dependencies..."
pip install numpy scipy opencv-python pillow lmdb pyyaml
pip install addict future requests tensorboard yapf scikit-image tqdm
pip install einops gdown lpips matplotlib scikit-learn

print_status "Core dependencies installed"

# Clone and install BasicSR
echo ""
echo "Setting up BasicSR..."
if [ ! -d "BasicSR" ]; then
    git clone https://github.com/XPixelGroup/BasicSR.git
fi
cd BasicSR
pip install -r requirements.txt
pip install -e .
cd ..
print_status "BasicSR installed"

# Clone and install NAFNet
echo ""
echo "Setting up NAFNet..."
if [ ! -d "NAFNet" ]; then
    git clone https://github.com/megvii-research/NAFNet.git
fi
cd NAFNet
pip install -r requirements.txt
pip install -e .
cd ..
print_status "NAFNet installed"

# Download pretrained weights
echo ""
echo "Downloading pretrained weights..."
cd pretrained

# NAFNet-SIDD-width32 (smaller, faster model we'll use)
if [ ! -f "NAFNet-SIDD-width32.pth" ]; then
    echo "Downloading NAFNet-SIDD-width32.pth..."
    gdown "https://drive.google.com/uc?id=1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ" -O NAFNet-SIDD-width32.pth 2>/dev/null || \
    wget -O NAFNet-SIDD-width32.pth "https://github.com/megvii-research/NAFNet/releases/download/v0.0.1/NAFNet-SIDD-width32.pth" 2>/dev/null || \
    print_warning "Could not download pretrained weights automatically. Please download manually from: https://github.com/megvii-research/NAFNet"
fi

cd ..
print_status "Pretrained weights ready"

# Set environment variables
echo ""
echo "Setting environment variables..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Add to .bashrc for persistence
if ! grep -q "PYTORCH_CUDA_ALLOC_CONF" ~/.bashrc; then
    echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc
fi

print_status "Environment variables set"

# Final verification
echo ""
echo "=============================================="
echo "  Setup Complete! Running quick verification..."
echo "=============================================="

python3 << 'EOF'
import sys
print(f"Python: {sys.version.split()[0]}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

try:
    import basicsr
    print(f"BasicSR: OK")
except:
    print("BasicSR: FAILED")

print("\n✅ Setup complete! Run: python scripts/03_verify_setup.py for full verification")
EOF

echo ""
echo "=============================================="
echo "  Next Step: Prepare your dataset"
echo "  Run: python scripts/02_prepare_dataset.py --help"
echo "=============================================="

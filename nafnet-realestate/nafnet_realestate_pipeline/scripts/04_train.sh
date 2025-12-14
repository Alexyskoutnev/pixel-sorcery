#!/bin/bash
# ============================================================
# 04_train.sh - Launch NAFNet baseline training (fast config)
# Expected time: ~4-5 hours on GB10
#
# Notes:
# - Runs against the local BasicSR checkout under `nafnet-realestate/BasicSR/`
# - Can be run from anywhere
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"  # .../nafnet-realestate

CONFIG_FILE="${PROJECT_ROOT}/nafnet_realestate_pipeline/configs/nafnet_fast.yml"

echo "=============================================="
echo "  NAFNet Real Estate Enhancement Training"
echo "  Expected duration: ~4-5 hours"
echo "=============================================="
echo ""
echo "Project root: ${PROJECT_ROOT}"
echo "Config:       ${CONFIG_FILE}"
echo ""

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "❌ Config file not found: ${CONFIG_FILE}"
  exit 1
fi

if [ ! -d "${PROJECT_ROOT}/BasicSR/basicsr" ]; then
  echo "❌ BasicSR not found at: ${PROJECT_ROOT}/BasicSR"
  echo "   Run: ${PROJECT_ROOT}/nafnet_realestate_pipeline/scripts/01_setup_environment.sh"
  exit 1
fi

# Set environment variables for optimal performance / stability.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "GPU Status:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
  echo "nvidia-smi not found"
fi
echo ""

# Prefer the known-good conda env if present; allow override via PYTHON_BIN.
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ]; then
  if [ -x "/home/asus/miniconda3/envs/nafnet/bin/python" ]; then
    PYTHON_BIN="/home/asus/miniconda3/envs/nafnet/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

echo "Using python: ${PYTHON_BIN}"
echo ""

cd "${PROJECT_ROOT}"

# BasicSR is a local checkout; ensure it’s importable without requiring `pip install -e .`.
export PYTHONPATH="${PROJECT_ROOT}/BasicSR:${PYTHONPATH:-}"

echo "Starting training..."
echo "TensorBoard:"
echo "  tensorboard --logdir BasicSR/tb_logger"
echo ""

AUTO_RESUME_FLAG=""
if [ -f "${PROJECT_ROOT}/BasicSR/experiments/NAFNet_RealEstate_Fast/training_states/latest.state" ]; then
  AUTO_RESUME_FLAG="--auto_resume"
  echo "Found existing checkpoint → enabling --auto_resume"
  echo ""
fi

"${PYTHON_BIN}" -u BasicSR/basicsr/train.py -opt "${CONFIG_FILE}" ${AUTO_RESUME_FLAG}

echo ""
echo "=============================================="
echo "  Training Complete!"
echo "=============================================="
echo ""
echo "Checkpoints should be under:"
echo "  BasicSR/experiments/NAFNet_RealEstate_Fast/models/"
echo ""

#!/bin/bash
# ============================================================
# 08_train_color_finetune_3h.sh
#
# Purpose:
# - Apply the ColorAwareLoss patch into the local BasicSR checkout
# - Run a 3-hour fine-tune focused on color fidelity
#
# Notes:
# - Run from anywhere; the script will cd into `nafnet-realestate/`.
# - This does NOT require any new pip deps (YCbCr chroma constraint only).
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"  # .../nafnet-realestate

CONFIG_FILE="${PROJECT_ROOT}/nafnet_realestate_pipeline/configs/nafnet_color_finetune_3h.yml"

PATCH_SRC="${PROJECT_ROOT}/nafnet_realestate_pipeline/patches/basicsr/losses/color_aware_loss.py"
PATCH_DST="${PROJECT_ROOT}/BasicSR/basicsr/losses/color_aware_loss.py"

echo "=============================================="
echo "  NAFNet Color Fine-tune (3h budget)"
echo "=============================================="
echo ""
echo "Project root: ${PROJECT_ROOT}"
echo "Config:       ${CONFIG_FILE}"
echo ""

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "❌ Config file not found: ${CONFIG_FILE}"
  exit 1
fi

if [ ! -f "${PATCH_SRC}" ]; then
  echo "❌ Patch source not found: ${PATCH_SRC}"
  exit 1
fi

if [ ! -d "${PROJECT_ROOT}/BasicSR/basicsr" ]; then
  echo "❌ BasicSR not found at: ${PROJECT_ROOT}/BasicSR"
  echo "   Expected a local BasicSR checkout at nafnet-realestate/BasicSR/"
  exit 1
fi

echo "Applying BasicSR patch:"
echo "  ${PATCH_SRC}"
echo "→ ${PATCH_DST}"
mkdir -p "$(dirname "${PATCH_DST}")"
cp -f "${PATCH_SRC}" "${PATCH_DST}"
echo ""

# Environment variables for more stable CUDA allocations
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "GPU Status:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
  echo "nvidia-smi not found"
fi
echo ""

cd "${PROJECT_ROOT}"

echo "Starting training..."
echo "TensorBoard:"
echo "  tensorboard --logdir BasicSR/tb_logger"
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

# BasicSR is a local checkout; ensure it’s importable without requiring `pip install -e .`.
export PYTHONPATH="${PROJECT_ROOT}/BasicSR:${PYTHONPATH:-}"

AUTO_RESUME_FLAG=""
if [ "${AUTO_RESUME:-0}" = "1" ]; then
  AUTO_RESUME_FLAG="--auto_resume"
  echo "AUTO_RESUME=1 → enabling --auto_resume"
  echo ""
fi

"${PYTHON_BIN}" -u BasicSR/basicsr/train.py -opt "${CONFIG_FILE}" ${AUTO_RESUME_FLAG}

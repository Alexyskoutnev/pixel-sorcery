#!/bin/bash
# Script to push everything to GitHub and Hugging Face
# Run from: /home/asus/sebastian/pixel-sorcery

set -e

cd /home/asus/sebastian/pixel-sorcery

echo "=== Step 1: Commit folder rename and benchmark ==="
git add -A
git status
git commit -m "Rename pix2pix-train-v1 to nafnet-realestate, add benchmark results

- Renamed folder to reflect actual project (NAFNet, not pix2pix)
- Added BENCHMARK_RESULTS.md with full performance metrics
- RAM: 581 MB, GPU: 8.3 GB, Time: 4s/image (7MP)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

echo "=== Step 2: Rename branch ==="
git branch -m sebastian/pix2pix sebastian/nafnet-realestate

echo "=== Step 3: Push to GitHub ==="
# Replace YOUR_TOKEN with actual token
git push origin --delete sebastian/pix2pix 2>/dev/null || true
git push -u origin sebastian/nafnet-realestate

echo "=== Step 4: Push models to Hugging Face ==="
pip install huggingface_hub -q

python3 << 'PYTHON'
import os
from huggingface_hub import HfApi, create_repo

# Get token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set")

api = HfApi(token=HF_TOKEN)

# Create repo if doesn't exist
try:
    create_repo("SebRincon/nafnet-realestate", repo_type="model", private=False, token=HF_TOKEN)
    print("Created new repo: SebRincon/nafnet-realestate")
except Exception as e:
    print(f"Repo exists or error: {e}")

# Upload trained model
print("Uploading trained model...")
api.upload_file(
    path_or_fileobj="/home/asus/sebastian/pixel-sorcery/nafnet-realestate/BasicSR/experiments/NAFNet_RealEstate_Fast/models/net_g_12000.pth",
    path_in_repo="nafnet_realestate.pth",
    repo_id="SebRincon/nafnet-realestate",
)

# Upload ONNX model
print("Uploading ONNX model...")
api.upload_file(
    path_or_fileobj="/home/asus/sebastian/pixel-sorcery/nafnet-realestate/mobile_models/nafnet_realestate.onnx",
    path_in_repo="nafnet_realestate.onnx",
    repo_id="SebRincon/nafnet-realestate",
)

# Upload benchmark results
print("Uploading benchmark results...")
api.upload_file(
    path_or_fileobj="/home/asus/sebastian/pixel-sorcery/nafnet-realestate/BENCHMARK_RESULTS.md",
    path_in_repo="README.md",
    repo_id="SebRincon/nafnet-realestate",
)

print("\n✅ Done! Models uploaded to: https://huggingface.co/SebRincon/nafnet-realestate")
PYTHON

echo "=== All done! ==="

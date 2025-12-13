#!/bin/bash
# Sync checkpoints from DGX to local machine

# === CONFIGURATION ===
DGX_USER="asus"
DGX_HOST="gx10-d56e"  # Or use IP address like "192.168.1.100"
DGX_PATH="~/alexy/pixel-sorcery/checkpoints"
LOCAL_PATH="./checkpoints_dgx"

# === SYNC ===
echo "Syncing checkpoints from ${DGX_USER}@${DGX_HOST}..."

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_PATH"

# Rsync with progress (use -z for compression over slow connections)
rsync -avh --progress \
    "${DGX_USER}@${DGX_HOST}:${DGX_PATH}/" \
    "$LOCAL_PATH/"

echo ""
echo "Done! Checkpoints synced to: $LOCAL_PATH"
ls -la "$LOCAL_PATH"

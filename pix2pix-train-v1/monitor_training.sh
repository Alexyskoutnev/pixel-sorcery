#!/bin/bash
# Training monitor - checks every 30 minutes, restarts if crashed

LOG_FILE="/home/asus/sebastian/pixel-sorcery/pix2pix-train-v1/training.log"
MONITOR_LOG="/home/asus/sebastian/pixel-sorcery/pix2pix-train-v1/monitor.log"
CONFIG="/home/asus/sebastian/pixel-sorcery/pix2pix-train-v1/nafnet_realestate_pipeline/configs/nafnet_fast.yml"
WORKDIR="/home/asus/sebastian/pixel-sorcery/pix2pix-train-v1"

cd "$WORKDIR"

check_and_restart() {
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Check if training process is running
    PROC_COUNT=$(pgrep -c -f "python.*train.py" 2>/dev/null || echo "0")

    if [ "$PROC_COUNT" -lt 2 ]; then
        echo "[$TIMESTAMP] Training not running (procs: $PROC_COUNT). Checking log..." >> "$MONITOR_LOG"

        # Check last line of log for errors
        LAST_LINE=$(tail -1 "$LOG_FILE")

        if echo "$LAST_LINE" | grep -qi "error\|exception\|traceback"; then
            echo "[$TIMESTAMP] Error detected. Restarting training..." >> "$MONITOR_LOG"

            # Find latest checkpoint
            LATEST_STATE=$(ls -t $WORKDIR/BasicSR/experiments/NAFNet_RealEstate_Fast/training_states/*.state 2>/dev/null | head -1)

            if [ -n "$LATEST_STATE" ]; then
                echo "[$TIMESTAMP] Resuming from: $LATEST_STATE" >> "$MONITOR_LOG"
                # Update config with latest state (using sed)
                sed -i "s|resume_state:.*|resume_state: $LATEST_STATE|" "$CONFIG"
            fi

            # Restart training
            source ~/miniconda3/etc/profile.d/conda.sh
            conda activate nafnet
            nohup python BasicSR/basicsr/train.py -opt "$CONFIG" >> "$LOG_FILE" 2>&1 &
            echo "[$TIMESTAMP] Training restarted with PID: $!" >> "$MONITOR_LOG"
        else
            echo "[$TIMESTAMP] No error in log. Training may have completed." >> "$MONITOR_LOG"
        fi
    else
        # Get current iteration
        ITER=$(grep -oP 'iter:\s*\K[\d,]+' "$LOG_FILE" | tail -1 | tr -d ',')
        echo "[$TIMESTAMP] Training running OK (procs: $PROC_COUNT, iter: $ITER)" >> "$MONITOR_LOG"
    fi
}

echo "Starting training monitor at $(date)" >> "$MONITOR_LOG"

# Run check every 30 minutes for 8 hours (16 checks)
for i in {1..16}; do
    check_and_restart
    sleep 1800  # 30 minutes
done

echo "Monitor finished at $(date)" >> "$MONITOR_LOG"

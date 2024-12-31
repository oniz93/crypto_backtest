#!/bin/bash

# =====================================================================
# Script: deploy_bitmex.sh
# Description: Copies specific files and directories to multiple Raspberry Pi hosts in parallel using rsync.
# =====================================================================

# ---------------------------
# Configuration Variables
# ---------------------------

# Source directory on the local machine
SRC_DIR="/Users/teomiscia/PycharmProjects/bitmex_liquidation_2"

# Destination directory on the remote hosts
DEST_DIR="~/bitmex_liquidation_2/"

# Files and directories to copy
ITEMS=(
    "config.yaml"
    "convert_trades.py"
    "indicators_config.json"
    "main.py"
    "merge_funding_history.py"
    "merge_timeframe_files.py"
    "merge_trades.py"
    "precalculate_indicators.py"
    "printContent.sh"
    "requirements.txt"
    "src"
    "train_rl.py"
    "use_trained_model.py"
)

# Destination hosts
HOSTS=(
    "clusterctrl"
    "rpi3red"
    "rpi3nude"
    "rpi4"
    "mbp"
    "steamdeck"
)

# Maximum number of parallel jobs
MAX_PARALLEL=10

# ---------------------------
# Utility Functions
# ---------------------------

# Function to copy files to a single host
copy_to_host() {
    local HOST=$1
    echo "[$HOST] Starting rsync..."
    
    rsync -avz --progress "$SRC_DIR/${ITEMS[@]}" "$HOST":"$DEST_DIR" \
    && echo "[$HOST] rsync completed successfully." \
    || echo "[$HOST] rsync encountered errors."
}

# Function to limit the number of parallel jobs
wait_for_jobs() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 1
    done
}

# ---------------------------
# Main Script Execution
# ---------------------------

# Check if source directory exists
if [ ! -d "$SRC_DIR" ]; then
    echo "Error: Source directory '$SRC_DIR' does not exist."
    exit 1
fi

# Start copying to each host
for HOST in "${HOSTS[@]}"; do
    # Wait if maximum parallel jobs are running
    wait_for_jobs
    
    # Start the copy in the background
    copy_to_host "$HOST" &
done

# Wait for all background jobs to finish
wait

echo "All rsync operations have completed."

#!/bin/bash
# syncall.sh
# -------------
# This script deploys the project files to multiple Raspberry Pi hosts in parallel using rsync.
# It copies specified files and directories from the local machine to the destination hosts.
#
# Usage:
#   ./syncall.sh
#
# Configuration:
SRC_DIR="/Users/teomiscia/PycharmProjects/bitmex_liquidation_2"
DEST_DIR="~/bitmex_liquidation_2/"
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
HOSTS=(
    "clusterctrl"
    "rpi3red"
    "rpi3nude"
    "rpi4"
    "mbpw"
    "mbpm3"
)
MAX_PARALLEL=10

# Function to copy files to a single host.
copy_to_host() {
    local HOST=$1
    echo "[$HOST] Starting rsync..."
    rsync -avz --progress "$SRC_DIR/${ITEMS[@]}" "$HOST":"$DEST_DIR" \
    && echo "[$HOST] rsync completed successfully." \
    || echo "[$HOST] rsync encountered errors."
}

# Function to limit the number of parallel jobs.
wait_for_jobs() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 1
    done
}

# Main execution: iterate over hosts and copy files in parallel.
if [ ! -d "$SRC_DIR" ]; then
    echo "Error: Source directory '$SRC_DIR' does not exist."
    exit 1
fi

for HOST in "${HOSTS[@]}"; do
    wait_for_jobs
    copy_to_host "$HOST" &
done

# Wait for all background jobs to finish.
wait
echo "All rsync operations have completed."

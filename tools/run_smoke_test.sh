#!/bin/bash
set -e

# ============================================================================
# Smoke Test Script for Cloud Container
# 
# Objective: Verify environment, paths, dataloader, and train loop without 
# waiting for a full epoch.
# ============================================================================

echo "==========================================================="
echo "  Starting Smoke Test..."
echo "==========================================================="

# Use a very small train_ratio and just 1 epoch for quick validation
TRAIN_RATIO=0.01
EPOCHS=1
BATCH_SIZE=4
LR=1e-5
LOG_DIR="logs/smoke_test"

mkdir -p $LOG_DIR

# Ensure we are running with the gtauav conda env if available
# But we assume the user has already activated it.
if ! command -v python &> /dev/null; then
    echo "Python could not be found. Please activate conda environment first."
    exit 1
fi

echo ">>> Running minimal baseline forward/backward..."
python Game4Loc/train_gta.py \
    --data_root Game4Loc/data/GTA-UAV-data \
    --log_path $LOG_DIR \
    --train_ratio $TRAIN_RATIO \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --no_wandb \
    --num_workers 2

echo -e "\n==========================================================="
echo "  Smoke Test Completed Successfully!"
echo "  Environment and data paths are correct."
echo "  You can now run formal experiments."
echo "==========================================================="

#!/bin/bash
set -e

TRAIN_RATIO=0.05
EPOCHS=1
BATCH_SIZE=8
LR=1e-5

echo "==========================================================="
echo "  Starting Sanity Experiment (Short Run)"
echo "  Data Ratio: $TRAIN_RATIO, Epochs: $EPOCHS"
echo "==========================================================="

echo -e "\n\n>>> 1. Running Baseline..."
LOG_DIR_BASE="logs/sanity_baseline"
mkdir -p $LOG_DIR_BASE

conda run -n gtauav python Game4Loc/train_gta.py \
    --data_root "Game4Loc/data/GTA-UAV-data" \
    --log_path $LOG_DIR_BASE \
    --train_ratio $TRAIN_RATIO \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --no_wandb

echo -e "\n\n>>> 2. Running Pose Attention Gate..."
LOG_DIR_POSE="logs/sanity_pose_attention"
mkdir -p $LOG_DIR_POSE

conda run -n gtauav python Game4Loc/train_gta.py \
    --data_root "Game4Loc/data/GTA-UAV-data" \
    --log_path $LOG_DIR_POSE \
    --train_ratio $TRAIN_RATIO \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --no_wandb \
    --use_pose_attention

echo -e "\n\n==========================================================="
echo "  Sanity Experiment Completed!"
echo "  Please check the logs in:"
echo "    - $LOG_DIR_BASE"
echo "    - $LOG_DIR_POSE"
echo "==========================================================="
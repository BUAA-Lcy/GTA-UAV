#!/bin/bash
set -e

# ============================================================================
# Pose Attention Gating: 2~3 Epoch Minimal Formal Comparison
# 
# Objective: Validate if residual0.25 and floor0.8 hold stable advantages
# over baseline in a slightly longer (but still small-scale) run.
# ============================================================================

TRAIN_RATIO=0.1
EPOCHS=3
BATCH_SIZE=8
LR=1e-5

BASE_CMD="conda run -n gtauav python Game4Loc/train_gta.py \
    --data_root Game4Loc/data/GTA-UAV-data \
    --train_ratio $TRAIN_RATIO \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --no_wandb"

echo "==========================================================="
echo "  Starting Minimal Formal Comparison Experiment"
echo "  Data Ratio: $TRAIN_RATIO, Epochs: $EPOCHS"
echo "==========================================================="

# 1. Baseline
echo -e "\n\n>>> [1/3] Running Baseline..."
DIR_BASE="logs/compare_baseline"
mkdir -p $DIR_BASE
$BASE_CMD --log_path $DIR_BASE

# 2. Pose Attention: pre_blocks + residual + lambda=0.25
echo -e "\n\n>>> [2/3] Running Pose Attention: residual + lambda=0.25"
DIR_RES="logs/compare_residual025"
mkdir -p $DIR_RES
$BASE_CMD --log_path $DIR_RES \
    --use_pose_attention \
    --pose_attn_floor 0.3 \
    --pose_gate_mode residual \
    --pose_gate_lambda 0.25 \
    --pose_gate_insert_stage pre_blocks

# 3. Pose Attention: pre_blocks + multiplicative + floor=0.8
echo -e "\n\n>>> [3/3] Running Pose Attention: multiplicative + floor=0.8"
DIR_FLOOR="logs/compare_floor08"
mkdir -p $DIR_FLOOR
$BASE_CMD --log_path $DIR_FLOOR \
    --use_pose_attention \
    --pose_attn_floor 0.8 \
    --pose_gate_mode multiplicative \
    --pose_gate_insert_stage pre_blocks

echo -e "\n\n==========================================================="
echo "  All 3 Experiments Completed!"
echo "==========================================================="
echo -e "\n  [Results Summary]"

# A simple grep logic to extract final best Recall@1
extract_results() {
    local dir=$1
    local name=$2
    local log_file=$(ls -t $dir/*.log | head -n 1)
    
    if [ -f "$log_file" ]; then
        echo "--------------------------------------------------------"
        echo "Experiment: $name"
        # Print hyperparams from log
        grep -E "Use pose attention:|Pose Attention Config:" $log_file | tail -n 2
        # Print final results
        echo "Final Best Metrics:"
        grep "训练完成，最佳" $log_file || echo "Training might not have completed normally."
        # Print the last evaluation summary
        grep "评估结果摘要:" $log_file | tail -n 1
    else
        echo "Log file not found in $dir"
    fi
}

extract_results $DIR_BASE "Baseline"
extract_results $DIR_RES "Residual (lambda=0.25)"
extract_results $DIR_FLOOR "Multiplicative (floor=0.8)"
echo "--------------------------------------------------------"

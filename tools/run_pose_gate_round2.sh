#!/bin/bash
set -e

# ============================================================================
# Pose Attention Gating: Round 2 Short-run Experiments
# 
# Objective: 
# 1. Verify residual gate robustness and sweep best lambda [0.10, 0.25, 0.50]
# 2. Check if multiplicative mode works better at lower floor [0.10, 0.20, 0.30]
# ============================================================================

# Short-run config
TRAIN_RATIO=0.1
EPOCHS=3
BATCH_SIZE=8
LR=1e-5

# 移除 conda run 强绑定，假设用户已激活环境
BASE_CMD="python Game4Loc/train_gta.py \
    --data_root Game4Loc/data/GTA-UAV-data \
    --train_ratio $TRAIN_RATIO \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --no_wandb"

echo "==========================================================="
echo "  Starting Pose Gate Round 2 Experiments"
echo "  Data Ratio: $TRAIN_RATIO, Epochs: $EPOCHS"
echo "==========================================================="

# A. Baseline
echo -e "\n\n>>> [1/7] Running 1_baseline..."
DIR_1="logs/1_baseline"
mkdir -p $DIR_1
$BASE_CMD --log_path $DIR_1

# B. Residual, lambda=0.10
echo -e "\n\n>>> [2/7] Running 2_residual_l010..."
DIR_2="logs/2_residual_l010"
mkdir -p $DIR_2
$BASE_CMD --log_path $DIR_2 \
    --use_pose_attention \
    --pose_attn_floor 0.3 \
    --pose_gate_mode residual \
    --pose_gate_lambda 0.10 \
    --pose_gate_insert_stage pre_blocks

# C. Residual, lambda=0.25
echo -e "\n\n>>> [3/7] Running 3_residual_l025..."
DIR_3="logs/3_residual_l025"
mkdir -p $DIR_3
$BASE_CMD --log_path $DIR_3 \
    --use_pose_attention \
    --pose_attn_floor 0.3 \
    --pose_gate_mode residual \
    --pose_gate_lambda 0.25 \
    --pose_gate_insert_stage pre_blocks

# D. Residual, lambda=0.50
echo -e "\n\n>>> [4/7] Running 4_residual_l050..."
DIR_4="logs/4_residual_l050"
mkdir -p $DIR_4
$BASE_CMD --log_path $DIR_4 \
    --use_pose_attention \
    --pose_attn_floor 0.3 \
    --pose_gate_mode residual \
    --pose_gate_lambda 0.50 \
    --pose_gate_insert_stage pre_blocks

# E. Multiplicative, floor=0.10
echo -e "\n\n>>> [5/7] Running 5_mult_f010..."
DIR_5="logs/5_mult_f010"
mkdir -p $DIR_5
$BASE_CMD --log_path $DIR_5 \
    --use_pose_attention \
    --pose_attn_floor 0.10 \
    --pose_gate_mode multiplicative \
    --pose_gate_insert_stage pre_blocks

# F. Multiplicative, floor=0.20
echo -e "\n\n>>> [6/7] Running 6_mult_f020..."
DIR_6="logs/6_mult_f020"
mkdir -p $DIR_6
$BASE_CMD --log_path $DIR_6 \
    --use_pose_attention \
    --pose_attn_floor 0.20 \
    --pose_gate_mode multiplicative \
    --pose_gate_insert_stage pre_blocks

# G. Multiplicative, floor=0.30
echo -e "\n\n>>> [7/7] Running 7_mult_f030..."
DIR_7="logs/7_mult_f030"
mkdir -p $DIR_7
$BASE_CMD --log_path $DIR_7 \
    --use_pose_attention \
    --pose_attn_floor 0.30 \
    --pose_gate_mode multiplicative \
    --pose_gate_insert_stage pre_blocks


echo -e "\n\n==========================================================="
echo "  All 7 Experiments Completed!"
echo "==========================================================="
echo -e "\n  [Results Summary]"

extract_results() {
    local dir=$1
    local name=$2
    local log_file=$(ls -t $dir/*.log 2>/dev/null | head -n 1)
    
    if [ -f "$log_file" ]; then
        echo "--------------------------------------------------------"
        echo "Experiment: $name"
        grep -E "Use pose attention:|Pose Attention Config:" $log_file | tail -n 2
        echo "Final Best Metrics:"
        grep "训练完成，最佳" $log_file || echo "Training might not have completed normally."
        grep "评估结果摘要:" $log_file | tail -n 1
    else
        echo "Log file not found in $dir"
    fi
}

extract_results $DIR_1 "1_baseline"
extract_results $DIR_2 "2_residual_l010"
extract_results $DIR_3 "3_residual_l025"
extract_results $DIR_4 "4_residual_l050"
extract_results $DIR_5 "5_mult_f010"
extract_results $DIR_6 "6_mult_f020"
extract_results $DIR_7 "7_mult_f030"
echo "--------------------------------------------------------"

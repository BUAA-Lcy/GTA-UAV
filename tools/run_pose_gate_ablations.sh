#!/bin/bash
set -e

# ============================================================================
# Pose Attention Gating Ablation Experiments (Short Run)
# 
# 这个脚本用于快速定位 gating 导致负收益的原因：
# 是强度过高 (floor 太低 / 硬乘法) 还是 插入点不对 (pre_blocks)？
# ============================================================================

TRAIN_RATIO=0.05
EPOCHS=1
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
echo "  Starting Pose Gate Ablation Experiments (Short Run) "
echo "==========================================================="

# 1. Baseline
echo -e "\n\n>>> [1/8] Running Baseline..."
$BASE_CMD --log_path logs/ablation_1_baseline

# 2. 实验 A: 提高 gating floor
echo -e "\n\n>>> [2/8] Experiment A1: floor=0.5"
$BASE_CMD --log_path logs/ablation_2_floor0.5 --use_pose_attention --pose_attn_floor 0.5 --pose_gate_mode multiplicative --pose_gate_insert_stage pre_blocks

echo -e "\n\n>>> [3/8] Experiment A2: floor=0.7"
$BASE_CMD --log_path logs/ablation_3_floor0.7 --use_pose_attention --pose_attn_floor 0.7 --pose_gate_mode multiplicative --pose_gate_insert_stage pre_blocks

echo -e "\n\n>>> [4/8] Experiment A3: floor=0.8"
$BASE_CMD --log_path logs/ablation_4_floor0.8 --use_pose_attention --pose_attn_floor 0.8 --pose_gate_mode multiplicative --pose_gate_insert_stage pre_blocks

# 3. 实验 B: 改成 residual gate
echo -e "\n\n>>> [5/8] Experiment B1: residual lambda=0.25"
$BASE_CMD --log_path logs/ablation_5_residual0.25 --use_pose_attention --pose_attn_floor 0.3 --pose_gate_mode residual --pose_gate_lambda 0.25 --pose_gate_insert_stage pre_blocks

echo -e "\n\n>>> [6/8] Experiment B2: residual lambda=0.5"
$BASE_CMD --log_path logs/ablation_6_residual0.5 --use_pose_attention --pose_attn_floor 0.3 --pose_gate_mode residual --pose_gate_lambda 0.5 --pose_gate_insert_stage pre_blocks

# 4. 实验 C: 后移插入点
echo -e "\n\n>>> [7/8] Experiment C1: after_block2"
$BASE_CMD --log_path logs/ablation_7_after_block2 --use_pose_attention --pose_attn_floor 0.3 --pose_gate_mode multiplicative --pose_gate_insert_stage after_block2

echo -e "\n\n>>> [8/8] Experiment C2: after_block4"
$BASE_CMD --log_path logs/ablation_8_after_block4 --use_pose_attention --pose_attn_floor 0.3 --pose_gate_mode multiplicative --pose_gate_insert_stage after_block4

echo -e "\n\n==========================================================="
echo "  All Ablation Experiments Completed! "
echo "  Logs saved to logs/ablation_* directories."
echo "==========================================================="

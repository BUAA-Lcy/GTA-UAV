#!/bin/bash
set -e

# ============================================================================
# Same-area Residual Mainline Comparison (Slightly Longer Run)
# Baseline vs Residual (lambda=0.25, pre_blocks)
# ============================================================================

# Slightly longer but still local-friendly configuration
TRAIN_RATIO=0.1
EPOCHS=5
BATCH_SIZE=8
LR=1e-5

# Same-area metafiles
TRAIN_META="same-area-drone2sate-train.json"
TEST_META="same-area-drone2sate-test.json"

BASE_CMD="python Game4Loc/train_gta.py \
    --data_root Game4Loc/data/GTA-UAV-data \
    --train_pairs_meta_file $TRAIN_META \
    --test_pairs_meta_file $TEST_META \
    --train_ratio $TRAIN_RATIO \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --no_wandb"

echo "==========================================================="
echo "  Same-area Residual Mainline Comparison"
echo "  Data Ratio: $TRAIN_RATIO, Epochs: $EPOCHS"
echo "  Metafiles: train=$TRAIN_META, test=$TEST_META"
echo "==========================================================="

# Baseline
echo -e "\n\n>>> [1/2] Running baseline (same-area)..."
DIR_BASE="logs/same_mainline_baseline"
mkdir -p "$DIR_BASE"
$BASE_CMD --log_path "$DIR_BASE"

# Residual 0.25 @ pre_blocks
echo -e "\n\n>>> [2/2] Running residual (lambda=0.25, pre_blocks) (same-area)..."
DIR_RES="logs/same_mainline_residual_l025"
mkdir -p "$DIR_RES"
$BASE_CMD --log_path "$DIR_RES" \
    --use_pose_attention \
    --pose_attn_floor 0.3 \
    --pose_gate_mode residual \
    --pose_gate_lambda 0.25 \
    --pose_gate_insert_stage pre_blocks

echo -e "\n\n==========================================================="
echo "  Both same-area experiments completed!"
echo "==========================================================="
echo -e "\n  [Results Summary]"

extract_results() {
    local dir=$1
    local name=$2
    local log_file
    log_file=$(ls -t "$dir"/*.log 2>/dev/null | head -n 1)
    if [ -f "$log_file" ]; then
        echo "--------------------------------------------------------"
        echo "Experiment: $name"
        grep -E "Use pose attention:|Pose Attention Config:" "$log_file" | tail -n 2 || true
        echo "Final Best Metrics:"
        grep "训练完成，最佳" "$log_file" || echo "Best metrics line not found."
        grep "评估结果摘要:" "$log_file" | tail -n 1 || echo "Evaluation summary line not found."
    else
        echo "Log file not found in $dir"
    fi
}

extract_results "$DIR_BASE" "same_mainline_baseline"
extract_results "$DIR_RES" "same_mainline_residual_l025"
echo "--------------------------------------------------------"

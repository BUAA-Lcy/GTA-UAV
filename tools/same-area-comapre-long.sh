#!/bin/bash
set -e

# ============================================================================
# 同区域长序列对比实验 (Same-area Compare Long)
# 
# 原理说明：
# 本脚本用于在 Game4Loc 的 UAV-VisLoc 数据集上进行 Same-area（同区域）设置的对比实验。
# Same-area 设置会将序列 03 和 04 的无人机图像混合，并随机划分为训练集和测试集。
#
# 本实验共包含 4 组对比，旨在评估不同的训练比例 (train_ratio) 和 训练轮数 (epochs) 
# 对 Baseline（基线模型）和 Residual0.25（带有残差姿态门控机制的模型）的影响。
# ============================================================================

# 固定参数（根据原脚本保持不变）
BATCH_SIZE=8
LR=1e-5

# Same-area 数据集的元文件
TRAIN_META="same-area-drone2sate-train.json"
TEST_META="same-area-drone2sate-test.json"

# 步骤 1：定义基础命令
# 原理：将所有实验共享的参数（如数据路径、batch_size、学习率等）提取出来，避免代码重复。
BASE_CMD="python Game4Loc/train_gta.py \
    --data_root Game4Loc/data/GTA-UAV-data \
    --train_pairs_meta_file $TRAIN_META \
    --test_pairs_meta_file $TEST_META \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --no_wandb"

echo "==========================================================="
echo "  Same-area Compare Long Experiments (4 Groups)"
echo "  Metafiles: train=$TRAIN_META, test=$TEST_META"
echo "==========================================================="

# ==========================================
# 步骤 2：执行 组 1 实验 (Baseline)
# 过程：设置 train_ratio=0.1, epochs=8，作为基础对比组。
# ==========================================
echo -e "\n\n>>> [1/4] Running Group 1: Baseline (train_ratio=0.1, epochs=8)..."
DIR_G1="logs/same_long_g1_baseline_r0.1_e8"
mkdir -p "$DIR_G1"
$BASE_CMD --train_ratio 0.1 --epochs 8 --log_path "$DIR_G1"

# ==========================================
# 步骤 3：执行 组 2 实验 (Residual0.25)
# 过程：在与组1相同的数据量和轮数下，加入姿态注意力残差门控机制（lambda=0.25）。
# 原理：验证在较少数据 (0.1) 和较长训练 (8 epochs) 的情况下，残差机制是否能带来性能提升。
# ==========================================
echo -e "\n\n>>> [2/4] Running Group 2: Residual 0.25 (train_ratio=0.1, epochs=8)..."
DIR_G2="logs/same_long_g2_residual_r0.1_e8"
mkdir -p "$DIR_G2"
$BASE_CMD --train_ratio 0.1 --epochs 8 --log_path "$DIR_G2" \
    --use_pose_attention \
    --pose_attn_floor 0.3 \
    --pose_gate_mode residual \
    --pose_gate_lambda 0.25 \
    --pose_gate_insert_stage pre_blocks

# ==========================================
# 步骤 4：执行 组 3 实验 (Baseline)
# 过程：设置 train_ratio=0.2, epochs=5。
# 原理：增加训练数据比例到 0.2，但减少轮数到 5，作为另一组基线。
# ==========================================
echo -e "\n\n>>> [3/4] Running Group 3: Baseline (train_ratio=0.2, epochs=5)..."
DIR_G3="logs/same_long_g3_baseline_r0.2_e5"
mkdir -p "$DIR_G3"
$BASE_CMD --train_ratio 0.2 --epochs 5 --log_path "$DIR_G3"

# ==========================================
# 步骤 5：执行 组 4 实验 (Residual0.25)
# 过程：在与组3相同的数据量和轮数下，同样加入残差姿态机制。
# 原理：验证在较多数据 (0.2) 和较短训练 (5 epochs) 的情况下，残差机制的表现。
# ==========================================
echo -e "\n\n>>> [4/4] Running Group 4: Residual 0.25 (train_ratio=0.2, epochs=5)..."
DIR_G4="logs/same_long_g4_residual_r0.2_e5"
mkdir -p "$DIR_G4"
$BASE_CMD --train_ratio 0.2 --epochs 5 --log_path "$DIR_G4" \
    --use_pose_attention \
    --pose_attn_floor 0.3 \
    --pose_gate_mode residual \
    --pose_gate_lambda 0.25 \
    --pose_gate_insert_stage pre_blocks

# ==========================================
# 步骤 6：提取并输出结果总结
# 原理：训练完成后，从各个实验的日志文件中提取最佳指标，方便直接对比这 4 组实验的效果。
# ==========================================
echo -e "\n\n==========================================================="
echo "  All 4 same-area experiments completed!"
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

extract_results "$DIR_G1" "Group 1: Baseline (ratio=0.1, ep=8)"
extract_results "$DIR_G2" "Group 2: Residual 0.25 (ratio=0.1, ep=8)"
extract_results "$DIR_G3" "Group 3: Baseline (ratio=0.2, ep=5)"
extract_results "$DIR_G4" "Group 4: Residual 0.25 (ratio=0.2, ep=5)"
echo "--------------------------------------------------------"

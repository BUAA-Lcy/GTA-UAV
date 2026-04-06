#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EPOCHS="${EPOCHS:-5}"
export EVAL_EVERY_N_EPOCH="${EVAL_EVERY_N_EPOCH:-1}"
export TRAIN_RATIO="${TRAIN_RATIO:-0.2}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export BATCH_SIZE_EVAL="${BATCH_SIZE_EVAL:-64}"
export LR="${LR:-1e-5}"
export PROB_FLIP="${PROB_FLIP:-0.0}"
export POSE_FOV_DEG="${POSE_FOV_DEG:-36}"
export POSE_ATTN_FLOOR="${POSE_ATTN_FLOOR:-0.3}"
export POSE_GATE_MODE="${POSE_GATE_MODE:-residual}"
export POSE_GATE_LAMBDA="${POSE_GATE_LAMBDA:-0.25}"
export POSE_GATE_INSERT_STAGE="${POSE_GATE_INSERT_STAGE:-pre_blocks}"
export RUN_NAME="${RUN_NAME:-visloc_pose_abcd_large_$(date +%Y%m%d_%H%M%S)}"

exec "${SCRIPT_DIR}/run_visloc_pose_smoke.sh"

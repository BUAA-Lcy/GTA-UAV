#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-vit_base_patch16_rope_reg1_gap_256.sbb_in1k}"
DATA_ROOT="${DATA_ROOT:-./data/UAV_VisLoc_dataset}"
TRAIN_META="${TRAIN_META:-same-area-drone2sate-train.json}"
TEST_META="${TEST_META:-same-area-drone2sate-test.json}"
GPU_IDS="${GPU_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
K_WEIGHT="${K_WEIGHT:-5}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-5}"
STAGE1_LR="${STAGE1_LR:-1e-4}"
STAGE1_CKPT="${STAGE1_CKPT:-./pretrained/gta/vit_base_eva_gta_same_area.pth}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
STAGE2_LR="${STAGE2_LR:-5e-5}"
WORK_ROOT="./work_dir/visloc/${MODEL}"

mkdir -p "$WORK_ROOT"

before_stage1="$(mktemp)"
after_stage1="$(mktemp)"
before_stage2="$(mktemp)"
after_stage2="$(mktemp)"
trap 'rm -f "$before_stage1" "$after_stage1" "$before_stage2" "$after_stage2"' EXIT

find "$WORK_ROOT" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort > "$before_stage1"

python train_visloc.py \
  --data_root "$DATA_ROOT" \
  --train_pairs_meta_file "$TRAIN_META" \
  --test_pairs_meta_file "$TEST_META" \
  --model "$MODEL" \
  --checkpoint_start "$STAGE1_CKPT" \
  --with_weight \
  --k "$K_WEIGHT" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$STAGE1_EPOCHS" \
  --train_ratio 1.0 \
  --lr "$STAGE1_LR" \
  --gpu_ids "$GPU_IDS" \
  --num_workers "$NUM_WORKERS"

find "$WORK_ROOT" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort > "$after_stage1"
stage1_run="$(comm -13 "$before_stage1" "$after_stage1" | tail -n 1)"
if [[ -z "$stage1_run" ]]; then
  stage1_run="$(find "$WORK_ROOT" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort | tail -n 1)"
fi

stage1_dir="${WORK_ROOT}/${stage1_run}"
stage1_best_name="$(find "$stage1_dir" -maxdepth 1 -type f -name 'weights_e*.pth' -printf '%f\n' | sort -t_ -k3,3Vr | head -n 1)"
if [[ -z "$stage1_best_name" ]]; then
  echo "Failed to locate stage-1 best checkpoint under ${stage1_dir}" >&2
  exit 1
fi

stage1_best="${stage1_dir}/${stage1_best_name}"

find "$WORK_ROOT" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort > "$before_stage2"

python train_visloc.py \
  --data_root "$DATA_ROOT" \
  --train_pairs_meta_file "$TRAIN_META" \
  --test_pairs_meta_file "$TEST_META" \
  --model "$MODEL" \
  --checkpoint_start "$stage1_best" \
  --with_weight \
  --k "$K_WEIGHT" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$STAGE2_EPOCHS" \
  --train_ratio 1.0 \
  --lr "$STAGE2_LR" \
  --gpu_ids "$GPU_IDS" \
  --num_workers "$NUM_WORKERS"

find "$WORK_ROOT" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort > "$after_stage2"
stage2_run="$(comm -13 "$before_stage2" "$after_stage2" | tail -n 1)"
if [[ -z "$stage2_run" ]]; then
  stage2_run="$(find "$WORK_ROOT" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort | tail -n 1)"
fi

printf 'stage1_dir=%s\n' "$stage1_dir"
printf 'stage1_best=%s\n' "$stage1_best"
printf 'stage2_dir=%s\n' "${WORK_ROOT}/${stage2_run}"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/lcy/miniconda3/envs/gtauav/bin/python}"
MODEL="${MODEL:-vit_base_patch16_rope_reg1_gap_256.sbb_in1k}"
DATA_ROOT="${DATA_ROOT:-./data/GTA-UAV-data}"
TEST_META="${TEST_META:-same-area-drone2sate-test.json}"
CHECKPOINT_START="${CHECKPOINT_START:-./pretrained/gta/vit_base_eva_gta_same_area.pth}"
TEACHER_CACHE="${TEACHER_CACHE:-./work_dir/gta_vop_same_area_runs/my_samearea_run_overnight/artifacts/gta_samearea_teacher.pt}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GPU_IDS="${GPU_IDS:-0}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-6}"
ORIENTATION_TOPK="${ORIENTATION_TOPK:-4}"
RUN_TAG="${RUN_TAG:-gta_samearea_supervision_compare_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-./work_dir/gta_vop_samearea_supervision_compare_runs}"
RUN_DIR="${RUN_ROOT}/${RUN_TAG}"
LOG_DIR="${RUN_DIR}/logs"
CMD_DIR="${RUN_DIR}/commands"
ARTIFACT_DIR="${RUN_DIR}/artifacts"
SUMMARY_MD="${RUN_DIR}/summary.md"
STAGE_TSV="${RUN_DIR}/stage_status.tsv"

mkdir -p "$LOG_DIR" "$CMD_DIR" "$ARTIFACT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$TEACHER_CACHE" ]]; then
  echo "Teacher cache not found: $TEACHER_CACHE" >&2
  exit 1
fi

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

iso_timestamp() {
  date "+%Y-%m-%dT%H:%M:%S%z"
}

append_summary() {
  printf '%s\n' "$*" >> "$SUMMARY_MD"
}

extract_last_matching_line() {
  local pattern="$1"
  local file_path="$2"
  grep -E "$pattern" "$file_path" | tail -n 1 || true
}

extract_auto_log_path() {
  local stdout_log="$1"
  grep -E "自动日志路径:" "$stdout_log" | tail -n 1 | sed 's/.*自动日志路径: //' || true
}

write_command_file() {
  local stage_name="$1"
  shift
  local cmd_file="${CMD_DIR}/${stage_name}.sh"
  {
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    printf 'cd %q\n' "$ROOT_DIR"
    printf '%q ' "$@"
    echo
  } > "$cmd_file"
  chmod +x "$cmd_file"
}

run_stage() {
  local stage_name="$1"
  shift
  local stdout_log="${LOG_DIR}/${stage_name}.stdout.log"
  local app_log_txt="${LOG_DIR}/${stage_name}.app_log_path.txt"
  local start_epoch
  local end_epoch
  local elapsed
  local status
  local app_log_path
  local metric_line
  local ma_line
  local summary_line
  local robustness_line
  local match_stats_line
  local fine_runtime_line
  local vop_line
  local teacher_filter_line
  local pair_weight_line
  local split_line

  write_command_file "$stage_name" "$@"

  start_epoch="$(date +%s)"
  {
    echo "[${stage_name}] START $(timestamp)"
    echo "[${stage_name}] cwd=${ROOT_DIR}"
    echo "[${stage_name}] command_file=${CMD_DIR}/${stage_name}.sh"
    printf '[%s] command=' "$stage_name"
    printf '%q ' "$@"
    echo
  } | tee "$stdout_log"

  set +e
  "$@" 2>&1 | tee -a "$stdout_log"
  status=${PIPESTATUS[0]}
  set -e

  end_epoch="$(date +%s)"
  elapsed="$((end_epoch - start_epoch))"

  app_log_path="$(extract_auto_log_path "$stdout_log")"
  if [[ -n "$app_log_path" ]]; then
    printf '%s\n' "$app_log_path" > "$app_log_txt"
  fi

  metric_line="$(extract_last_matching_line 'Recall@1:|mAP:|Dis@1:' "$stdout_log")"
  ma_line="$(extract_last_matching_line 'MA@3m:|MA@5m:|MA@10m:|MA@20m:' "$stdout_log")"
  summary_line="$(extract_last_matching_line '评估结果摘要:' "$stdout_log")"
  robustness_line="$(extract_last_matching_line '稳健性统计\(按查询汇总\):' "$stdout_log")"
  match_stats_line="$(extract_last_matching_line '最终匹配统计\(按查询汇总\):' "$stdout_log")"
  fine_runtime_line="$(extract_last_matching_line '细定位耗时\(按查询汇总\):' "$stdout_log")"
  vop_line="$(extract_last_matching_line 'VOP 摘要:' "$stdout_log")"
  teacher_filter_line="$(extract_last_matching_line 'Teacher filtering summary:' "$stdout_log")"
  pair_weight_line="$(extract_last_matching_line 'Pair-weight summary:' "$stdout_log")"
  split_line="$(extract_last_matching_line 'Dataset split summary:' "$stdout_log")"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$stage_name" \
    "$status" \
    "$(iso_timestamp)" \
    "$elapsed" \
    "$stdout_log" \
    "${CMD_DIR}/${stage_name}.sh" \
    "${app_log_path:-}" >> "$STAGE_TSV"

  append_summary "## ${stage_name}"
  append_summary "- status: ${status}"
  append_summary "- elapsed_sec: ${elapsed}"
  append_summary "- stdout_log: \`${stdout_log}\`"
  append_summary "- command_file: \`${CMD_DIR}/${stage_name}.sh\`"
  if [[ -n "$app_log_path" ]]; then
    append_summary "- app_log: \`${app_log_path}\`"
  fi
  if [[ -n "$teacher_filter_line" ]]; then
    append_summary "- teacher_filter_line: \`${teacher_filter_line}\`"
  fi
  if [[ -n "$pair_weight_line" ]]; then
    append_summary "- pair_weight_line: \`${pair_weight_line}\`"
  fi
  if [[ -n "$split_line" ]]; then
    append_summary "- split_line: \`${split_line}\`"
  fi
  if [[ -n "$metric_line" ]]; then
    append_summary "- metric_line: \`${metric_line}\`"
  fi
  if [[ -n "$ma_line" ]]; then
    append_summary "- ma_line: \`${ma_line}\`"
  fi
  if [[ -n "$summary_line" ]]; then
    append_summary "- summary_line: \`${summary_line}\`"
  fi
  if [[ -n "$robustness_line" ]]; then
    append_summary "- robustness_line: \`${robustness_line}\`"
  fi
  if [[ -n "$match_stats_line" ]]; then
    append_summary "- match_stats_line: \`${match_stats_line}\`"
  fi
  if [[ -n "$fine_runtime_line" ]]; then
    append_summary "- fine_runtime_line: \`${fine_runtime_line}\`"
  fi
  if [[ -n "$vop_line" ]]; then
    append_summary "- vop_line: \`${vop_line}\`"
  fi
  append_summary ""

  if [[ "$status" -ne 0 ]]; then
    echo "[${stage_name}] FAILED with exit code ${status}. See ${stdout_log}" >&2
    exit "$status"
  fi
}

{
  echo "# GTA-UAV Same-Area Supervision Compare"
  echo
  echo "- started_at: $(iso_timestamp)"
  echo "- run_dir: \`${RUN_DIR}\`"
  echo "- python_bin: \`${PYTHON_BIN}\`"
  echo "- model: \`${MODEL}\`"
  echo "- data_root: \`${DATA_ROOT}\`"
  echo "- test_meta: \`${TEST_META}\`"
  echo "- checkpoint_start: \`${CHECKPOINT_START}\`"
  echo "- teacher_cache: \`${TEACHER_CACHE}\`"
  echo "- train_batch_size: \`${TRAIN_BATCH_SIZE}\`"
  echo "- eval_batch_size: \`${EVAL_BATCH_SIZE}\`"
  echo "- num_workers: \`${NUM_WORKERS}\`"
  echo "- train_epochs: \`${TRAIN_EPOCHS}\`"
  echo "- orientation_topk: \`${ORIENTATION_TOPK}\`"
  echo
} > "$SUMMARY_MD"

printf 'stage\tstatus\tfinished_at\telapsed_sec\tstdout_log\tcommand_file\tapp_log_path\n' > "$STAGE_TSV"

git branch --show-current > "${RUN_DIR}/git_branch.txt"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short > "${RUN_DIR}/git_status.txt"
git diff > "${RUN_DIR}/git_diff.txt"
cp "$0" "${RUN_DIR}/run_script_snapshot.sh"

EXP_A_CKPT="${ARTIFACT_DIR}/exp_a_current_teacher_e${TRAIN_EPOCHS}.pth"
EXP_B_CKPT="${ARTIFACT_DIR}/exp_b_clean30_e${TRAIN_EPOCHS}.pth"
EXP_C_CKPT="${ARTIFACT_DIR}/exp_c_useful5_weight30_e${TRAIN_EPOCHS}.pth"

baseline_cmd=(
  env WANDB_MODE=disabled
  "$PYTHON_BIN" eval_gta.py
  --data_root "$DATA_ROOT"
  --test_pairs_meta_file "$TEST_META"
  --model "$MODEL"
  --checkpoint_start "$CHECKPOINT_START"
  --with_match
  --sparse
  --num_workers "$NUM_WORKERS"
  --batch_size "$EVAL_BATCH_SIZE"
  --gpu_ids "$GPU_IDS"
  --orientation_mode off
)

train_exp_a_cmd=(
  env WANDB_MODE=disabled
  "$PYTHON_BIN" train_vop.py
  --teacher_cache "$TEACHER_CACHE"
  --output_path "$EXP_A_CKPT"
  --model "$MODEL"
  --checkpoint_start "$CHECKPOINT_START"
  --batch_size "$TRAIN_BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --epochs "$TRAIN_EPOCHS"
  --supervision_mode posterior
  --pair_weight_mode uniform
)

train_exp_b_cmd=(
  env WANDB_MODE=disabled
  "$PYTHON_BIN" train_vop.py
  --teacher_cache "$TEACHER_CACHE"
  --output_path "$EXP_B_CKPT"
  --model "$MODEL"
  --checkpoint_start "$CHECKPOINT_START"
  --batch_size "$TRAIN_BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --epochs "$TRAIN_EPOCHS"
  --supervision_mode posterior
  --pair_weight_mode uniform
  --filter_best_distance_max 30
)

train_exp_c_cmd=(
  env WANDB_MODE=disabled
  "$PYTHON_BIN" train_vop.py
  --teacher_cache "$TEACHER_CACHE"
  --output_path "$EXP_C_CKPT"
  --model "$MODEL"
  --checkpoint_start "$CHECKPOINT_START"
  --batch_size "$TRAIN_BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --epochs "$TRAIN_EPOCHS"
  --supervision_mode useful_bce
  --useful_delta_m 5
  --ce_weight 1.0
  --pair_weight_mode best_distance_sigmoid
  --pair_weight_center_m 30
  --pair_weight_scale_m 10
)

eval_exp_a_cmd=(
  env WANDB_MODE=disabled
  "$PYTHON_BIN" eval_gta.py
  --data_root "$DATA_ROOT"
  --test_pairs_meta_file "$TEST_META"
  --model "$MODEL"
  --checkpoint_start "$CHECKPOINT_START"
  --with_match
  --sparse
  --num_workers "$NUM_WORKERS"
  --batch_size "$EVAL_BATCH_SIZE"
  --gpu_ids "$GPU_IDS"
  --orientation_checkpoint "$EXP_A_CKPT"
  --orientation_mode prior_topk
  --orientation_topk "$ORIENTATION_TOPK"
)

eval_exp_b_cmd=(
  env WANDB_MODE=disabled
  "$PYTHON_BIN" eval_gta.py
  --data_root "$DATA_ROOT"
  --test_pairs_meta_file "$TEST_META"
  --model "$MODEL"
  --checkpoint_start "$CHECKPOINT_START"
  --with_match
  --sparse
  --num_workers "$NUM_WORKERS"
  --batch_size "$EVAL_BATCH_SIZE"
  --gpu_ids "$GPU_IDS"
  --orientation_checkpoint "$EXP_B_CKPT"
  --orientation_mode prior_topk
  --orientation_topk "$ORIENTATION_TOPK"
)

eval_exp_c_cmd=(
  env WANDB_MODE=disabled
  "$PYTHON_BIN" eval_gta.py
  --data_root "$DATA_ROOT"
  --test_pairs_meta_file "$TEST_META"
  --model "$MODEL"
  --checkpoint_start "$CHECKPOINT_START"
  --with_match
  --sparse
  --num_workers "$NUM_WORKERS"
  --batch_size "$EVAL_BATCH_SIZE"
  --gpu_ids "$GPU_IDS"
  --orientation_checkpoint "$EXP_C_CKPT"
  --orientation_mode prior_topk
  --orientation_topk "$ORIENTATION_TOPK"
)

run_stage eval_baseline "${baseline_cmd[@]}"
run_stage train_exp_a "${train_exp_a_cmd[@]}"
run_stage eval_exp_a "${eval_exp_a_cmd[@]}"
run_stage train_exp_b "${train_exp_b_cmd[@]}"
run_stage eval_exp_b "${eval_exp_b_cmd[@]}"
run_stage train_exp_c "${train_exp_c_cmd[@]}"
run_stage eval_exp_c "${eval_exp_c_cmd[@]}"

append_summary "## Final Notes"
append_summary "- finished_at: $(iso_timestamp)"
append_summary "- run_dir: \`${RUN_DIR}\`"
append_summary "- stage_status: \`${STAGE_TSV}\`"
append_summary ""

ln -sfn "$(basename "$RUN_DIR")" "${RUN_ROOT}/latest"

echo "Supervision comparison completed."
echo "Run directory: ${RUN_DIR}"
echo "Summary: ${SUMMARY_MD}"

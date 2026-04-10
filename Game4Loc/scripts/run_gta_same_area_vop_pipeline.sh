#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROFILE="${PROFILE:-overnight}"
PYTHON_BIN="${PYTHON_BIN:-/home/lcy/miniconda3/envs/gtauav/bin/python}"
MODEL="${MODEL:-vit_base_patch16_rope_reg1_gap_256.sbb_in1k}"
DATA_ROOT="${DATA_ROOT:-./data/GTA-UAV-data}"
TRAIN_META="${TRAIN_META:-same-area-drone2sate-train.json}"
TEST_META="${TEST_META:-same-area-drone2sate-test.json}"
CHECKPOINT_START="${CHECKPOINT_START:-./pretrained/gta/vit_base_eva_gta_same_area.pth}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GPU_IDS="${GPU_IDS:-0}"
ROTATE_STEP="${ROTATE_STEP:-10}"
TEMPERATURE_M="${TEMPERATURE_M:-25}"
USEFUL_DELTA_M="${USEFUL_DELTA_M:-5}"
PAIR_WEIGHT_CENTER_M="${PAIR_WEIGHT_CENTER_M:-30}"
PAIR_WEIGHT_SCALE_M="${PAIR_WEIGHT_SCALE_M:-10}"
ORIENTATION_TOPK="${ORIENTATION_TOPK:-4}"
EST_TEACHER_SEC_PER_QUERY="${EST_TEACHER_SEC_PER_QUERY:-3.1}"
TRAINING_RECIPE="${TRAINING_RECIPE:-exp_c_useful_weighted}"
RUN_TAG="${RUN_TAG:-gta_samearea_vop_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-./work_dir/gta_vop_same_area_runs}"
RUN_DIR="${RUN_ROOT}/${RUN_TAG}"
LOG_DIR="${RUN_DIR}/logs"
CMD_DIR="${RUN_DIR}/commands"
ARTIFACT_DIR="${RUN_DIR}/artifacts"
SUMMARY_MD="${RUN_DIR}/summary.md"
STAGE_TSV="${RUN_DIR}/stage_status.tsv"
METADATA_TXT="${RUN_DIR}/run_metadata.txt"
GIT_STATUS_TXT="${RUN_DIR}/git_status.txt"
GIT_DIFF_TXT="${RUN_DIR}/git_diff.txt"
SCRIPT_SNAPSHOT="${RUN_DIR}/run_script_snapshot.sh"

TEACHER_CACHE_PATH="${ARTIFACT_DIR}/gta_samearea_teacher.pt"

mkdir -p "$LOG_DIR" "$CMD_DIR" "$ARTIFACT_DIR"
cp "$0" "$SCRIPT_SNAPSHOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

case "$PROFILE" in
  smoke|overnight|full)
    ;;
  *)
    echo "Unsupported PROFILE: $PROFILE (expected: smoke, overnight, full)" >&2
    exit 1
    ;;
esac

if [[ -z "${TRAIN_EPOCHS+x}" ]]; then
  case "$PROFILE" in
    smoke) TRAIN_EPOCHS=1 ;;
    overnight|full) TRAIN_EPOCHS=6 ;;
  esac
fi

if [[ -z "${TEACHER_QUERY_LIMIT+x}" ]]; then
  case "$PROFILE" in
    smoke) TEACHER_QUERY_LIMIT=200 ;;
    overnight) TEACHER_QUERY_LIMIT=3000 ;;
    full) TEACHER_QUERY_LIMIT=0 ;;
  esac
fi

if [[ -z "${EVAL_QUERY_LIMIT+x}" ]]; then
  case "$PROFILE" in
    smoke) EVAL_QUERY_LIMIT=100 ;;
    overnight|full) EVAL_QUERY_LIMIT=0 ;;
  esac
fi

VOP_CHECKPOINT_PATH="${ARTIFACT_DIR}/gta_samearea_useful5_weight30_e${TRAIN_EPOCHS}.pth"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

iso_timestamp() {
  date "+%Y-%m-%dT%H:%M:%S%z"
}

append_summary() {
  printf '%s\n' "$*" >> "$SUMMARY_MD"
}

extract_auto_log_path() {
  local stdout_log="$1"
  grep -E "自动日志路径:" "$stdout_log" | tail -n 1 | sed 's/.*自动日志路径: //' || true
}

extract_last_matching_line() {
  local pattern="$1"
  local file_path="$2"
  grep -E "$pattern" "$file_path" | tail -n 1 || true
}

count_positive_queries() {
  "$PYTHON_BIN" - "$DATA_ROOT" "$TRAIN_META" <<'PY'
import json
import os
import sys

data_root, meta_file = sys.argv[1:3]
with open(os.path.join(data_root, meta_file), "r", encoding="utf-8") as f:
    items = json.load(f)
count = 0
for item in items:
    if item.get("pair_pos_sate_img_list") or []:
        count += 1
print(count)
PY
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

TOTAL_TEACHER_QUERIES="$(count_positive_queries)"
if [[ "$TEACHER_QUERY_LIMIT" == "0" ]]; then
  EFFECTIVE_TEACHER_QUERIES="$TOTAL_TEACHER_QUERIES"
else
  if (( TEACHER_QUERY_LIMIT < TOTAL_TEACHER_QUERIES )); then
    EFFECTIVE_TEACHER_QUERIES="$TEACHER_QUERY_LIMIT"
  else
    EFFECTIVE_TEACHER_QUERIES="$TOTAL_TEACHER_QUERIES"
  fi
fi
ESTIMATED_TEACHER_HOURS="$(awk -v n="$EFFECTIVE_TEACHER_QUERIES" -v s="$EST_TEACHER_SEC_PER_QUERY" 'BEGIN { printf "%.2f", (n * s) / 3600.0 }')"

echo "Resolved profile=${PROFILE}"
echo "Resolved training recipe=${TRAINING_RECIPE}"
echo "Teacher subset: ${EFFECTIVE_TEACHER_QUERIES}/${TOTAL_TEACHER_QUERIES} queries"
echo "Rough build_teacher time estimate: ${ESTIMATED_TEACHER_HOURS} hours (using ${EST_TEACHER_SEC_PER_QUERY}s/query)"

if [[ "$TRAINING_RECIPE" != "exp_c_useful_weighted" ]]; then
  echo "Unsupported TRAINING_RECIPE: $TRAINING_RECIPE (expected: exp_c_useful_weighted)" >&2
  exit 1
fi

{
  echo "# GTA-UAV Same-Area VOP Pipeline"
  echo
  echo "- started_at: $(iso_timestamp)"
  echo "- profile: \`${PROFILE}\`"
  echo "- run_dir: \`${RUN_DIR}\`"
  echo "- python_bin: \`${PYTHON_BIN}\`"
  echo "- model: \`${MODEL}\`"
  echo "- data_root: \`${DATA_ROOT}\`"
  echo "- train_meta: \`${TRAIN_META}\`"
  echo "- test_meta: \`${TEST_META}\`"
  echo "- checkpoint_start: \`${CHECKPOINT_START}\`"
  echo "- train_batch_size: \`${TRAIN_BATCH_SIZE}\`"
  echo "- eval_batch_size: \`${EVAL_BATCH_SIZE}\`"
  echo "- num_workers: \`${NUM_WORKERS}\`"
  echo "- train_epochs: \`${TRAIN_EPOCHS}\`"
  echo "- training_recipe: \`${TRAINING_RECIPE}\`"
  echo "- total_teacher_queries: \`${TOTAL_TEACHER_QUERIES}\`"
  echo "- effective_teacher_queries: \`${EFFECTIVE_TEACHER_QUERIES}\`"
  echo "- estimated_teacher_hours: \`${ESTIMATED_TEACHER_HOURS}\`"
  echo "- teacher_query_limit: \`${TEACHER_QUERY_LIMIT}\`"
  echo "- eval_query_limit: \`${EVAL_QUERY_LIMIT}\`"
  echo "- orientation_topk: \`${ORIENTATION_TOPK}\`"
  echo
  echo "## Artifacts"
  echo "- teacher_cache: \`${TEACHER_CACHE_PATH}\`"
  echo "- vop_checkpoint: \`${VOP_CHECKPOINT_PATH}\`"
  echo
} > "$SUMMARY_MD"

printf 'stage\tstatus\tfinished_at\telapsed_sec\tstdout_log\tcommand_file\tapp_log_path\n' > "$STAGE_TSV"

{
  echo "run_tag=${RUN_TAG}"
  echo "profile=${PROFILE}"
  echo "run_dir=${RUN_DIR}"
  echo "root_dir=${ROOT_DIR}"
  echo "python_bin=${PYTHON_BIN}"
  echo "model=${MODEL}"
  echo "data_root=${DATA_ROOT}"
  echo "train_meta=${TRAIN_META}"
  echo "test_meta=${TEST_META}"
  echo "checkpoint_start=${CHECKPOINT_START}"
  echo "train_batch_size=${TRAIN_BATCH_SIZE}"
  echo "eval_batch_size=${EVAL_BATCH_SIZE}"
  echo "num_workers=${NUM_WORKERS}"
  echo "train_epochs=${TRAIN_EPOCHS}"
  echo "training_recipe=${TRAINING_RECIPE}"
  echo "total_teacher_queries=${TOTAL_TEACHER_QUERIES}"
  echo "effective_teacher_queries=${EFFECTIVE_TEACHER_QUERIES}"
  echo "estimated_teacher_hours=${ESTIMATED_TEACHER_HOURS}"
  echo "rotate_step=${ROTATE_STEP}"
  echo "temperature_m=${TEMPERATURE_M}"
  echo "useful_delta_m=${USEFUL_DELTA_M}"
  echo "pair_weight_center_m=${PAIR_WEIGHT_CENTER_M}"
  echo "pair_weight_scale_m=${PAIR_WEIGHT_SCALE_M}"
  echo "orientation_topk=${ORIENTATION_TOPK}"
  echo "teacher_query_limit=${TEACHER_QUERY_LIMIT}"
  echo "eval_query_limit=${EVAL_QUERY_LIMIT}"
} > "$METADATA_TXT"

git branch --show-current > "${RUN_DIR}/git_branch.txt"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git status --short > "$GIT_STATUS_TXT"
git diff > "$GIT_DIFF_TXT"

teacher_cmd=(
  env WANDB_MODE=disabled
  "$PYTHON_BIN" build_vop_teacher.py
  --dataset gta
  --data_root "$DATA_ROOT"
  --pairs_meta_file "$TRAIN_META"
  --model "$MODEL"
  --checkpoint_start "$CHECKPOINT_START"
  --rotate_step "$ROTATE_STEP"
  --temperature_m "$TEMPERATURE_M"
  --output_path "$TEACHER_CACHE_PATH"
)
if [[ "$TEACHER_QUERY_LIMIT" != "0" ]]; then
  teacher_cmd+=(--query_limit "$TEACHER_QUERY_LIMIT")
fi

train_cmd=(
  env WANDB_MODE=disabled
  "$PYTHON_BIN" train_vop.py
  --teacher_cache "$TEACHER_CACHE_PATH"
  --output_path "$VOP_CHECKPOINT_PATH"
  --model "$MODEL"
  --checkpoint_start "$CHECKPOINT_START"
  --batch_size "$TRAIN_BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --epochs "$TRAIN_EPOCHS"
  --supervision_mode useful_bce
  --useful_delta_m "$USEFUL_DELTA_M"
  --ce_weight 1.0
  --pair_weight_mode best_distance_sigmoid
  --pair_weight_center_m "$PAIR_WEIGHT_CENTER_M"
  --pair_weight_scale_m "$PAIR_WEIGHT_SCALE_M"
)

eval_baseline_cmd=(
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
if [[ "$EVAL_QUERY_LIMIT" != "0" ]]; then
  eval_baseline_cmd+=(--query_limit "$EVAL_QUERY_LIMIT")
fi

eval_vop_cmd=(
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
  --orientation_checkpoint "$VOP_CHECKPOINT_PATH"
  --orientation_mode prior_topk
  --orientation_topk "$ORIENTATION_TOPK"
)
if [[ "$EVAL_QUERY_LIMIT" != "0" ]]; then
  eval_vop_cmd+=(--query_limit "$EVAL_QUERY_LIMIT")
fi

run_stage build_teacher "${teacher_cmd[@]}"
run_stage train_vop "${train_cmd[@]}"
run_stage eval_baseline "${eval_baseline_cmd[@]}"
run_stage eval_prior_topk "${eval_vop_cmd[@]}"

append_summary "## Final Notes"
append_summary "- finished_at: $(iso_timestamp)"
append_summary "- run_dir: \`${RUN_DIR}\`"
append_summary "- stage_status: \`${STAGE_TSV}\`"
append_summary "- git_status: \`${GIT_STATUS_TXT}\`"
append_summary "- git_diff: \`${GIT_DIFF_TXT}\`"
append_summary ""

ln -sfn "$(basename "$RUN_DIR")" "${RUN_ROOT}/latest"

echo "Pipeline completed."
echo "Run directory: ${RUN_DIR}"
echo "Summary: ${SUMMARY_MD}"

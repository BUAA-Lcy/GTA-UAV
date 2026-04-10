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
BASELINE_COMPARE_SUMMARY="${BASELINE_COMPARE_SUMMARY:-./work_dir/gta_vop_samearea_supervision_compare_runs/gta_samearea_supervision_compare_q2000_20260410/summary.md}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GPU_IDS="${GPU_IDS:-0}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-6}"
ORIENTATION_TOPK="${ORIENTATION_TOPK:-4}"
RUN_TAG="${RUN_TAG:-gta_samearea_exp_c_followup_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-./work_dir/gta_exp_c_followup_runs}"
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

append_teacher_stats() {
  local label="$1"
  local useful_delta="$2"
  local pair_weight_center="$3"
  local pair_weight_scale="$4"

  append_summary "### ${label} teacher_stats"
  "$PYTHON_BIN" - "$TEACHER_CACHE" "$useful_delta" "$pair_weight_center" "$pair_weight_scale" >> "$SUMMARY_MD" <<'PY'
import math
import sys
import torch
from statistics import mean, median

teacher_cache_path, useful_delta_s, center_s, scale_s = sys.argv[1:5]
useful_delta = float(useful_delta_s)
center = float(center_s)
scale = float(scale_s)

cache = torch.load(teacher_cache_path, map_location="cpu")
records = list(cache["records"])

def pct(sorted_vals, p):
    n = len(sorted_vals)
    idx = min(n - 1, max(0, int(round((n - 1) * p))))
    return sorted_vals[idx]

sizes = []
weights = []
for record in records:
    distances = [float(x) for x in record["distances_m"]]
    best = min(distances)
    sizes.append(sum(1 for x in distances if math.isfinite(x) and x <= best + useful_delta))
    best_distance = float(record.get("best_distance_m", float("inf")))
    if not math.isfinite(best_distance):
        weights.append(0.0)
    else:
        weights.append(1.0 / (1.0 + math.exp((best_distance - center) / max(scale, 1e-6))))

sizes_sorted = sorted(sizes)
weights_sorted = sorted(weights)
print(f"- teacher_records: {len(records)}")
print(
    "- useful_set_size_stats: "
    f"delta={useful_delta:.1f} "
    f"mean={mean(sizes):.4f} median={median(sizes):.4f} "
    f"min={min(sizes)} max={max(sizes)} "
    f"p25={pct(sizes_sorted, 0.25)} p75={pct(sizes_sorted, 0.75)} "
    f"ge2_ratio={sum(v >= 2 for v in sizes)/len(sizes):.4f} "
    f"ge3_ratio={sum(v >= 3 for v in sizes)/len(sizes):.4f} "
    f"ge4_ratio={sum(v >= 4 for v in sizes)/len(sizes):.4f}"
)
print(
    "- pair_weight_stats: "
    f"center={center:.1f} scale={scale:.1f} "
    f"mean={mean(weights):.6f} median={median(weights):.6f} "
    f"min={min(weights):.6f} max={max(weights):.6f} "
    f"p25={pct(weights_sorted, 0.25):.6f} p75={pct(weights_sorted, 0.75):.6f} "
    f"ge0.5_ratio={sum(v >= 0.5 for v in weights)/len(weights):.4f} "
    f"ge0.8_ratio={sum(v >= 0.8 for v in weights)/len(weights):.4f}"
)
PY
  append_summary ""
}

{
  echo "# GTA-UAV Same-Area Exp C Follow-up"
  echo
  echo "- started_at: $(iso_timestamp)"
  echo "- run_dir: \`${RUN_DIR}\`"
  echo "- python_bin: \`${PYTHON_BIN}\`"
  echo "- model: \`${MODEL}\`"
  echo "- data_root: \`${DATA_ROOT}\`"
  echo "- test_meta: \`${TEST_META}\`"
  echo "- checkpoint_start: \`${CHECKPOINT_START}\`"
  echo "- teacher_cache: \`${TEACHER_CACHE}\`"
  echo "- baseline_compare_summary: \`${BASELINE_COMPARE_SUMMARY}\`"
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

C1_CKPT="${ARTIFACT_DIR}/exp_c1_useful3_weight30_e${TRAIN_EPOCHS}.pth"
C2_CKPT="${ARTIFACT_DIR}/exp_c2_useful5_weight20_e${TRAIN_EPOCHS}.pth"

append_summary "## Baseline Reference"
append_summary "- source_summary: \`${BASELINE_COMPARE_SUMMARY}\`"
if [[ -f "$BASELINE_COMPARE_SUMMARY" ]]; then
  append_summary "- eval_exp_c_metric_line: \`$(extract_last_matching_line 'eval_exp_c|Dis@1:' "$BASELINE_COMPARE_SUMMARY")\`"
  append_summary "- eval_exp_c_section:"
  sed -n '/^## eval_exp_c$/,/^## Final Notes$/p' "$BASELINE_COMPARE_SUMMARY" >> "$SUMMARY_MD"
else
  append_summary "- warning: baseline summary file not found"
fi
append_summary ""

append_teacher_stats "exp_c_baseline" 5 30 10
append_teacher_stats "exp_c1_useful3" 3 30 10
append_teacher_stats "exp_c2_weight20" 5 20 10

train_c1_cmd=(
  env WANDB_MODE=disabled
  "$PYTHON_BIN" train_vop.py
  --teacher_cache "$TEACHER_CACHE"
  --output_path "$C1_CKPT"
  --model "$MODEL"
  --checkpoint_start "$CHECKPOINT_START"
  --batch_size "$TRAIN_BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --epochs "$TRAIN_EPOCHS"
  --supervision_mode useful_bce
  --useful_delta_m 3
  --ce_weight 1.0
  --pair_weight_mode best_distance_sigmoid
  --pair_weight_center_m 30
  --pair_weight_scale_m 10
)

eval_c1_cmd=(
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
  --orientation_checkpoint "$C1_CKPT"
  --orientation_mode prior_topk
  --orientation_topk "$ORIENTATION_TOPK"
)

train_c2_cmd=(
  env WANDB_MODE=disabled
  "$PYTHON_BIN" train_vop.py
  --teacher_cache "$TEACHER_CACHE"
  --output_path "$C2_CKPT"
  --model "$MODEL"
  --checkpoint_start "$CHECKPOINT_START"
  --batch_size "$TRAIN_BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --epochs "$TRAIN_EPOCHS"
  --supervision_mode useful_bce
  --useful_delta_m 5
  --ce_weight 1.0
  --pair_weight_mode best_distance_sigmoid
  --pair_weight_center_m 20
  --pair_weight_scale_m 10
)

eval_c2_cmd=(
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
  --orientation_checkpoint "$C2_CKPT"
  --orientation_mode prior_topk
  --orientation_topk "$ORIENTATION_TOPK"
)

run_stage train_c1 "${train_c1_cmd[@]}"
run_stage eval_c1 "${eval_c1_cmd[@]}"
run_stage train_c2 "${train_c2_cmd[@]}"
run_stage eval_c2 "${eval_c2_cmd[@]}"

append_summary "## Final Notes"
append_summary "- finished_at: $(iso_timestamp)"
append_summary "- run_dir: \`${RUN_DIR}\`"
append_summary "- stage_status: \`${STAGE_TSV}\`"
append_summary ""

ln -sfn "$(basename "$RUN_DIR")" "${RUN_ROOT}/latest"

echo "Exp C follow-up completed."
echo "Run directory: ${RUN_DIR}"
echo "Summary: ${SUMMARY_MD}"

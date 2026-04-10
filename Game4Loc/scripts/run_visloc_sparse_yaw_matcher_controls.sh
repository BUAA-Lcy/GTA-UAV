#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/lcy/miniconda3/envs/gtauav/bin/python}"
RUNS_ROOT="$ROOT_DIR/work_dir/visloc_sparse_yaw_matcher_control_runs"
RUN_TAG="${RUN_TAG:-visloc_sparse_yaw_matcher_control_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$RUNS_ROOT/$RUN_TAG"
LOG_DIR="$RUN_DIR/logs"
CMD_DIR="$RUN_DIR/commands"
METRIC_DIR="$RUN_DIR/metrics"

SMALL_QUERY_LIMIT="${SMALL_QUERY_LIMIT:-12}"
QUARTER_QUERY_LIMIT="${QUARTER_QUERY_LIMIT:-29}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GPU_IDS="${GPU_IDS:-0}"

mkdir -p "$RUNS_ROOT" "$RUN_DIR" "$LOG_DIR" "$CMD_DIR" "$METRIC_DIR"
ln -sfn "$RUN_DIR" "$RUNS_ROOT/latest"

git rev-parse --abbrev-ref HEAD > "$RUN_DIR/git_branch.txt" 2>/dev/null || true
git rev-parse HEAD > "$RUN_DIR/git_commit.txt" 2>/dev/null || true
git status --short > "$RUN_DIR/git_status.txt" 2>/dev/null || true

COMMON_ARGS=(
  "$PYTHON_BIN" eval_visloc.py
  --data_root ./data/UAV_VisLoc_dataset
  --test_pairs_meta_file same-area-drone2sate-test.json
  --test_mode pos
  --query_mode D2S
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k
  --checkpoint_start ./pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth
  --with_match
  --sparse
  --use_yaw
  --no_rotate
  --num_workers "$NUM_WORKERS"
  --batch_size "$BATCH_SIZE"
  --gpu_ids "$GPU_IDS"
  --sparse_save_final_vis False
)

write_cmd_script() {
  local outfile="$1"
  shift
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd $(printf '%q' "$ROOT_DIR")"
    echo -n "env WANDB_MODE=disabled"
    for arg in "$@"; do
      echo -n " $(printf '%q' "$arg")"
    done
    echo
  } > "$outfile"
  chmod +x "$outfile"
}

parse_metrics() {
  local source_log="$1"
  local output_json="$2"
  "$PYTHON_BIN" - "$source_log" "$output_json" <<'PY'
import json
import re
import sys
from pathlib import Path

source_log = Path(sys.argv[1])
output_json = Path(sys.argv[2])
text = source_log.read_text(encoding="utf-8", errors="ignore")

patterns = {
    "summary": re.compile(r"评估摘要：Recall@1=([0-9.]+)%, Recall@5=([0-9.]+)%, Recall@10=([0-9.]+)%, AP=([0-9.]+)%"),
    "dis": re.compile(r"Dis@1:\s*([0-9.]+)\s*-\s*Dis@3:\s*([0-9.]+)\s*-\s*Dis@5:\s*([0-9.]+)"),
    "ma": re.compile(r"FineLoc 阈值成功率: MA@3m=([0-9.]+)% MA@5m=([0-9.]+)% MA@10m=([0-9.]+)% MA@20m=([0-9.]+)%"),
    "robust": re.compile(r"FineLoc 稳健性统计: worse_than_coarse=([0-9]+)\(([0-9.]+)%\) fallback=([0-9]+)\(([0-9.]+)%\) identity_H_fallback=([0-9]+) out_of_bounds=([0-9]+) projection_invalid=([0-9]+)"),
    "match": re.compile(r"FineLoc 匹配统计: mean_hypotheses=([0-9.]+) mean_retained_matches=([0-9.]+) mean_inliers=([0-9.]+) mean_inlier_ratio=([0-9.]+)"),
    "runtime": re.compile(r"FineLoc 运行时间: mean_vop_forward=([0-9.]+)s/query mean_matcher=([0-9.]+)s/query mean_total=([0-9.]+)s/query"),
}

metrics = {}

m = patterns["summary"].search(text)
if m:
    metrics["Recall@1"] = float(m.group(1))
    metrics["Recall@5"] = float(m.group(2))
    metrics["Recall@10"] = float(m.group(3))
    metrics["AP"] = float(m.group(4))

m = patterns["dis"].search(text)
if m:
    metrics["Dis@1"] = float(m.group(1))
    metrics["Dis@3"] = float(m.group(2))
    metrics["Dis@5"] = float(m.group(3))

m = patterns["ma"].search(text)
if m:
    metrics["MA@3m"] = float(m.group(1))
    metrics["MA@5m"] = float(m.group(2))
    metrics["MA@10m"] = float(m.group(3))
    metrics["MA@20m"] = float(m.group(4))

m = patterns["robust"].search(text)
if m:
    metrics["worse_than_coarse_count"] = int(m.group(1))
    metrics["worse_than_coarse_ratio"] = float(m.group(2))
    metrics["fallback_count"] = int(m.group(3))
    metrics["fallback_ratio"] = float(m.group(4))
    metrics["identity_H_fallback_count"] = int(m.group(5))
    metrics["out_of_bounds_count"] = int(m.group(6))
    metrics["projection_invalid_count"] = int(m.group(7))

m = patterns["match"].search(text)
if m:
    metrics["mean_hypotheses"] = float(m.group(1))
    metrics["mean_retained_matches"] = float(m.group(2))
    metrics["mean_inliers"] = float(m.group(3))
    metrics["mean_inlier_ratio"] = float(m.group(4))

m = patterns["runtime"].search(text)
if m:
    metrics["mean_vop_forward_s"] = float(m.group(1))
    metrics["mean_matcher_s"] = float(m.group(2))
    metrics["mean_total_s"] = float(m.group(3))

output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

run_stage() {
  local stage_name="$1"
  local query_limit="$2"
  shift 2

  local cmd_file="$CMD_DIR/${stage_name}.sh"
  local stdout_log="$LOG_DIR/${stage_name}.stdout.log"
  local app_log_file="$LOG_DIR/${stage_name}.app_log_path.txt"
  local metrics_json="$METRIC_DIR/${stage_name}.json"

  local cmd=( "${COMMON_ARGS[@]}" --query_limit "$query_limit" "$@" )
  write_cmd_script "$cmd_file" "${cmd[@]}"

  sleep 1
  echo "[$stage_name] START $(date '+%Y-%m-%d %H:%M:%S')" | tee "$stdout_log"
  echo "[$stage_name] query_limit=$query_limit" | tee -a "$stdout_log"
  echo "[$stage_name] command_file=$cmd_file" | tee -a "$stdout_log"
  env WANDB_MODE=disabled "${cmd[@]}" 2>&1 | tee -a "$stdout_log"

  local app_log
  app_log="$(sed -n 's/^.*自动日志路径: //p' "$stdout_log" | tail -n 1)"
  if [[ -n "$app_log" && -f "$app_log" ]]; then
    printf '%s\n' "$app_log" > "$app_log_file"
    parse_metrics "$app_log" "$metrics_json"
  else
    printf '%s\n' "$stdout_log" > "$app_log_file"
    parse_metrics "$stdout_log" "$metrics_json"
  fi
}

run_stage baseline_small "$SMALL_QUERY_LIMIT"
run_stage baseline_quarter "$QUARTER_QUERY_LIMIT"

run_stage lg_official_small "$SMALL_QUERY_LIMIT" \
  --sparse_lightglue_profile official_default
run_stage lg_official_quarter "$QUARTER_QUERY_LIMIT" \
  --sparse_lightglue_profile official_default

run_stage ms_query_only_small "$SMALL_QUERY_LIMIT" \
  --sparse_multi_scale_mode query_only
run_stage ms_query_only_quarter "$QUARTER_QUERY_LIMIT" \
  --sparse_multi_scale_mode query_only

run_stage ms_gallery_only_small "$SMALL_QUERY_LIMIT" \
  --sparse_multi_scale_mode gallery_only
run_stage ms_gallery_only_quarter "$QUARTER_QUERY_LIMIT" \
  --sparse_multi_scale_mode gallery_only

run_stage dedup5_small "$SMALL_QUERY_LIMIT" \
  --sparse_cross_scale_dedup_radius 5
run_stage dedup5_quarter "$QUARTER_QUERY_LIMIT" \
  --sparse_cross_scale_dedup_radius 5

"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
metric_dir = run_dir / "metrics"
summary_path = run_dir / "summary.md"

sections = [
    ("Small subset", [
        ("baseline_small", "baseline"),
        ("lg_official_small", "lg_official"),
        ("ms_query_only_small", "ms_query_only"),
        ("ms_gallery_only_small", "ms_gallery_only"),
        ("dedup5_small", "dedup5"),
    ]),
    ("Quarter subset", [
        ("baseline_quarter", "baseline"),
        ("lg_official_quarter", "lg_official"),
        ("ms_query_only_quarter", "ms_query_only"),
        ("ms_gallery_only_quarter", "ms_gallery_only"),
        ("dedup5_quarter", "dedup5"),
    ]),
]

lines = [
    "# VisLoc Sparse Yaw Matcher Controlled Experiments",
    "",
    "Protocol: UAV-VisLoc same-area, `test_mode=pos`, `query_mode=D2S`, `with_match+sparse`, `use_yaw`, no VOP, no extra rotation search.",
    "",
]

for title, items in sections:
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| Variant | Recall@1 | AP | Dis@1 | MA@5m | MA@10m | MA@20m | fallback% | worse% | retained | inliers | ratio | total_s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for filename, label in items:
        metrics_path = metric_dir / f"{filename}.json"
        if not metrics_path.exists():
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        lines.append(
            "| {label} | {r1:.2f} | {ap:.2f} | {d1:.2f} | {ma5:.2f} | {ma10:.2f} | {ma20:.2f} | {fb:.2f} | {wc:.2f} | {ret:.1f} | {inl:.1f} | {ratio:.4f} | {total:.6f} |".format(
                label=label,
                r1=data.get("Recall@1", float("nan")),
                ap=data.get("AP", float("nan")),
                d1=data.get("Dis@1", float("nan")),
                ma5=data.get("MA@5m", float("nan")),
                ma10=data.get("MA@10m", float("nan")),
                ma20=data.get("MA@20m", float("nan")),
                fb=data.get("fallback_ratio", float("nan")),
                wc=data.get("worse_than_coarse_ratio", float("nan")),
                ret=data.get("mean_retained_matches", float("nan")),
                inl=data.get("mean_inliers", float("nan")),
                ratio=data.get("mean_inlier_ratio", float("nan")),
                total=data.get("mean_total_s", float("nan")),
            )
        )
    lines.append("")

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "Run directory: $RUN_DIR"
echo "Summary: $RUN_DIR/summary.md"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/lcy/miniconda3/envs/gtauav/bin/python}"
RUNS_ROOT="$ROOT_DIR/work_dir/visloc_sparse_yaw_scale_contrib_runs"
RUN_TAG="${RUN_TAG:-visloc_sparse_yaw_scale_contrib_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$RUNS_ROOT/$RUN_TAG"
LOG_DIR="$RUN_DIR/logs"
CMD_DIR="$RUN_DIR/commands"
METRIC_DIR="$RUN_DIR/metrics"
SMALL_QUERY_LIMIT="${SMALL_QUERY_LIMIT:-12}"
QUARTER_QUERY_LIMIT="${QUARTER_QUERY_LIMIT:-29}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GPU_IDS="${GPU_IDS:-0}"
TOPK_QUARTER="${TOPK_QUARTER:-2}"

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
scale_pattern = re.compile(
    r"Sparse尺度贡献: label=([^ ]+) q_scale=([0-9.]+) g_scale=([0-9.]+) selected_queries=([0-9]+) matched_queries=([0-9]+) inlier_queries=([0-9]+) mean_retained_matches=([0-9.]+) mean_inliers=([0-9.]+) inlier_share=([0-9.]+)"
)

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

scale_rows = []
for match in scale_pattern.finditer(text):
    scale_rows.append(
        {
            "label": match.group(1),
            "q_scale": float(match.group(2)),
            "g_scale": float(match.group(3)),
            "selected_queries": int(match.group(4)),
            "matched_queries": int(match.group(5)),
            "inlier_queries": int(match.group(6)),
            "mean_retained_matches": float(match.group(7)),
            "mean_inliers": float(match.group(8)),
            "inlier_share": float(match.group(9)),
        }
    )

metrics["scale_summary"] = scale_rows
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

# Stage 1: small-sample screening.
run_stage baseline_small "$SMALL_QUERY_LIMIT" \
  --sparse_scales 1.0,0.8,0.6,1.2

run_stage dense_down_small "$SMALL_QUERY_LIMIT" \
  --sparse_scales 1.0,0.9,0.8,0.7,0.6,0.5

run_stage true_up_small "$SMALL_QUERY_LIMIT" \
  --sparse_scales 1.0,0.8,0.6,1.2 \
  --sparse_allow_upsample True

run_stage dense_mix_up_small "$SMALL_QUERY_LIMIT" \
  --sparse_scales 1.0,0.9,0.8,0.7,0.6,1.2 \
  --sparse_allow_upsample True

"$PYTHON_BIN" - "$RUN_DIR" "$TOPK_QUARTER" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
topk_quarter = max(1, int(sys.argv[2]))
metric_dir = run_dir / "metrics"

small_variants = [
    ("baseline_small", "baseline"),
    ("dense_down_small", "dense_down"),
    ("true_up_small", "true_up"),
    ("dense_mix_up_small", "dense_mix_up"),
]

def load_metrics(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

rows = []
for filename, label in small_variants:
    data = load_metrics(metric_dir / f"{filename}.json")
    if not data:
        continue
    rows.append(
        {
            "filename": filename,
            "label": label,
            "dis1": float(data.get("Dis@1", math.inf)),
            "fallback": float(data.get("fallback_ratio", math.inf)),
            "ma20": -float(data.get("MA@20m", -math.inf)),
            "total": float(data.get("mean_total_s", math.inf)),
        }
    )

rows.sort(key=lambda row: (row["dis1"], row["fallback"], row["ma20"], row["total"], row["label"]))
selected = []
for row in rows:
    if row["label"] == "baseline":
        continue
    selected.append(row["label"])
    if len(selected) >= topk_quarter:
        break
if not selected:
    selected = ["dense_down"]

(run_dir / "selected_variants.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

mapfile -t SELECTED_VARIANTS < <("$PYTHON_BIN" - "$RUN_DIR/selected_variants.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
items = json.loads(path.read_text(encoding="utf-8"))
for item in items:
    print(str(item))
PY
)

# Stage 2: quarter-subset validation.
run_stage baseline_quarter "$QUARTER_QUERY_LIMIT" \
  --sparse_scales 1.0,0.8,0.6,1.2

for variant in "${SELECTED_VARIANTS[@]}"; do
  case "$variant" in
    dense_down)
      run_stage dense_down_quarter "$QUARTER_QUERY_LIMIT" \
        --sparse_scales 1.0,0.9,0.8,0.7,0.6,0.5
      ;;
    true_up)
      run_stage true_up_quarter "$QUARTER_QUERY_LIMIT" \
        --sparse_scales 1.0,0.8,0.6,1.2 \
        --sparse_allow_upsample True
      ;;
    dense_mix_up)
      run_stage dense_mix_up_quarter "$QUARTER_QUERY_LIMIT" \
        --sparse_scales 1.0,0.9,0.8,0.7,0.6,1.2 \
        --sparse_allow_upsample True
      ;;
  esac
done

"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
metric_dir = run_dir / "metrics"
summary_path = run_dir / "summary.md"
selected_variants = json.loads((run_dir / "selected_variants.json").read_text(encoding="utf-8"))

small_order = [
    ("baseline_small", "baseline", "1.0,0.8,0.6,1.2", False),
    ("dense_down_small", "dense_down", "1.0,0.9,0.8,0.7,0.6,0.5", False),
    ("true_up_small", "true_up", "1.0,0.8,0.6,1.2", True),
    ("dense_mix_up_small", "dense_mix_up", "1.0,0.9,0.8,0.7,0.6,1.2", True),
]
quarter_order = [("baseline_quarter", "baseline", "1.0,0.8,0.6,1.2", False)]
variant_meta = {
    "dense_down": ("dense_down_quarter", "dense_down", "1.0,0.9,0.8,0.7,0.6,0.5", False),
    "true_up": ("true_up_quarter", "true_up", "1.0,0.8,0.6,1.2", True),
    "dense_mix_up": ("dense_mix_up_quarter", "dense_mix_up", "1.0,0.9,0.8,0.7,0.6,1.2", True),
}
for variant in selected_variants:
    if variant in variant_meta:
        quarter_order.append(variant_meta[variant])

def load_metrics(name):
    path = metric_dir / f"{name}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def fmt(value, digits=2):
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(value) or math.isinf(value):
        return "NA"
    return f"{value:.{digits}f}"

def add_section(lines, title, rows):
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| Variant | Scales | Upsample | Dis@1 | MA@20m | fallback% | worse% | retained | inliers | total_s |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for filename, label, scales, allow_up in rows:
        data = load_metrics(filename)
        lines.append(
            "| {label} | `{scales}` | {allow_up} | {d1} | {ma20} | {fb} | {worse} | {ret} | {inl} | {total} |".format(
                label=label,
                scales=scales,
                allow_up="yes" if allow_up else "no",
                d1=fmt(data.get("Dis@1")),
                ma20=fmt(data.get("MA@20m")),
                fb=fmt(data.get("fallback_ratio")),
                worse=fmt(data.get("worse_than_coarse_ratio")),
                ret=fmt(data.get("mean_retained_matches")),
                inl=fmt(data.get("mean_inliers")),
                total=fmt(data.get("mean_total_s"), digits=6),
            )
        )
    lines.append("")
    for filename, label, scales, allow_up in rows:
        data = load_metrics(filename)
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"- scales: `{scales}`")
        lines.append(f"- allow_upsample: {'yes' if allow_up else 'no'}")
        lines.append("- scale contribution:")
        scale_rows = data.get("scale_summary", [])
        if not scale_rows:
            lines.append("  none")
            lines.append("")
            continue
        lines.append("")
        lines.append("| label | q_scale | g_scale | mean_retained | mean_inliers | inlier_share | inlier_queries |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for item in scale_rows:
            lines.append(
                "| {label} | {q_scale} | {g_scale} | {mean_retained} | {mean_inliers} | {inlier_share} | {inlier_queries} |".format(
                    label=item.get("label", "NA"),
                    q_scale=fmt(item.get("q_scale")),
                    g_scale=fmt(item.get("g_scale")),
                    mean_retained=fmt(item.get("mean_retained_matches")),
                    mean_inliers=fmt(item.get("mean_inliers")),
                    inlier_share=fmt(item.get("inlier_share"), digits=4),
                    inlier_queries=int(item.get("inlier_queries", 0)),
                )
            )
        lines.append("")

lines = [
    "# VisLoc Sparse Yaw Multi-Scale Contribution Ablation",
    "",
    "Protocol: UAV-VisLoc same-area, `test_mode=pos`, `query_mode=D2S`, `with_match+sparse`, `use_yaw`, `no VOP`, `rotate=0`.",
    "",
    "Screening policy: run 4 variants on the small subset, then promote the best non-baseline variants by `Dis@1 -> fallback -> MA@20 -> runtime` to the quarter subset.",
    "",
    f"Selected quarter variants: {', '.join(selected_variants) if selected_variants else 'none'}",
    "",
]

add_section(lines, "Small Subset", small_order)
add_section(lines, "Quarter Subset", quarter_order)

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "Run directory: $RUN_DIR"
echo "Summary: $RUN_DIR/summary.md"

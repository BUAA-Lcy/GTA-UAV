#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lcy/Workplace/GTA-UAV/Game4Loc"
PYTHON_BIN="/home/lcy/miniconda3/envs/gtauav/bin/python"

META_FILE=""
CHECKPOINT=""
RUN_DIR=""
SHARD_SIZE=500
START_OFFSET=0
END_OFFSET=0
BATCH_SIZE=32
GPU_IDS=0
NUM_WORKERS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --meta)
      META_FILE="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --shard-size)
      SHARD_SIZE="$2"
      shift 2
      ;;
    --start-offset)
      START_OFFSET="$2"
      shift 2
      ;;
    --end-offset)
      END_OFFSET="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --gpu-ids)
      GPU_IDS="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${META_FILE}" || -z "${CHECKPOINT}" || -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 --meta <meta.json> --checkpoint <ckpt.pth> --run-dir <dir> [--shard-size 500]" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/stdout"

TOTAL_QUERIES=$(
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
meta = Path("${ROOT_DIR}") / "data" / "GTA-UAV-data" / "${META_FILE}"
data = json.loads(meta.read_text())
print(sum(1 for item in data if item.get("pair_pos_sate_img_list")))
PY
)

if [[ "${END_OFFSET}" -le 0 || "${END_OFFSET}" -gt "${TOTAL_QUERIES}" ]]; then
  END_OFFSET="${TOTAL_QUERIES}"
fi

cd "${ROOT_DIR}"
echo "[runner] meta=${META_FILE} checkpoint=${CHECKPOINT} total_queries=${TOTAL_QUERIES} shard_size=${SHARD_SIZE} range=[${START_OFFSET}, ${END_OFFSET})"

offset="${START_OFFSET}"
while [[ "${offset}" -lt "${END_OFFSET}" ]]; do
  remaining=$((END_OFFSET - offset))
  count="${SHARD_SIZE}"
  if [[ "${remaining}" -lt "${count}" ]]; then
    count="${remaining}"
  fi

  app_log="${RUN_DIR}/logs/dense_offset$(printf '%05d' "${offset}")_count$(printf '%05d' "${count}").app.log"
  stdout_log="${RUN_DIR}/stdout/dense_offset$(printf '%05d' "${offset}")_count$(printf '%05d' "${count}").stdout.log"

  if [[ -f "${app_log}" ]] && grep -q "评估结果摘要:" "${app_log}"; then
    echo "[runner] skip completed shard offset=${offset} count=${count}"
  else
    echo "[runner] start shard offset=${offset} count=${count}"
    env WANDB_MODE=disabled "${PYTHON_BIN}" eval_gta.py \
      --data_root ./data/GTA-UAV-data \
      --test_pairs_meta_file "${META_FILE}" \
      --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
      --checkpoint_start "${CHECKPOINT}" \
      --with_match --dense --no_rotate \
      --num_workers "${NUM_WORKERS}" \
      --batch_size "${BATCH_SIZE}" \
      --gpu_ids "${GPU_IDS}" \
      --query_offset "${offset}" \
      --query_limit "${count}" \
      --log_to_file \
      --log_path "${app_log}" \
      > "${stdout_log}" 2>&1
  fi

  "${PYTHON_BIN}" ./scripts/merge_gta_eval_shards.py \
    --log_glob "${RUN_DIR}/logs/*.app.log" \
    --output_path "${RUN_DIR}/merged_summary.md" \
    >/dev/null || true

  offset=$((offset + count))
done

echo "[runner] finished all configured shards"
"${PYTHON_BIN}" ./scripts/merge_gta_eval_shards.py \
  --log_glob "${RUN_DIR}/logs/*.app.log" \
  --output_path "${RUN_DIR}/merged_summary.md"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAME4LOC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${GAME4LOC_DIR}/.." && pwd)"

CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-gtauav}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_HOME}/envs/${CONDA_ENV_NAME}/bin/python}"

DATA_ROOT="${DATA_ROOT:-${GAME4LOC_DIR}/data/UAV_VisLoc_dataset}"
TRAIN_JSON="${TRAIN_JSON:-same-area-drone2sate-train-pose.json}"
TEST_JSON="${TEST_JSON:-same-area-drone2sate-test-pose.json}"

MODEL="${MODEL:-vit_base_patch16_rope_reg1_gap_256.sbb_in1k}"
GPU_IDS="${GPU_IDS:-0}"
EPOCHS="${EPOCHS:-1}"
EVAL_EVERY_N_EPOCH="${EVAL_EVERY_N_EPOCH:-1}"
TRAIN_RATIO="${TRAIN_RATIO:-0.05}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BATCH_SIZE_EVAL="${BATCH_SIZE_EVAL:-64}"
LR="${LR:-1e-4}"
PROB_FLIP="${PROB_FLIP:-0.0}"
POSE_FOV_DEG="${POSE_FOV_DEG:-36}"
POSE_ATTN_FLOOR="${POSE_ATTN_FLOOR:-0.3}"
POSE_GATE_MODE="${POSE_GATE_MODE:-residual}"
POSE_GATE_LAMBDA="${POSE_GATE_LAMBDA:-0.25}"
POSE_GATE_INSERT_STAGE="${POSE_GATE_INSERT_STAGE:-pre_blocks}"

RUN_NAME="${RUN_NAME:-visloc_pose_smoke_abcd_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${GAME4LOC_DIR}/work_dir/${RUN_NAME}}"
LOG_ROOT="${RUN_ROOT}/logs"
CKPT_ROOT="${RUN_ROOT}/checkpoints"
SUMMARY_FILE="${RUN_ROOT}/summary.txt"
RUNNER_LOG="${RUN_ROOT}/runner.log"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${CKPT_ROOT}"
export MPLCONFIGDIR="${RUN_ROOT}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

exec > >(tee -a "${RUNNER_LOG}") 2>&1

echo "Run root: ${RUN_ROOT}"
echo "Log root: ${LOG_ROOT}"
echo "Checkpoint root: ${CKPT_ROOT}"
echo "Python: ${PYTHON_BIN}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -f "${DATA_ROOT}/${TRAIN_JSON}" || ! -f "${DATA_ROOT}/${TEST_JSON}" ]]; then
    echo "Pose JSON not found. Generating pose-enriched metadata under ${DATA_ROOT}"
    TRAIN_JSON_RAW="${TRAIN_JSON/-pose.json/.json}"
    TEST_JSON_RAW="${TEST_JSON/-pose.json/.json}"
    python3 "${REPO_ROOT}/scripts/prepare_dataset/enrich_visloc_pose_metadata.py" \
        --data_root "${DATA_ROOT}" \
        --input_json "${TRAIN_JSON_RAW}" "${TEST_JSON_RAW}"
fi

COMMON_ARGS=(
    --data_root "${DATA_ROOT}"
    --train_pairs_meta_file "${TRAIN_JSON}"
    --test_pairs_meta_file "${TEST_JSON}"
    --model "${MODEL}"
    --epochs "${EPOCHS}"
    --eval_every_n_epoch "${EVAL_EVERY_N_EPOCH}"
    --train_ratio "${TRAIN_RATIO}"
    --batch_size "${BATCH_SIZE}"
    --batch_size_eval "${BATCH_SIZE_EVAL}"
    --gpu_ids "${GPU_IDS}"
    --lr "${LR}"
    --prob_flip "${PROB_FLIP}"
    --pose_fov_deg "${POSE_FOV_DEG}"
    --pose_attn_floor "${POSE_ATTN_FLOOR}"
    --no_zero_shot
    --no_wandb
)

run_experiment() {
    local name="$1"
    shift

    local exp_log_dir="${LOG_ROOT}/${name}"
    local exp_ckpt_root="${CKPT_ROOT}/${name}"
    mkdir -p "${exp_log_dir}" "${exp_ckpt_root}"

    echo
    echo "===== Running ${name} ====="
    "${PYTHON_BIN}" "${GAME4LOC_DIR}/train_visloc_pose.py" \
        "${COMMON_ARGS[@]}" \
        --log_path "${exp_log_dir}" \
        --model_path "${exp_ckpt_root}" \
        "$@"

    local latest_log
    latest_log="$(find "${exp_log_dir}" -maxdepth 1 -type f -name '*.log' | sort | tail -n 1 || true)"
    local latest_weight
    latest_weight="$(find "${exp_ckpt_root}" -type f -name 'weights_end.pth' | sort | tail -n 1 || true)"

    {
        echo "[${name}]"
        echo "log_dir=${exp_log_dir}"
        echo "latest_log=${latest_log}"
        echo "checkpoint_root=${exp_ckpt_root}"
        echo "weights_end=${latest_weight}"
        echo
    } >> "${SUMMARY_FILE}"

    echo "Latest log: ${latest_log}"
    echo "weights_end: ${latest_weight}"
}

: > "${SUMMARY_FILE}"
{
    echo "Run root: ${RUN_ROOT}"
    echo "Log root: ${LOG_ROOT}"
    echo "Checkpoint root: ${CKPT_ROOT}"
    echo "Runner log: ${RUNNER_LOG}"
    echo
    echo "Config:"
    echo "train_json=${TRAIN_JSON}"
    echo "test_json=${TEST_JSON}"
    echo "model=${MODEL}"
    echo "epochs=${EPOCHS}"
    echo "eval_every_n_epoch=${EVAL_EVERY_N_EPOCH}"
    echo "train_ratio=${TRAIN_RATIO}"
    echo "batch_size=${BATCH_SIZE}"
    echo "batch_size_eval=${BATCH_SIZE_EVAL}"
    echo "lr=${LR}"
    echo "pose_gate_mode=${POSE_GATE_MODE}"
    echo "pose_gate_lambda=${POSE_GATE_LAMBDA}"
    echo
    echo "Experiments:"
    echo "A_noalign_nopose=NoAlign + NoPose"
    echo "B_noalign_pose=NoAlign + Pose"
    echo "C_align_nopose=Align + NoPose"
    echo "D_align_pose=Align + Pose"
    echo
} >> "${SUMMARY_FILE}"

run_experiment \
    A_noalign_nopose

run_experiment \
    B_noalign_pose \
    --use_pose_attention \
    --pose_gate_mode "${POSE_GATE_MODE}" \
    --pose_gate_lambda "${POSE_GATE_LAMBDA}"

run_experiment \
    C_align_nopose \
    --rotate_query_to_north

run_experiment \
    D_align_pose \
    --rotate_query_to_north \
    --use_pose_attention \
    --pose_gate_mode "${POSE_GATE_MODE}" \
    --pose_gate_lambda "${POSE_GATE_LAMBDA}"

echo
"${PYTHON_BIN}" "${GAME4LOC_DIR}/scripts/rebuild_visloc_run_summary.py" \
    --run_root "${RUN_ROOT}" \
    --summary_file "${SUMMARY_FILE}"

echo
echo "All experiments finished."
echo "Summary file: ${SUMMARY_FILE}"
echo "Runner log: ${RUNNER_LOG}"
echo "Per-experiment logs live under: ${LOG_ROOT}"

#!/usr/bin/env bash
# Fine-tune Phi_0 on SIMPLE G1 whole-body LeRobot data (Psi0 finetune-simple aligned).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

TASK="${TASK:-G1WholebodyBendPick-v0-psi0}"
EXP="${EXP:-experiments/simple_g1_act}"
CONFIG="${CONFIG:-train_simple_g1_act}"
CKPT_NAME="${CKPT_NAME:-simple_g1_act}"
DATA_ROOT="${SIMPLE_DATA_ROOT:-./data/simple}"
NGPUS="${NGPUS:-1}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR_ACTION="${LR_ACTION:-1e-4}"
MAX_STEPS="${MAX_STEPS:-40000}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONPATH="${ROOT}/src:${ROOT}/third_party/SIMPLE/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${EXP}"

echo "==> SIMPLE G1 fine-tune"
echo "    task=${TASK} data=${DATA_ROOT}"
echo "    config=${CONFIG} output=${EXP}"
echo "    gpus=${NGPUS} batch=${BATCH_SIZE} steps=${MAX_STEPS}"

TRAIN_ARGS=(
  scripts/train.py
  --config-name "${CONFIG}"
  output_dir="${EXP}"
  checkpoint_name="${CKPT_NAME}"
  data.simple_root="${DATA_ROOT}"
  data.simple_repo_id="${TASK}"
  batch_size="${BATCH_SIZE}"
  max_steps="${MAX_STEPS}"
  learning_rate_action="${LR_ACTION}"
)

if [[ "${NGPUS}" -gt 1 ]]; then
  exec torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NGPUS}" \
    "${TRAIN_ARGS[@]}" \
    distributed=true \
    "$@"
else
  exec python "${TRAIN_ARGS[@]}" "$@"
fi

#!/usr/bin/env bash
# Single-GPU LIBERO delta-EEF ablation: VLM + ACT only (no VGGT / dual cross-attn).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
EXP="${EXP:-experiments/libero_spatial_vlm_only_15k_single}"
CONFIG="${CONFIG:-train_libero_spatial_act_delta_15k_single_vlm_only}"
CKPT_NAME="${CKPT_NAME:-libero_spatial_vlm_only_15k_single}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR_ACTION="${LR_ACTION:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONPATH="${ROOT}/src:${ROOT}/../FastWAM/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

mkdir -p "${EXP}"

echo "==> Single-GPU VLM-only ablation: GPU ${CUDA_DEVICES}"
echo "    config=${CONFIG} output=${EXP}"
echo "    batch=${BATCH_SIZE} lr=${LR_ACTION} warmup=${WARMUP_STEPS} scheduler=${LR_SCHEDULER}"

EXTRA_ARGS=(
  learning_rate_action="${LR_ACTION}"
  lr_scale=none
  learning_rate_warmup_steps="${WARMUP_STEPS}"
  learning_rate_scheduler="${LR_SCHEDULER}"
  weight_decay=1e-6
  adam_beta1=0.95
  adam_beta2=0.999
)

exec conda run --no-capture-output -n Phi-0-wpy python scripts/train.py \
  --config-name "${CONFIG}" \
  output_dir="${EXP}" \
  checkpoint_name="${CKPT_NAME}" \
  batch_size="${BATCH_SIZE}" \
  distributed=false \
  auto_resume=false \
  "${EXTRA_ARGS[@]}" \
  "$@"

#!/usr/bin/env bash
# Ablation A: VLM + ACT only, 4-GPU DDP, 35k steps, lr=2e-4, bs=16/GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

NGPUS="${NGPUS:-4}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
EXP="${EXP:-experiments/libero_spatial_vlm_only_35k_ddp4}"
CONFIG="${CONFIG:-train_libero_spatial_vlm_only_35k_ddp4}"
CKPT_NAME="${CKPT_NAME:-libero_spatial_vlm_only_35k_ddp4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR_ACTION="${LR_ACTION:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
MAX_STEPS="${MAX_STEPS:-35000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="${ROOT}/src:${ROOT}/../FastWAM/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

mkdir -p "${EXP}"

echo "==> VLM-only 35k DDP: ${NGPUS} GPUs (${CUDA_DEVICES})"
echo "    config=${CONFIG} output=${EXP}"
echo "    per_device_batch=${BATCH_SIZE} effective_batch=$((BATCH_SIZE * NGPUS))"
echo "    max_steps=${MAX_STEPS} save_every=${SAVE_EVERY} overwrite=true"
echo "    lr=${LR_ACTION} warmup=${WARMUP_STEPS} scheduler=${LR_SCHEDULER}"

exec conda run --no-capture-output -n Phi-0-wpy torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${NGPUS}" \
  scripts/train.py \
  --config-name "${CONFIG}" \
  output_dir="${EXP}" \
  checkpoint_name="${CKPT_NAME}" \
  batch_size="${BATCH_SIZE}" \
  max_steps="${MAX_STEPS}" \
  save_every_steps="${SAVE_EVERY}" \
  checkpoint_overwrite=true \
  distributed=true \
  auto_resume=false \
  compile_action_expert=false \
  learning_rate_action="${LR_ACTION}" \
  lr_scale=none \
  learning_rate_warmup_steps="${WARMUP_STEPS}" \
  learning_rate_scheduler="${LR_SCHEDULER}" \
  weight_decay=1e-6 \
  adam_beta1=0.95 \
  adam_beta2=0.999 \
  "$@"

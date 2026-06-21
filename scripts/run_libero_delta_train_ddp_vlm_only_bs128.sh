#!/usr/bin/env bash
# Ablation 3 DDP: vlm_only effective bs=128 = 4 GPUs × bs32, LR=1e-4 (same as single-GPU bs128 run).
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
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
EXP="${EXP:-experiments/libero_spatial_vlm_only_15k_single_bs128}"
CONFIG="${CONFIG:-train_libero_spatial_act_delta_15k_single_vlm_only_bs128}"
CKPT_NAME="${CKPT_NAME:-libero_spatial_vlm_only_15k_single_bs128}"
# per-GPU batch; global effective = BATCH_SIZE * NGPUS
BATCH_SIZE="${BATCH_SIZE:-32}"
LR_ACTION="${LR_ACTION:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="${ROOT}/src:${ROOT}/../FastWAM/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

mkdir -p "${EXP}"

echo "==> VLM-only bs128 ablation (DDP): ${NGPUS} GPUs (${CUDA_DEVICES})"
echo "    config=${CONFIG} output=${EXP}"
echo "    per_device_batch=${BATCH_SIZE} effective_batch=$((BATCH_SIZE * NGPUS))"
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

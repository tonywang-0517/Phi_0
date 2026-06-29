#!/usr/bin/env bash
# 4-GPU DDP: pick-tissue SONIC unified (43s/100a), 8k steps, save every 4k.
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
CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}"
EXP="${EXP:-experiments/pick_tissue_sonic_unified_8k_ddp4}"
CONFIG="${CONFIG:-train_sonic_unified_ddp4_8k}"
CKPT_NAME="${CKPT_NAME:-pick_tissue_sonic_unified_act}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-8000}"
SAVE_EVERY="${SAVE_EVERY:-4000}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="${ROOT}/src:${ROOT}/../FastWAM/src:${ROOT}/../vggt-omega:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${EXP}"
LOG_FILE="${LOG_FILE:-${EXP}/train.log}"

echo "==> SONIC unified DDP: ${NGPUS} GPUs (${CUDA_DEVICES})"
echo "    config=${CONFIG} output=${EXP}"
echo "    per_device_batch=${BATCH_SIZE} effective_batch=$((BATCH_SIZE * NGPUS))"
echo "    max_steps=${MAX_STEPS} save_every=${SAVE_EVERY}"
echo "    log=${LOG_FILE}"

exec conda run --no-capture-output -n Phi-0-wpy torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NGPUS}" \
  scripts/train.py \
  --config-name "${CONFIG}" \
  output_dir="${EXP}" \
  checkpoint_name="${CKPT_NAME}" \
  batch_size="${BATCH_SIZE}" \
  max_steps="${MAX_STEPS}" \
  save_every_steps="${SAVE_EVERY}" \
  distributed=true \
  auto_resume=false \
  save_action_expert_only=true \
  "$@"

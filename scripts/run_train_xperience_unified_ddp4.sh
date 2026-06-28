#!/usr/bin/env bash
# 4-GPU DDP: Xperience unified 512-d action, freeze VLM, save action expert only.
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
EXP="${EXP:-experiments/xperience_unified_act_1k_ddp4}"
CONFIG="${CONFIG:-train_xperience_unified}"
CKPT_NAME="${CKPT_NAME:-xperience_unified_act}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_STEPS="${MAX_STEPS:-1000}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="${ROOT}/src:${ROOT}/../FastWAM/src:${ROOT}/../vggt-omega:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${EXP}"

echo "==> Xperience unified DDP: ${NGPUS} GPUs (${CUDA_DEVICES})"
echo "    config=${CONFIG} output=${EXP}"
echo "    per_device_batch=${BATCH_SIZE} effective_batch=$((BATCH_SIZE * NGPUS)) max_steps=${MAX_STEPS}"

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
  distributed=true \
  save_action_expert_only=true \
  "$@"

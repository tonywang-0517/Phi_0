#!/usr/bin/env bash
# 8-GPU DDP: pick_tissue_valid 512-d unified finetune with RTC enabled.
# Default per-device batch=4 → effective batch=32 (same as 4×8).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

NGPUS="${NGPUS:-8}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
EXP="${EXP:-experiments/pick_tissue_valid_wholebody_rtc_8k_ddp4}"
CONFIG="${CONFIG:-train_pick_tissue_finetune_rtc_ddp4}"
CKPT_NAME="${CKPT_NAME:-pick_tissue_valid_wholebody_rtc_act}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_STEPS="${MAX_STEPS:-8000}"
SAVE_EVERY="${SAVE_EVERY:-4000}"
AUTO_RESUME="${AUTO_RESUME:-false}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="${ROOT}/src:${ROOT}/../FastWAM/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${EXP}"
LOG_FILE="${LOG_FILE:-${EXP}/train.log}"

PHI0_PY="${PHI0_PY:-/mnt/data/miniconda3/envs/Phi-0-wpy/bin/python}"

echo "==> Pick-tissue valid RTC finetune DDP: ${NGPUS} GPUs (${CUDA_DEVICES})"
echo "    config=${CONFIG} output=${EXP}"
echo "    per_device_batch=${BATCH_SIZE} effective_batch=$((BATCH_SIZE * NGPUS))"
echo "    max_steps=${MAX_STEPS} save_every=${SAVE_EVERY}"
echo "    rtc=enabled (model cfg + deploy --rtc)"
echo "    log=${LOG_FILE}"

exec "${PHI0_PY}" -m torch.distributed.run \
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
  auto_resume="${AUTO_RESUME}" \
  save_action_expert_only=true \
  2>&1 | tee -a "${LOG_FILE}"

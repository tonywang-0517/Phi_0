#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

export PSI_HOME="${PSI_HOME:-${ROOT}}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
export PYTHONPATH="${ROOT}/src:${ROOT}/src/openpi/openpi-client/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# shellcheck disable=SC1091
source "${ROOT}/.venv-openpi/bin/activate"

NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
ulimit -n 65535
echo "Training π0.5 with $NPROC_PER_NODE GPUs (${CUDA_VISIBLE_DEVICES})"

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <config_name>"
  echo "Example: $0 G1WholebodyBendPickTeleop-v0"
  exit 1
fi

TASK="$1"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
MAX_STEPS="${MAX_STEPS:-}"

EXTRA=()
if [[ -n "${MAX_STEPS}" ]]; then
  EXTRA+=(--num_train_steps="${MAX_STEPS}")
fi

exec torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  src/openpi/train_pytorch.py \
  "${TASK}" \
  --exp_name="${TASK}" \
  --save_interval="${SAVE_INTERVAL}" \
  --checkpoint_base_dir=.runs/openpi-05 \
  "${EXTRA[@]}" \
  "${@:2}"

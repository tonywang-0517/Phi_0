#!/usr/bin/env bash
# Launch full Phi_0 training (Cosmos video tower + ActionDiT). Requires complete Cosmos weights.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

export PYTHONPATH="${ROOT}/src:/mnt/data2/wpy/workspace/FastWAM/src:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "==> Verifying Cosmos weights..."
if ! python scripts/verify_weights.py; then
  echo "Cosmos weights incomplete. Run: bash scripts/download_cosmos_weights.sh"
  exit 1
fi

echo "==> Starting train_full on ${CUDA_VISIBLE_DEVICES:-all GPUs}..."
exec python scripts/train.py "$@"

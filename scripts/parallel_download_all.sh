#!/usr/bin/env bash
# Download minimal Phi_0 demo samples only (no full Cosmos / EgoDex zip).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${ROOT}/download_logs"
mkdir -p "${LOG_DIR}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export PHI0_WORKSPACE="${PHI0_WORKSPACE:-/mnt/data2/wpy/workspace}"

echo "[$(date '+%H:%M:%S')] START minimal_samples" | tee "${LOG_DIR}/minimal_samples.log"
python "${ROOT}/scripts/download_samples.py" >>"${LOG_DIR}/minimal_samples.log" 2>&1
echo "[$(date '+%H:%M:%S')] OK   minimal_samples" | tee -a "${LOG_DIR}/summary.log"

echo ""
echo "Minimal sample download finished."
echo "For Cosmos weights (full GPU training only): bash scripts/download_cosmos_weights.sh"
echo "Logs: ${LOG_DIR}/"

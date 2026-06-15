#!/usr/bin/env bash
# Download Cosmos-Predict2.5-2B (diffusers/base/post-trained) to checkpoints/nvidia/.
# Requires HF token + NVIDIA license acceptance on huggingface.co (main branch 404).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/checkpoints/nvidia/Cosmos-Predict2.5-2B"
REPO="nvidia/Cosmos-Predict2.5-2B"
REVISION="diffusers/base/post-trained"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p "${DEST}"

echo "==> Cosmos-Predict2.5-2B (${REVISION})"
echo "    Destination: ${DEST}"
if ! hf download "${REPO}" --revision "${REVISION}" --local-dir "${DEST}" 2>/dev/null; then
  echo "Mirror failed for ${REPO}; retrying huggingface.co ..."
  HF_ENDPOINT=https://huggingface.co hf download "${REPO}" \
    --revision "${REVISION}" --local-dir "${DEST}"
fi

echo "Done. Run: python scripts/verify_weights.py"

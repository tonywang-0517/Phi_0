#!/usr/bin/env bash
# Download Wan2.2 / T5 / VAE / tokenizer (~34 GB total).
# Uses Phi_0/.env; tries hf-mirror first, falls back to huggingface.co per file.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="${ROOT}/checkpoints"
WAN22="${CKPT}/Wan-AI/Wan2.2-TI2V-5B"
WAN21="${CKPT}/Wan-AI/Wan2.1-T2V-1.3B"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p "${WAN22}" "${WAN21}"

_hf_one() {
  local repo="$1" file="$2" dest="$3"
  if ! hf download "${repo}" "${file}" --local-dir "${dest}" 2>/dev/null; then
    echo "Mirror failed for ${repo}/${file}; retrying huggingface.co ..."
    HF_ENDPOINT=https://huggingface.co hf download "${repo}" "${file}" --local-dir "${dest}"
  fi
}

echo "==> Wan2.2 TI2V-5B shards + VAE + T5"
for f in \
  diffusion_pytorch_model-00001-of-00003.safetensors \
  diffusion_pytorch_model-00002-of-00003.safetensors \
  diffusion_pytorch_model-00003-of-00003.safetensors \
  Wan2.2_VAE.pth \
  models_t5_umt5-xxl-enc-bf16.pth; do
  echo "  ${f}"
  _hf_one "Wan-AI/Wan2.2-TI2V-5B" "${f}" "${WAN22}"
done

echo "==> Wan2.1 tokenizer"
if ! hf download Wan-AI/Wan2.1-T2V-1.3B --include "google/umt5-xxl/*" --local-dir "${WAN21}" 2>/dev/null; then
  HF_ENDPOINT=https://huggingface.co hf download Wan-AI/Wan2.1-T2V-1.3B \
    --include "google/umt5-xxl/*" --local-dir "${WAN21}"
fi

echo "Done. Run: python scripts/verify_weights.py"

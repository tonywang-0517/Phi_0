#!/usr/bin/env bash
# Resume downloads from hf-mirror.com using `hf download`.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/mnt/data3/hf_home}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-3600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-3600}"

HF="${ROOT}/.venv-openpi/bin/hf"
LOG_DIR="${ROOT}/experiments"
mkdir -p "${LOG_DIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG_DIR}/hf_mirror_download.log"; }

download_pi05() {
  local dest="${ROOT}/cache/checkpoints/openpi-05/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/40000"
  mkdir -p "${dest}"
  log "=== pi0.5 LocomotionPick ckpt (resume) ==="
  "${HF}" download USC-PSI-Lab/psi-model \
    "openpi-05/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/40000/model.safetensors" \
    --local-dir "${ROOT}/cache/checkpoints" \
    --token "${HF_TOKEN}" \
    2>&1 | tee -a "${LOG_DIR}/hf_pi05.log"
  log "pi0.5 done: $(ls -lh "${dest}/model.safetensors")"
}

download_qwen() {
  log "=== Qwen3-VL-2B-Instruct (missing files) ==="
  "${HF}" download Qwen/Qwen3-VL-2B-Instruct \
    video_preprocessor_config.json \
    tokenizer.json \
    vocab.json \
    merges.txt \
    preprocessor_config.json \
    tokenizer_config.json \
    chat_template.json \
    generation_config.json \
    config.json \
    --cache-dir "${HF_HOME}/hub" \
    --token "${HF_TOKEN}" \
    2>&1 | tee -a "${LOG_DIR}/hf_qwen.log"
  log "Qwen done."
}

case "${1:-all}" in
  pi05) download_pi05 ;;
  qwen) download_qwen ;;
  all)  download_pi05; download_qwen ;;
  *)    echo "Usage: $0 [all|pi05|qwen]" >&2; exit 1 ;;
esac

log "All requested downloads finished."

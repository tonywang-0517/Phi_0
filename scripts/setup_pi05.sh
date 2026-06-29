#!/usr/bin/env bash
# One-shot OpenPI π0.5 env setup for Phi_0 SIMPLE training.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
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

PY="${PY:-python3.10}"
VENV="${ROOT}/.venv-openpi"

echo "==> OpenPI π0.5 setup (PSI_HOME=${PSI_HOME})"

if [[ ! -d "${VENV}" ]]; then
  echo "==> Creating venv ${VENV}"
  "${PY}" -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

pip install -U pip wheel setuptools
pip install python-dotenv tyro==0.9.32 omegaconf pydantic==2.10.6 rich polars numpydantic einops
pip install -e "${ROOT}/src/openpi/openpi-client"
# Pin stack before openpi deps (avoid phi0 pulling torch 2.12 / transformers 5.x)
pip install "numpy>=1.22.4,<2.0.0" "torch==2.7.1" "torchvision==0.22.1" "transformers==4.53.2"
pip install -r "${ROOT}/baselines/pi05/requirements-openpi.txt"

# transformers patch (OpenPI Paligemma)
SITE="${VENV}/lib/python3.10/site-packages/transformers"
if [[ -d "${SITE}" && -d "${ROOT}/src/openpi/models_pytorch/transformers_replace" ]]; then
  cp -r "${ROOT}/src/openpi/models_pytorch/transformers_replace/"* "${SITE}/"
  echo "==> Applied transformers_replace patch"
fi

mkdir -p "${PSI_HOME}/cache/checkpoints"
if [[ ! -f "${PSI_HOME}/cache/checkpoints/openpi/pi05_droid/model.safetensors" ]]; then
  echo "==> Downloading pi05_droid weights (HF mirror)"
  hf download USC-PSI-Lab/psi-model \
    --local-dir "${PSI_HOME}/cache/checkpoints" \
    --include "openpi/pi05_droid/*" \
    --repo-type model
fi

TASK="${TASK:-G1WholebodyBendPickTeleop-v0}"
DATA_DIR="${PSI_HOME}/data/simple/${TASK}"
if [[ ! -d "${DATA_DIR}/meta" ]]; then
  echo "ERROR: dataset missing at ${DATA_DIR}; run scripts/setup_simple_data.sh first"
  exit 1
fi

echo "==> Computing norm stats for ${TASK}"
python "${ROOT}/src/openpi/compute_norm_stats.py" --config-name "${TASK}"

echo "==> Setup complete. Train with:"
echo "    bash baselines/pi05/train_pi05.sh ${TASK}"

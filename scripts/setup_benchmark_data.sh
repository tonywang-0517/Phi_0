#!/usr/bin/env bash
# Wire Phi_0/data layouts for LIBERO/CALVIN train+eval (VLA-Adapter style).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ROOT_DIR}/data"
CALVIN_HOME="${DATA}/calvin"
THIRD_PARTY_CALVIN="${ROOT_DIR}/third_party/calvin"
LIBERO_VENDOR="${ROOT_DIR}/third_party/LIBERO"

echo "== Phi_0 benchmark data setup =="

mkdir -p "${CALVIN_HOME}/dataset"

# calvin_models: symlink to third_party if missing
if [[ ! -e "${CALVIN_HOME}/calvin_models" && -d "${THIRD_PARTY_CALVIN}/calvin_models" ]]; then
  ln -sfn "${THIRD_PARTY_CALVIN}/calvin_models" "${CALVIN_HOME}/calvin_models"
  echo "Linked calvin_models -> third_party/calvin/calvin_models"
fi

export CALVIN_ROOT="${CALVIN_HOME}"

# CALVIN sim validation (eval only; RLDS under calvin_abc_rlds/ is for bridge train)
VAL_DIR="${CALVIN_HOME}/dataset/task_ABC_D/validation"
if [[ ! -d "${VAL_DIR}" ]]; then
  if [[ -d "${CALVIN_HOME}/calvin_abc_rlds" ]] && ls "${CALVIN_HOME}/calvin_abc_rlds"/calvin_abc-train.tfrecord-* &>/dev/null; then
    echo "CALVIN RLDS present at ${CALVIN_HOME}/calvin_abc_rlds (bridge train OK)."
    echo "Skip sim-validation download — full ABC eval needs dataset/task_ABC_D/validation separately."
  else
    echo "WARN: no RLDS shards under ${CALVIN_HOME}/calvin_abc_rlds"
  fi
fi

# LIBERO package
if ! python -c "import libero" 2>/dev/null; then
  if [[ -d "${LIBERO_VENDOR}" ]]; then
    echo "Installing LIBERO from ${LIBERO_VENDOR} ..."
    pip install -e "${LIBERO_VENDOR}" -q || echo "WARN: LIBERO pip install failed"
  fi
fi

# tensorflow-cpu for RLDS bridge training (optional)
if ! python -c "import tensorflow" 2>/dev/null; then
  echo "Installing tensorflow-cpu for RLDS bridge training ..."
  pip install 'tensorflow-cpu==2.15.0' -q || echo "WARN: tensorflow install failed"
fi

echo ""
python "${ROOT_DIR}/scripts/check_benchmark_data.py"

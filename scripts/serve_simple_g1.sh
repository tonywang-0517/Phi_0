#!/usr/bin/env bash
# Serve Phi_0 checkpoint for SIMPLE closed-loop eval.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a Phi_0 .pt file}"
CONFIG="${CONFIG:-train_simple_g1_act}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-cuda:0}"
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-}"

export PYTHONPATH="${ROOT}/src:${ROOT}/third_party/SIMPLE/src:${PYTHONPATH:-}"

ARGS=(
  -m phi0.deploy.simple_serve
  --checkpoint "${CHECKPOINT}"
  --config-name "${CONFIG}"
  --host "${HOST}"
  --port "${PORT}"
  --device "${DEVICE}"
)

if [[ -n "${ACTION_EXEC_HORIZON}" ]]; then
  ARGS+=(--action-exec-horizon "${ACTION_EXEC_HORIZON}")
fi

exec python "${ARGS[@]}" "$@"

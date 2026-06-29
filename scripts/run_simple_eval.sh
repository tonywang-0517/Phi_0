#!/usr/bin/env bash
# Closed-loop SIMPLE eval: spawn Phi_0 HTTP server + SIMPLE EvalRunner.
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
DATA_DIR="${DATA_DIR:-./data/simple/G1WholebodyBendPick-v0-psi0}"
ENV_ID="${ENV_ID:-simple/G1WholebodyBendPick-v0}"
POLICY="${POLICY:-psi0}"
CONFIG="${CONFIG:-train_simple_g1_act}"

export PYTHONPATH="${ROOT}/src:${ROOT}/third_party/SIMPLE/src:${PYTHONPATH:-}"

exec python examples/simple/simple_eval.py \
  --checkpoint "${CHECKPOINT}" \
  --config-name "${CONFIG}" \
  --data-dir "${DATA_DIR}" \
  --env-id "${ENV_ID}" \
  --policy "${POLICY}" \
  "$@"

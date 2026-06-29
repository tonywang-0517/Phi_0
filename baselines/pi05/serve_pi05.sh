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
export PYTHONPATH="${ROOT}/src:${ROOT}/src/openpi/openpi-client/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <config_name> [ckpt_step] [port]"
  exit 1
fi

# shellcheck disable=SC1091
source "${ROOT}/.venv-openpi/bin/activate"

task="$1"
ckpt_step="${2:-40000}"
port="${3:-9000}"

echo "Serving π0.5 ${task} ckpt=${ckpt_step} on port ${port}"

exec python src/openpi/deploy/serve_policy.py \
  --port="${port}" \
  policy:checkpoint \
  --policy.config="${task}" \
  --policy.dir=".runs/openpi-05/${task}/${task}/${ckpt_step}"

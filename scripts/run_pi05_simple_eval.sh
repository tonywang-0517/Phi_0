#!/usr/bin/env bash
# π0.5 open-loop eval (train-set) + optional SIMPLE closed-loop.
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
export PYTHONPATH="${ROOT}/src:${ROOT}/src/openpi/openpi-client/src:${PYTHONPATH:-}"

TASK="${TASK:-G1WholebodyLocomotionPickBetweenTablesTeleop-v0}"
CKPT_STEP="${CKPT_STEP:-40000}"
PORT="${PORT:-9000}"
MODE="${MODE:-openloop}"  # openloop | simple
EVAL_DR="${EVAL_DR:-level-0}"
SIMPLE_ENV="${SIMPLE_ENV:-simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-${PSI_HOME}/data/evals/simple-eval/${TASK}/${EVAL_DR}}"
SIMPLE_PYTHON="${SIMPLE_PYTHON:-/mnt/data/miniconda3/envs/Phi-0-wpy/bin/python}"

setup_pi05_ckpt() {
  local link="${PSI_HOME}/.runs/openpi-05/${TASK}/${TASK}/${CKPT_STEP}"
  local src="${PSI_HOME}/cache/checkpoints/openpi-05/${TASK}/${CKPT_STEP}"
  local train_data="${PSI_HOME}/data/simple/${TASK}"
  [[ -d "${src}" ]] || { echo "Missing pi0.5 checkpoint: ${src}" >&2; exit 1; }
  mkdir -p "$(dirname "${link}")"
  rm -rf "${link}"
  ln -sfn "$(realpath "${src}")" "${link}"
  if [[ -d "${train_data}/meta" ]]; then
    local assets_dir asset_id="${TASK}"
    assets_dir="$(realpath "${link}")/assets"
    mkdir -p "${assets_dir}"
    ln -sfn "$(realpath "${train_data}")" "${assets_dir}/${asset_id}"
  fi
}

# shellcheck disable=SC1091
source "${ROOT}/.venv-openpi/bin/activate"

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_for_server() {
  local port="$1"
  local max_wait="${2:-600}"
  local waited=0
  echo "Waiting for policy server on port ${port} (up to ${max_wait}s)..."
  while (( waited < max_wait )); do
    if python - <<PY | grep -q ready
from openpi_client import websocket_client_policy as w
try:
    p = w.WebsocketClientPolicy(host="127.0.0.1", port=${port})
    p.get_server_metadata()
    print("ready")
except Exception:
    pass
PY
    then
      echo "Policy server is ready."
      return 0
    fi
    sleep 10
    waited=$((waited + 10))
    echo "  ... still loading (${waited}s)"
  done
  echo "Timed out waiting for policy server on port ${port}" >&2
  return 1
}

# GPU isolation: π0.5 serve and Isaac Sim eval use different physical GPUs.
SERVE_CUDA_VISIBLE_DEVICES="${SERVE_CUDA_VISIBLE_DEVICES:-${EVAL_CUDA_VISIBLE_DEVICES:-7}}"
SIMPLE_CUDA_VISIBLE_DEVICES="${SIMPLE_CUDA_VISIBLE_DEVICES:-0}"

setup_pi05_ckpt

echo "GPU isolation: serve=CUDA_VISIBLE_DEVICES=${SERVE_CUDA_VISIBLE_DEVICES}, simple=CUDA_VISIBLE_DEVICES=${SIMPLE_CUDA_VISIBLE_DEVICES}"

CUDA_VISIBLE_DEVICES="${SERVE_CUDA_VISIBLE_DEVICES}" \
  bash baselines/pi05/serve_pi05.sh "${TASK}" "${CKPT_STEP}" "${PORT}" &
server_pid=$!
wait_for_server "${PORT}" "${SERVER_WAIT_SECS:-600}"

if [[ "${MODE}" == "openloop" ]]; then
  exec python baselines/pi05/eval_openloop.py --port="${PORT}" --task="${TASK}"
fi

# SIMPLE closed-loop (Teleop task → decoupled WBC agent)
export DATA_DIR="${PSI_HOME}/data/simple"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${ROOT}/third_party/SIMPLE/src:${PYTHONPATH}"
cd "${ROOT}/third_party/SIMPLE"
CUDA_VISIBLE_DEVICES="${SIMPLE_CUDA_VISIBLE_DEVICES}" \
  "${SIMPLE_PYTHON}" -m simple.cli.eval_decoupled_wbc \
  "${SIMPLE_ENV}" \
  pi05_decoupled_wbc \
  "${EVAL_DR}" \
  --host=127.0.0.1 \
  --port="${PORT}" \
  --sim-mode=mujoco_isaac \
  --headless \
  --data-format=lerobot \
  --data-dir="${EVAL_DATA_DIR}" \
  --num-episodes="${NUM_EPISODES:-1}" \
  --save-video

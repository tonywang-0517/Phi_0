#!/usr/bin/env bash
# Run SIMPLE closed-loop eval for both Psi0 (HTTP) and pi0.5 (websocket) on the same task.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SIMPLE_ROOT="${ROOT}/third_party/SIMPLE"
PSI0_ROOT="${PSI0_ROOT:-/mnt/data2/wpy/workspace/Psi0-main}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

export PSI_HOME="${PSI_HOME:-${ROOT}}"
export DATA_HOME="${DATA_HOME:-${PSI_HOME}/data}"

TASK="${TASK:-G1WholebodyLocomotionPickBetweenTablesTeleop-v0}"
SIMPLE_ENV="${SIMPLE_ENV:-simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0}"
EVAL_DR="${EVAL_DR:-level-0}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-${PSI_HOME}/data/evals/simple-eval/${TASK}/${EVAL_DR}}"
NUM_EPISODES="${NUM_EPISODES:-1}"
SIM_MODE="${SIM_MODE:-mujoco_isaac}"
POLICY="${POLICY:-all}"  # all | pi05 | psi0

PI05_CKPT_STEP="${PI05_CKPT_STEP:-40000}"
PI05_PORT="${PI05_PORT:-9000}"
PSI0_PORT="${PSI0_PORT:-22085}"
EVAL_CUDA="${EVAL_CUDA_VISIBLE_DEVICES:-7}"

PSI0_RUN_DIR="${PSI0_RUN_DIR:-${PSI_HOME}/cache/checkpoints/psi0/simple-checkpoints/g1wholebodylocomotionpickbetweentablesteleop-v0.simple.flow1000.cosine.lr1.0e-04.b64.gpus4.2604081126}"
PSI0_CKPT_STEP="${PSI0_CKPT_STEP:-40000}"

PI05_CKPT_SRC="${PI05_CKPT_SRC:-${PSI_HOME}/cache/checkpoints/openpi-05/${TASK}/${PI05_CKPT_STEP}}"
PI05_CKPT_LINK="${PSI_HOME}/.runs/openpi-05/${TASK}/${TASK}/${PI05_CKPT_STEP}"

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_for_pi05() {
  local port="$1"
  local waited=0
  while (( waited < 600 )); do
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
      return 0
    fi
    sleep 10
    waited=$((waited + 10))
    echo "  waiting for pi0.5 server (${waited}s)..."
  done
  return 1
}

wait_for_psi0() {
  local port="$1"
  local waited=0
  while (( waited < 600 )); do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
    echo "  waiting for Psi0 server (${waited}s)..."
  done
  return 1
}

setup_pi05_ckpt() {
  mkdir -p "$(dirname "${PI05_CKPT_LINK}")"
  rm -rf "${PI05_CKPT_LINK}"
  ln -sfn "$(realpath "${PI05_CKPT_SRC}")" "${PI05_CKPT_LINK}"
  # norm stats for serve (hub ckpt is weights-only)
  local train_data="${PSI_HOME}/data/simple/${TASK}"
  if [[ -d "${train_data}/meta" && ! -d "${PI05_CKPT_LINK}/assets/${train_data}" ]]; then
    mkdir -p "${PI05_CKPT_LINK}/assets"
    ln -sfn "$(realpath "${train_data}")" "${PI05_CKPT_LINK}/assets/${train_data}"
  fi
}

run_simple_eval() {
  local agent="$1"
  local port="$2"
  echo "=== SIMPLE eval: ${agent} on ${SIMPLE_ENV} (${EVAL_DR}) ==="
  cd "${SIMPLE_ROOT}"
  export PYTHONPATH="${SIMPLE_ROOT}/src:${PYTHONPATH:-}"
  export DATA_DIR="${PSI_HOME}/data/simple"
  local py="${SIMPLE_PYTHON:-/mnt/data/miniconda3/envs/Phi-0-wpy/bin/python}"
  "${py}" -m simple.cli.eval_decoupled_wbc \
    "${SIMPLE_ENV}" \
    "${agent}" \
    "${EVAL_DR}" \
    --host=127.0.0.1 \
    --port="${port}" \
    --sim-mode="${SIM_MODE}" \
    --headless \
    --data-format=lerobot \
    --data-dir="${EVAL_DATA_DIR}" \
    --num-episodes="${NUM_EPISODES}" \
    --save-video
  echo "Videos: ${SIMPLE_ROOT}/data/evals/${agent}/$(basename "${SIMPLE_ENV}")/${EVAL_DR}/"
}

[[ -d "${EVAL_DATA_DIR}" ]] || { echo "Missing eval data: ${EVAL_DATA_DIR}" >&2; exit 1; }

if [[ "${POLICY}" == "all" || "${POLICY}" == "pi05" ]]; then
  setup_pi05_ckpt
  export PYTHONPATH="${ROOT}/src:${ROOT}/src/openpi/openpi-client/src:${PYTHONPATH:-}"
  export CUDA_VISIBLE_DEVICES="${EVAL_CUDA}"
  # shellcheck disable=SC1091
  source "${ROOT}/.venv-openpi/bin/activate"
  bash "${ROOT}/baselines/pi05/serve_pi05.sh" "${TASK}" "${PI05_CKPT_STEP}" "${PI05_PORT}" &
  server_pid=$!
  wait_for_pi05 "${PI05_PORT}"
  run_simple_eval "pi05_decoupled_wbc" "${PI05_PORT}"
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
  server_pid=""
  deactivate 2>/dev/null || true
fi

if [[ "${POLICY}" == "all" || "${POLICY}" == "psi0" ]]; then
  [[ -d "${PSI0_ROOT}" ]] || { echo "Psi0 repo not found: ${PSI0_ROOT}" >&2; exit 1; }
  [[ -d "${PSI0_RUN_DIR}" ]] || { echo "Psi0 checkpoint not found: ${PSI0_RUN_DIR}" >&2; exit 1; }
  ln -sfn "${PSI_HOME}/.env" "${PSI0_ROOT}/.env"
  cd "${PSI0_ROOT}"
  export CUDA_VISIBLE_DEVICES="${EVAL_CUDA}"
  export TRANSFORMERS_ATTN_IMPLEMENTATION="${TRANSFORMERS_ATTN_IMPLEMENTATION:-sdpa}"
  export PSI0_ATTN_IMPLEMENTATION="${PSI0_ATTN_IMPLEMENTATION:-sdpa}"
  export HF_HOME="${HF_HOME:-/mnt/data3/hf_home}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  PSI0_PYTHON="${PSI0_PYTHON:-/mnt/data/miniconda3/envs/Phi-0-wpy/bin/python}"
  "${PSI0_PYTHON}" -m psi.deploy.psi0_serve_simple \
    --host=0.0.0.0 \
    --port="${PSI0_PORT}" \
    --policy=psi0 \
    --run-dir="${PSI0_RUN_DIR}" \
    --ckpt-step="${PSI0_CKPT_STEP}" \
    --action-exec-horizon=24 \
    --rtc &
  server_pid=$!
  wait_for_psi0 "${PSI0_PORT}"
  run_simple_eval "psi0_decoupled_wbc" "${PSI0_PORT}"
fi

echo "Done."

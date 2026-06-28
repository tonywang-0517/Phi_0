#!/usr/bin/env bash
# Sim re-play pick-tissue parquets -> g1_debug base_trans -> patch valid -> rebuild v2.7 -> eval ep2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHI0_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GR00T_ROOT="$(cd "${PHI0_ROOT}/../GR00T-WholeBodyControl" && pwd)"
DEPLOY="${GR00T_ROOT}/gear_sonic_deploy"
WORK_DIR="${WORK_DIR:-${PHI0_ROOT}/../logs/pick_tissue_finetune/recollect_base_trans_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${WORK_DIR}/logs"
CAPTURE_DIR="${WORK_DIR}/captures"
mkdir -p "${LOG_DIR}" "${CAPTURE_DIR}"

VALID_ROOT="${VALID_ROOT:-${PHI0_ROOT}/../Isaac-GR00T/data/pick_tissue_valid}"
UNIFIED_OUT="${UNIFIED_OUT:-${PHI0_ROOT}/../Isaac-GR00T/data/pick_tissue_xperience_unified}"
PHI0_PY="${PHI0_PY:-/mnt/data/miniconda3/envs/Phi-0-wpy/bin/python}"
VENV_SIM="${VENV_SIM:-${GR00T_ROOT}/.venv_sim}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
MAX_EPISODES="${MAX_EPISODES:-0}"  # 0 = all parquets under valid/data
ONLY_EPISODE="${ONLY_EPISODE:-}"  # e.g. episode_000524 for single-ep test

export TensorRT_ROOT="${TensorRT_ROOT:-/mnt/data2/TensorRT-10.13.3.9}"
export onnxruntime_ROOT="${onnxruntime_ROOT:-/mnt/data2/wpy/deps/onnxruntime-linux-x64-1.16.3}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export LD_LIBRARY_PATH="${TensorRT_ROOT}/lib:${onnxruntime_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export PYTHONPATH="${GR00T_ROOT}:${PHI0_ROOT}/src:${PYTHONPATH:-}"

DEPLOY_FIFO="/tmp/pick_tissue_deploy_stdin_$$.fifo"
READY_FLAG="${WORK_DIR}/.replay_go"
SIM_PID=""
DEPLOY_PID=""

cleanup() {
  for pid in "${DEPLOY_PID}" "${SIM_PID}"; do
    [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
  done
  rm -f "${DEPLOY_FIFO}" "${READY_FLAG}"
  for port in 5555 5556 5557; do
    fuser -k "${port}/tcp" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

wait_log() {
  local file="$1" pattern="$2" timeout="$3"
  for _ in $(seq 1 "${timeout}"); do
    grep -qE "${pattern}" "${file}" 2>/dev/null && return 0
    sleep 1
  done
  return 1
}

send_deploy_key() { printf '%s' "$1" >&3; }

echo "[recollect] work_dir=${WORK_DIR}"

pkill -f "run_sim_loop_vla_record" 2>/dev/null || true
pkill -f "run_sim_loop_headless" 2>/dev/null || true
pkill -f "g1_deploy_onnx_ref.*zmq_manager" 2>/dev/null || true
pkill -f "replay_pick_tissue_parquet_zmq" 2>/dev/null || true
for port in 5555 5556 5557; do
  fuser -k "${port}/tcp" >/dev/null 2>&1 || true
done
sleep 2

# 1) MuJoCo sim (same path as HE gt replay — LowState bridge on lo)
(
  cd "${GR00T_ROOT}"
  source "${VENV_SIM}/bin/activate"
  python experiments/sonic_vla_overfit/scripts/run_sim_loop_vla_record.py \
    --no-enable-onscreen \
    --enable-image-publish --enable-offscreen \
    --keep-elastic-band \
    --camera-port 5555
) > "${LOG_DIR}/sim.log" 2>&1 &
SIM_PID=$!
wait_log "${LOG_DIR}/sim.log" "Sensor server running" 120 || {
  echo "[recollect] sim failed; tail sim.log"; tail -30 "${LOG_DIR}/sim.log"; exit 1
}
if grep -qE "Traceback|ValueError|error in simulator" "${LOG_DIR}/sim.log" 2>/dev/null; then
  echo "[recollect] sim crashed"; tail -30 "${LOG_DIR}/sim.log"; exit 1
fi
kill -0 "${SIM_PID}" 2>/dev/null || { echo "[recollect] sim died"; exit 1; }
echo "[recollect] sim up; waiting 15s for LowState bridge..."
sleep 15

# 2) C++ deploy zmq_manager
BIN="${DEPLOY}/target/release/g1_deploy_onnx_ref"
rm -f "${DEPLOY_FIFO}"; mkfifo "${DEPLOY_FIFO}"
(
  cd "${DEPLOY}"
  "${BIN}" lo policy/release/model_decoder.onnx reference/example/ \
    --obs-config policy/release/observation_config.yaml \
    --encoder-file policy/release/model_encoder.onnx \
    --planner-file planner/target_vel/V2/planner_sonic.onnx \
    --input-type zmq_manager --output-type all \
    --zmq-host 127.0.0.1 --zmq-port 5556 --disable-crc-check < "${DEPLOY_FIFO}"
) > "${LOG_DIR}/deploy.log" 2>&1 &
DEPLOY_PID=$!
exec 3>"${DEPLOY_FIFO}"
wait_log "${LOG_DIR}/deploy.log" "Init Done" 180 || true
sleep 5
send_deploy_key ']'
wait_log "${LOG_DIR}/deploy.log" "transitioning to CONTROL state|Safety check failed" 60 || true
if grep -q "Safety check failed" "${LOG_DIR}/deploy.log" 2>/dev/null; then
  echo "[recollect] deploy safety failed (no LowState). tail deploy.log"; tail -30 "${LOG_DIR}/deploy.log"; exit 1
fi
printf '\n' >&3
wait_log "${LOG_DIR}/deploy.log" "ZMQ STREAMING MODE: ENABLED" 90 || true
send_deploy_key 'I'
sleep 1
echo "[recollect] deploy streaming enabled"

mapfile -t PARQUETS < <(find "${VALID_ROOT}/data" -name 'episode_*.parquet' | sort)
if [[ -n "${ONLY_EPISODE}" ]]; then
  mapfile -t PARQUETS < <(find "${VALID_ROOT}/data" -name "${ONLY_EPISODE}.parquet")
fi
TOTAL=${#PARQUETS[@]}
if [[ "${MAX_EPISODES}" -gt 0 ]]; then
  PARQUETS=("${PARQUETS[@]:0:${MAX_EPISODES}}")
fi
echo "[recollect] patching ${#PARQUETS[@]}/${TOTAL} episodes in ${VALID_ROOT}"

EP=0
for PQ in "${PARQUETS[@]}"; do
  EP=$((EP + 1))
  BASE="$(basename "${PQ}" .parquet)"
  NFRAMES="$("${PHI0_PY}" - <<PY
import pandas as pd
print(len(pd.read_parquet("${PQ}")))
PY
)"
  CAP="${CAPTURE_DIR}/${BASE}.npy"
  rm -f "${READY_FLAG}"
  echo "[recollect] (${EP}/${#PARQUETS[@]}) ${BASE} frames=${NFRAMES}"

  "${PHI0_PY}" "${PHI0_ROOT}/scripts/data/capture_g1_debug_base_trans.py" \
    --num-frames "${NFRAMES}" \
    --out-npy "${CAP}" \
    --warmup-s 0.5 \
    --timeout-s "$((NFRAMES / 10 + 120))" \
    > "${LOG_DIR}/cap_${BASE}.log" 2>&1 &
  CAP_PID=$!
  sleep 0.3
  touch "${READY_FLAG}"
  "${PHI0_PY}" "${PHI0_ROOT}/scripts/data/replay_pick_tissue_parquet_zmq.py" \
    --parquet "${PQ}" \
    --ready-flag "${READY_FLAG}" \
    --fps 50 \
    > "${LOG_DIR}/replay_${BASE}.log" 2>&1 || true
  wait "${CAP_PID}" || {
    echo "[recollect] capture failed ${BASE}"; tail -15 "${LOG_DIR}/cap_${BASE}.log"; exit 1
  }
  "${PHI0_PY}" "${PHI0_ROOT}/scripts/data/backfill_parquet_base_trans.py" \
    --parquet "${PQ}" \
    --base-trans-npy "${CAP}" \
    --info-json "${VALID_ROOT}/meta/info.json"
done

echo "[recollect] sim re-play patch done; rebuild unified v2.7..."
"${PHI0_PY}" "${PHI0_ROOT}/scripts/data/isaac_groot_to_xperience_unified_lerobot.py" \
  --data-root "${VALID_ROOT}" \
  --out-dir "${UNIFIED_OUT}" \
  --num-workers 8 \
  2>&1 | tee "${LOG_DIR}/rebuild_v27.log"

echo "[recollect] verify ep524..."
"${PHI0_PY}" "${PHI0_ROOT}/scripts/data/verify_pick_tissue_qpos_labels.py" --dst-ep 524

EVAL_DIR="${WORK_DIR}/eval"
mkdir -p "${EVAL_DIR}"
WORK_DIR="${EVAL_DIR}" \
MANIFEST_SESSION=2026-06-25-16-09-43 MANIFEST_EP=2 \
DEPLOY_MODE=qpos USE_GT=1 SHOW_GT_VIEWS=0 MOTION_SECONDS=8 \
bash "${PHI0_ROOT}/scripts/run_pick_tissue_hgpt_zmq_eval.sh" \
  2>&1 | tee "${LOG_DIR}/eval.log"

echo "[recollect] done. work_dir=${WORK_DIR}"

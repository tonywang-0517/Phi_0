#!/usr/bin/env bash
# Pick-tissue GT SONIC motion_token (v4 latent) -> deploy zmq_manager -> MuJoCo mp4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/setup_env.sh"
GR00T_ROOT="$(cd "${GR00T_ROOT}" && pwd)"
VENV_SIM="${GR00T_ROOT}/.venv_sim"
DEPLOY="${GR00T_ROOT}/gear_sonic_deploy"
ROBOT_MOTION="${ROBOT_MOTION:-${GR00T_ROOT}/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl}"
WORK_DIR="${WORK_DIR:-${PHI0_ROOT}/../logs/pick_tissue_finetune/sonic_latent_$([ -n "${CHECKPOINT:-}" ] && echo model || echo gt)_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${WORK_DIR}"
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
LOG_DIR="${WORK_DIR}/logs"
mkdir -p "${LOG_DIR}"

PHI0_PY="${PHI0_PY:-/home/user/anaconda3/envs/Phi-0-wpy/bin/python}"
VALID_ROOT="${VALID_ROOT:-${PHI0_ROOT}/../Isaac-GR00T/data/pick_tissue_valid}"
UNIFIED_ROOT="${UNIFIED_ROOT:-${PHI0_ROOT}/../Isaac-GR00T/data/pick_tissue_xperience_unified}"
MANIFEST_PATH="${MANIFEST_PATH:-${PHI0_ROOT}/../Isaac-GR00T/data/pick_tissues.json}"
TOKEN_SOURCE="${TOKEN_SOURCE:-unified_slice}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RECORD_FPS="${RECORD_FPS:-30}"
CONTROL_FPS="${CONTROL_FPS:-50}"
MOTION_SECONDS="${MOTION_SECONDS:-8}"
MAX_FRAMES="${MAX_FRAMES:-0}"
RECORD_SETTLE_S="${RECORD_SETTLE_S:-8}"
RECORD_STABLE_S="${RECORD_STABLE_S:-5}"
GT_PANEL_LAYOUT="${GT_PANEL_LAYOUT:-inset}"
ENABLE_G1_DEBUG_OVERLAY="${ENABLE_G1_DEBUG_OVERLAY:-1}"
CHECKPOINT="${CHECKPOINT:-}"
CONFIG_NAME="${CONFIG_NAME:-train_pick_tissue_xperience_unified_ddp4_3k}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export PYTHONPATH="${GR00T_ROOT}:${PHI0_ROOT}/src:${PYTHONPATH:-}"
# TensorRT / onnxruntime / unitree DDS: setup_env.sh
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
SIM_WARMUP_S="${SIM_WARMUP_S:-10}"
DEPLOY_INIT_TIMEOUT_S="${DEPLOY_INIT_TIMEOUT_S:-300}"
REPLAY_READY_TIMEOUT_S="${REPLAY_READY_TIMEOUT_S:-900}"

# Resolve episode parquets (manifest ep2 -> unified idx 447, valid src ep524)
UNIFIED_EP="${UNIFIED_EP:-}"
VALID_EP="${VALID_EP:-}"
if [[ -n "${MANIFEST_SESSION:-}" && -n "${MANIFEST_EP:-}" ]]; then
  read -r UNIFIED_EP VALID_EP <<<"$(
    PHI0_ROOT="${PHI0_ROOT}" MANIFEST_PATH="${MANIFEST_PATH}" VALID_ROOT="${VALID_ROOT}" \
    MANIFEST_SESSION="${MANIFEST_SESSION}" MANIFEST_EP="${MANIFEST_EP}" "${PHI0_PY}" - <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.environ["PHI0_ROOT"], "src"))
from phi0.data.pick_tissue_episode_map import (
    manifest_ep_to_dst_ep,
    manifest_ep_to_unified_episode_index,
)
m, v, s, e = (
    os.environ["MANIFEST_PATH"],
    os.environ["VALID_ROOT"],
    os.environ["MANIFEST_SESSION"],
    int(os.environ["MANIFEST_EP"]),
)
ui = manifest_ep_to_unified_episode_index(m, v, s, e)
di = manifest_ep_to_dst_ep(m, s, e)
print(ui, di)
PY
  )"
  echo "[sonic_latent] manifest ${MANIFEST_SESSION} ep${MANIFEST_EP} -> unified=${UNIFIED_EP} valid=${VALID_EP}"
fi
UNIFIED_EP="${UNIFIED_EP:-447}"
if [[ -z "${VALID_EP:-}" ]]; then
  VALID_EP="$("${PHI0_PY}" - <<PY
import sys
sys.path.insert(0, "${PHI0_ROOT}/src")
from phi0.data.pick_tissue_episode_map import _sorted_valid_parquets
files = _sorted_valid_parquets("${VALID_ROOT}")
idx = int("${UNIFIED_EP}")
name = files[idx].stem  # episode_000544
print(int(name.split("_")[-1]))
PY
)"
fi

UNIFIED_PARQUET="${UNIFIED_PARQUET:-${UNIFIED_ROOT}/data/chunk-000/episode_$(printf '%06d' "${UNIFIED_EP}").parquet}"
VALID_PARQUET="${VALID_PARQUET:-${VALID_ROOT}/data/chunk-000/episode_$(printf '%06d' "${VALID_EP}").parquet}"
EGO_MP4="${EGO_MP4:-${UNIFIED_ROOT}/videos/chunk-000/observation.images.ego_view/episode_$(printf '%06d' "${UNIFIED_EP}").mp4}"
WRIST_MP4="${WRIST_MP4:-${UNIFIED_ROOT}/videos/chunk-000/observation.images.left_wrist/episode_$(printf '%06d' "${UNIFIED_EP}").mp4}"
OUT_MP4="${OUT_MP4:-${WORK_DIR}/pick_tissue_ep${UNIFIED_EP}_sonic_latent_$([ -n "${CHECKPOINT}" ] && echo model || echo gt).mp4}"

if [[ -n "${MOTION_NPZ:-}" && -f "${MOTION_NPZ}" ]]; then
  _NPZ_FRAMES="$("${PHI0_PY}" - <<PY
import numpy as np
print(int(np.load("${MOTION_NPZ}")["tokens"].shape[0]))
PY
  )"
  if [[ "${MAX_FRAMES}" -le 0 ]]; then
    MAX_FRAMES="${_NPZ_FRAMES}"
  else
    MAX_FRAMES="$("${PHI0_PY}" - <<PY
print(min(int("${MAX_FRAMES}"), int("${_NPZ_FRAMES}")))
PY
)"
  fi
  MOTION_SECONDS="$("${PHI0_PY}" - <<PY
print(round(int("${MAX_FRAMES}") / float("${CONTROL_FPS}"), 2))
PY
  )"
  echo "[sonic_latent] MOTION_NPZ=${MOTION_NPZ} frames=${MAX_FRAMES} (~${MOTION_SECONDS}s)"
elif [[ "${MAX_FRAMES}" -le 0 ]]; then
  MAX_FRAMES="$("${PHI0_PY}" - <<PY
import math
print(int(math.ceil(float("${MOTION_SECONDS}") * float("${CONTROL_FPS}"))))
PY
)"
fi
PARQUET_ROWS="$("${PHI0_PY}" - <<PY
import pyarrow.parquet as pq
print(pq.read_metadata("${UNIFIED_PARQUET}").num_rows)
PY
)"
REQUESTED="${MAX_FRAMES}"
if [[ -z "${MOTION_NPZ:-}" && "${PARQUET_ROWS}" -lt "${MAX_FRAMES}" ]]; then
  MAX_FRAMES="${PARQUET_ROWS}"
  EP_DUR="$("${PHI0_PY}" - <<PY
print(round(${PARQUET_ROWS}/float("${CONTROL_FPS}"), 2))
PY
)"
  echo "[sonic_latent] WARN: episode has ${PARQUET_ROWS} frames (~${EP_DUR}s @ ${CONTROL_FPS}Hz), less than MOTION_SECONDS=${MOTION_SECONDS} (${REQUESTED} frames)"
fi

RECORD_START="${WORK_DIR}/.record_start"
RECORD_STOP="${WORK_DIR}/.record_stop"
ARM_FLAG="${WORK_DIR}/.arm_deploy"
REPLAY_READY="${WORK_DIR}/.replay_go"
DEPLOY_FIFO="${DEPLOY_FIFO:-/tmp/pick_tissue_sonic_deploy_$$.fifo}"

SIM_PID=""
DEPLOY_PID=""
REPLAY_PID=""

log_step() {
  echo "[sonic_latent][$(date '+%H:%M:%S')] $*"
}

pid_alive() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

require_pid() {
  local pid="$1" name="$2" logfile="${3:-}"
  if pid_alive "${pid}"; then
    return 0
  fi
  log_step "ERROR: ${name} (pid=${pid}) exited unexpectedly"
  if [[ -n "${logfile}" && -f "${logfile}" ]]; then
    log_step "--- tail ${logfile} ---"
    tail -25 "${logfile}" || true
  fi
  exit 1
}

require_replay() {
  local label="replay"
  if [[ -n "${MOTION_NPZ:-}" ]]; then
    label="npz replay"
  elif [[ -n "${CHECKPOINT:-}" ]]; then
    label="model publisher"
  fi
  require_pid "${REPLAY_PID}" "${label}" "${LOG_DIR}/replay.log"
  if grep -qE "Traceback \(most recent call last\)" "${LOG_DIR}/replay.log" 2>/dev/null; then
    log_step "model publisher crashed"
    tail -30 "${LOG_DIR}/replay.log"
    exit 1
  fi
}

_fall_count() {
  local n=0
  if [[ -f "$1" ]]; then
    n=$(grep -cF '[sim_health] FALL' "$1" 2>/dev/null) || n=0
  fi
  echo "${n}"
}

cleanup() {
  for pid in "${REPLAY_PID}" "${DEPLOY_PID}" "${SIM_PID}"; do
    [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null && kill "${pid}" 2>/dev/null || true
  done
  rm -f "${DEPLOY_FIFO}" "${RECORD_START}" "${RECORD_STOP}" "${ARM_FLAG}" "${REPLAY_READY}"
}
trap cleanup EXIT

wait_log() {
  local file="$1" pattern="$2" timeout="$3"
  local label="${4:-pattern}"
  log_step "wait_log: ${label} (timeout=${timeout}s) -> ${file}"
  for i in $(seq 1 "${timeout}"); do
    if grep -qE "${pattern}" "${file}" 2>/dev/null; then
      log_step "wait_log: OK ${label} (${i}s)"
      return 0
    fi
    if (( i % 10 == 0 )); then
      log_step "wait_log: still waiting ${label} (${i}/${timeout}s)"
      tail -2 "${file}" 2>/dev/null | sed 's/^/[sonic_latent]   /' || true
    fi
    sleep 1
  done
  log_step "wait_log: TIMEOUT ${label} after ${timeout}s"
  tail -20 "${file}" 2>/dev/null | sed 's/^/[sonic_latent]   /' || true
  return 1
}

send_deploy_key() { printf '%s' "$1" >&3; }

echo "[sonic_latent] unified=${UNIFIED_PARQUET} valid_hands=${VALID_PARQUET}"
echo "[sonic_latent] token_source=${TOKEN_SOURCE} frames=${MAX_FRAMES} out=${OUT_MP4}"
if [[ -n "${CHECKPOINT}" ]]; then
  echo "[sonic_latent] MODEL checkpoint=${CHECKPOINT} config=${CONFIG_NAME}"
elif [[ -n "${MOTION_NPZ:-}" ]]; then
  echo "[sonic_latent] NPZ replay motion=${MOTION_NPZ} hand_ramp=${HAND_RAMP_FRAMES:-0}"
fi
log_step "work_dir=${WORK_DIR} gpu=${CUDA_VISIBLE_DEVICES} sim_warmup=${SIM_WARMUP_S}s deploy_timeout=${DEPLOY_INIT_TIMEOUT_S}s"

pkill -f "run_sim_loop_vla_record" 2>/dev/null || true
pkill -f "g1_deploy_onnx_ref.*zmq_manager" 2>/dev/null || true
pkill -f "replay_pick_tissue_sonic_latent_zmq_v4" 2>/dev/null || true
pkill -f "replay_sonic_latent_npz_zmq_v4" 2>/dev/null || true
for port in 5555 5556 5557; do
  fuser -k "${port}/tcp" >/dev/null 2>&1 || true
done
sleep 2
rm -f "${RECORD_START}" "${RECORD_STOP}" "${ARM_FLAG}" "${REPLAY_READY}"

PRECOMPUTE_NPZ="${PRECOMPUTE_IN:-${WORK_DIR}/sonic_latent_precompute.npz}"
USE_PRECOMPUTE_NPZ=0
if [[ -n "${CHECKPOINT}" ]]; then
  if [[ "${FORCE_PRECOMPUTE:-}" == "1" ]]; then
    if [[ ! -f "${PRECOMPUTE_NPZ}" ]]; then
      log_step "optional offline precompute -> ${PRECOMPUTE_NPZ} (set FORCE_PRECOMPUTE=0 for inline infer)"
      (
        cd "${PHI0_ROOT}"
        "${PHI0_PY}" "${PHI0_ROOT}/scripts/phi0_sonic_latent_zmq_publisher.py" \
          --checkpoint "${CHECKPOINT}" \
          --config-name "${CONFIG_NAME}" \
          --episode-idx "${UNIFIED_EP}" \
          --control-fps "${CONTROL_FPS}" \
          --motion-seconds "${MOTION_SECONDS}" \
          --max-frames "${MAX_FRAMES}" \
          --precompute-out "${PRECOMPUTE_NPZ}"
      ) > "${LOG_DIR}/precompute.log" 2>&1 || {
        log_step "precompute failed"
        tail -40 "${LOG_DIR}/precompute.log"
        exit 1
      }
      log_step "precompute done ($(wc -l < "${LOG_DIR}/precompute.log" | tr -d ' ') log lines)"
    else
      log_step "reusing precompute ${PRECOMPUTE_NPZ}"
    fi
    USE_PRECOMPUTE_NPZ=1
  elif [[ -n "${PRECOMPUTE_IN:-}" ]]; then
    if [[ ! -f "${PRECOMPUTE_NPZ}" ]]; then
      log_step "ERROR: PRECOMPUTE_IN=${PRECOMPUTE_IN} not found"
      exit 1
    fi
    log_step "reuse precompute ${PRECOMPUTE_NPZ}"
    USE_PRECOMPUTE_NPZ=1
  else
    log_step "inline infer at publisher (dataset clip + VLM, no offline precompute)"
  fi
fi

# 1) MuJoCo sim + mp4 (defaults match sonic_latent_gt_20260628_030836)
SIM_EXTRA_ARGS=()
SIM_EGO_GT="${EGO_MP4}"
if [[ "${GT_PANEL_LAYOUT}" == "top" ]]; then
  SIM_EXTRA_ARGS+=(--gt-panel-layout top --wrist-gt-video "${WRIST_MP4}")
elif [[ "${GT_PANEL_LAYOUT}" == "sim" ]]; then
  SIM_EGO_GT=""
  SIM_EXTRA_ARGS+=(--camera-host 127.0.0.1)
fi
if [[ "${ENABLE_G1_DEBUG_OVERLAY}" == "1" ]]; then
  SIM_EXTRA_ARGS+=(--enable-g1-debug-overlay)
else
  SIM_EXTRA_ARGS+=(--g1-debug-snap --no-enable-g1-debug-overlay)
fi
# Cluster-rsync'd .venv_sim needs local python + VIRTUAL_ENV paths.
bash "${PHI0_ROOT}/scripts/fix_venv_sim.sh" >> "${LOG_DIR}/fix_venv_sim.log" 2>&1 || true
(
  cd "${GR00T_ROOT}"
  source "${VENV_SIM}/bin/activate"
  python -u experiments/sonic_vla_overfit/scripts/run_sim_loop_vla_record.py \
    --no-enable-onscreen \
    --enable-image-publish --enable-offscreen \
    --disable-elastic-band \
    --camera-port 5555 \
    --g1-debug-host 127.0.0.1 \
    --g1-debug-port 5557 \
    --no-snap-on-record-start \
    ${SIM_EGO_GT:+--ego-gt-video "${SIM_EGO_GT}"} \
    --record-mp4 "${OUT_MP4}" \
    --record-start-flag "${RECORD_START}" \
    --record-stop-flag "${RECORD_STOP}" \
    --record-fps "${RECORD_FPS}" \
    "${SIM_EXTRA_ARGS[@]}"
) > "${LOG_DIR}/sim.log" 2>&1 &
SIM_PID=$!
log_step "sim pid=${SIM_PID} log=${LOG_DIR}/sim.log"
wait_log "${LOG_DIR}/sim.log" "Sensor server running" 120 "sim Sensor server" || {
  if ss -tlnp 2>/dev/null | grep -qE ":${CAMERA_PORT:-5555}\\b"; then
    log_step "wait_log: OK sim Sensor server (port ${CAMERA_PORT:-5555} listening)"
  else
    require_pid "${SIM_PID}" "sim" "${LOG_DIR}/sim.log"
    tail -30 "${LOG_DIR}/sim.log"; exit 1;
  fi
}
log_step "sim up; LowState bridge warmup ${SIM_WARMUP_S}s..."
for ((w=SIM_WARMUP_S; w>0; w-=5)); do
  require_pid "${SIM_PID}" "sim" "${LOG_DIR}/sim.log"
  log_step "warmup ${w}s remaining..."
  sleep 5
done

# 2) ZMQ v4 publisher: arm_flag -> command start; replay_go -> pose stream
if [[ -n "${MOTION_NPZ:-}" ]]; then
  log_step "starting motion npz replay (${MOTION_NPZ})..."
  (
    "${PHI0_PY}" "${PHI0_ROOT}/scripts/replay_sonic_latent_npz_zmq_v4.py" \
      --npz "${MOTION_NPZ}" \
      --zmq-port 5556 \
      --fps "${CONTROL_FPS}" \
      --max-frames "${MAX_FRAMES}" \
      --start-delay-s 0.5 \
      --hand-ramp-frames "${HAND_RAMP_FRAMES:-0}" \
      --arm-flag "${ARM_FLAG}" \
      --ready-flag "${REPLAY_READY}"
  ) > "${LOG_DIR}/replay.log" 2>&1 &
elif [[ -n "${CHECKPOINT}" ]]; then
  log_step "starting Phi-0 model publisher..."
  PUBLISHER_ARGS=(
    --config-name "${CONFIG_NAME}"
    --episode-idx "${UNIFIED_EP}"
    --zmq-port 5556
    --control-fps "${CONTROL_FPS}"
    --motion-seconds "${MOTION_SECONDS}"
    --max-frames "${MAX_FRAMES}"
    --start-delay-s 0.5
    --arm-flag "${ARM_FLAG}"
    --ready-flag "${REPLAY_READY}"
    --ready-timeout-s "${REPLAY_READY_TIMEOUT_S}"
  )
  if [[ "${USE_PRECOMPUTE_NPZ}" == "1" ]]; then
    PUBLISHER_ARGS=(--precompute-in "${PRECOMPUTE_NPZ}" "${PUBLISHER_ARGS[@]}")
  else
    PUBLISHER_ARGS=(--checkpoint "${CHECKPOINT}" "${PUBLISHER_ARGS[@]}")
  fi
  (
    cd "${PHI0_ROOT}"
    "${PHI0_PY}" "${PHI0_ROOT}/scripts/phi0_sonic_latent_zmq_publisher.py" \
      "${PUBLISHER_ARGS[@]}"
  ) > "${LOG_DIR}/replay.log" 2>&1 &
else
  log_step "starting GT replay publisher..."
  (
    "${PHI0_PY}" "${PHI0_ROOT}/scripts/data/replay_pick_tissue_sonic_latent_zmq_v4.py" \
      --parquet "${UNIFIED_PARQUET}" \
      --token-source "${TOKEN_SOURCE}" \
      --valid-parquet-for-hands "${VALID_PARQUET}" \
      --zmq-port 5556 \
      --fps "${CONTROL_FPS}" \
      --max-frames "${MAX_FRAMES}" \
      --start-delay-s 0.5 \
      --arm-flag "${ARM_FLAG}" \
      --ready-flag "${REPLAY_READY}"
  ) > "${LOG_DIR}/replay.log" 2>&1 &
fi
REPLAY_PID=$!
log_step "replay pid=${REPLAY_PID} log=${LOG_DIR}/replay.log"
if [[ -n "${MOTION_NPZ:-}" ]]; then
  wait_log "${LOG_DIR}/replay.log" "bound tcp://" 30 "npz replay bound" || require_pid "${REPLAY_PID}" "npz replay" "${LOG_DIR}/replay.log"
elif [[ -n "${CHECKPOINT}" ]]; then
  # Precomputed path binds tcp in ~1s; inline VLM+inference may take minutes.
  wait_timeout=600
  if [[ "${USE_PRECOMPUTE_NPZ}" == "1" ]]; then
    wait_timeout=60
  fi
  for i in $(seq 1 "${wait_timeout}"); do
    require_pid "${REPLAY_PID}" "model publisher" "${LOG_DIR}/replay.log"
    if grep -qE "bound tcp://" "${LOG_DIR}/replay.log" 2>/dev/null; then
      log_step "model publisher ready (${i}s)"
      break
    fi
    if grep -qE "Traceback \(most recent call last\)" "${LOG_DIR}/replay.log" 2>/dev/null; then
      log_step "model publisher failed during load"
      tail -30 "${LOG_DIR}/replay.log"
      exit 1
    fi
    if (( i % 15 == 0 )); then
      log_step "model publisher loading... (${i}/${wait_timeout}s)"
      tail -3 "${LOG_DIR}/replay.log" 2>/dev/null | sed 's/^/[sonic_latent]   /' || true
    fi
    sleep 1
  done
else
  wait_log "${LOG_DIR}/replay.log" "bound tcp://" 30 "replay bound" || require_pid "${REPLAY_PID}" "replay" "${LOG_DIR}/replay.log"
fi

# 3) C++ deploy zmq_manager (TensorRT init ~1–3 min — looks idle in terminal)
log_step "starting deploy TensorRT (pid pending, log=${LOG_DIR}/deploy.log)..."
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
log_step "deploy pid=${DEPLOY_PID}"
if ! wait_log "${LOG_DIR}/deploy.log" "Init Done" "${DEPLOY_INIT_TIMEOUT_S}" "deploy Init Done"; then
  require_pid "${DEPLOY_PID}" "deploy" "${LOG_DIR}/deploy.log"
  exit 1
fi
log_step "deploy Init Done; arming via replay/publisher..."

touch "${ARM_FLAG}"
if ! wait_log "${LOG_DIR}/deploy.log" "transitioning to CONTROL state|ZMQManager.*Planner enabled" 90 "deploy arm/CONTROL"; then
  require_pid "${REPLAY_PID}" "replay/publisher" "${LOG_DIR}/replay.log"
  log_step "deploy did not arm; tails:"
  tail -30 "${LOG_DIR}/deploy.log"
  tail -15 "${LOG_DIR}/replay.log"
  exit 1
fi
if ! wait_log "${LOG_DIR}/deploy.log" "transitioning to CONTROL state" 60 "deploy CONTROL"; then
  log_step "deploy never entered CONTROL"
  tail -40 "${LOG_DIR}/deploy.log"
  exit 1
fi
log_step "deploy in CONTROL; enabling ZMQ streaming (ENTER)..."
for attempt in $(seq 1 12); do
  grep -q "ZMQ STREAMING MODE: ENABLED" "${LOG_DIR}/deploy.log" 2>/dev/null && break
  printf '\n' >&3
  sleep 2
done
if ! grep -q "ZMQ STREAMING MODE: ENABLED" "${LOG_DIR}/deploy.log" 2>/dev/null; then
  echo "[sonic_latent] ZMQ streaming not enabled"; tail -20 "${LOG_DIR}/deploy.log"; exit 1
fi
send_deploy_key 'I'
sleep 0.5

log_step "wait deploy lowcmd active (sim standing under deploy)..."
if ! wait_log "${LOG_DIR}/sim.log" "deploy lowcmd active" 120 "deploy lowcmd"; then
  tail -30 "${LOG_DIR}/sim.log"
  exit 1
fi
require_replay
log_step "wait sim stable (no new FALL for ${RECORD_STABLE_S}s)..."
stable_deadline=$(( $(date +%s) + 90 ))
while (( $(date +%s) < stable_deadline )); do
  require_replay
  fall_count="$(_fall_count "${LOG_DIR}/sim.log")"
  sleep "${RECORD_STABLE_S}"
  fall_after="$(_fall_count "${LOG_DIR}/sim.log")"
  if [[ "${fall_after}" -le "${fall_count}" ]]; then
    log_step "sim stable (${RECORD_STABLE_S}s without new FALL, total_fall=${fall_after})"
    break
  fi
  log_step "sim still settling (FALL ${fall_count} -> ${fall_after}), waiting..."
done
log_step "settle ${RECORD_SETTLE_S}s after deploy active before record..."
sleep "${RECORD_SETTLE_S}"

touch "${RECORD_START}"
if ! wait_log "${LOG_DIR}/sim.log" "sim_record] started" 60 "sim record start"; then
  log_step "record flag paths: start=${RECORD_START} stop=${RECORD_STOP}"
  ls -la "${RECORD_START}" "${REPLAY_READY}" 2>&1 || true
  tail -20 "${LOG_DIR}/sim.log"; exit 1
fi
touch "${REPLAY_READY}"
log_step "recording + streaming ${MAX_FRAMES} frames @ ${CONTROL_FPS}Hz..."
require_replay

while kill -0 "${REPLAY_PID}" 2>/dev/null; do
  require_pid "${SIM_PID}" "sim" "${LOG_DIR}/sim.log"
  require_pid "${DEPLOY_PID}" "deploy" "${LOG_DIR}/deploy.log"
  sleep 2
done
wait "${REPLAY_PID}" || true
sleep 1
touch "${RECORD_STOP}"
for _ in $(seq 1 60); do
  grep -q "sim_record] saved" "${LOG_DIR}/sim.log" 2>/dev/null && break
  if ! kill -0 "${SIM_PID}" 2>/dev/null; then
    log_step "WARN: sim exited before record save; see ${LOG_DIR}/sim.log"
    break
  fi
  sleep 0.5
done
if [[ -n "${MOTION_NPZ:-}" ]]; then
  token_rx="$(grep -c "Received 64D token" "${LOG_DIR}/deploy.log" 2>/dev/null || echo 0)"
  log_step "deploy received ${token_rx} x 64D token frames"
fi

# ponytail: OpenCV mp4v is unreadable on many players; remux to H.264
OUT_H264="${OUT_MP4%.mp4}_h264.mp4"
if command -v ffmpeg >/dev/null 2>&1 && [[ -f "${OUT_MP4}" ]] && [[ $(stat -c%s "${OUT_MP4}") -gt 1000 ]]; then
  if ffmpeg -y -loglevel error -i "${OUT_MP4}" \
    -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart \
    "${OUT_H264}"; then
    mv "${OUT_H264}" "${OUT_MP4}"
    echo "[sonic_latent] remuxed to H.264 (libx264)"
  fi
fi

echo "[sonic_latent] replay tail:"
tail -8 "${LOG_DIR}/replay.log" || true
echo "[sonic_latent] deploy token/hand:"
grep -E "64D token|hand joints set" "${LOG_DIR}/deploy.log" | tail -6 || true
echo "[sonic_latent] video: ${OUT_MP4}"
ls -lh "${OUT_MP4}" 2>/dev/null || true
echo "[sonic_latent] done work_dir=${WORK_DIR}"

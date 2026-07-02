#!/usr/bin/env bash
# Phi-0 closed-loop SONIC deploy: live camera + deploy g1_debug proprio -> ZMQ v4.
#
# Prerequisites (you start these):
#   1. SONIC composed_camera on CAMERA_PORT (default 5555)
#      e.g. on robot: python -m gear_sonic.camera.composed_camera --port 5555
#   2. g1_deploy_onnx_ref --input-type zmq_manager --zmq-port ZMQ_PORT (default 5556)
#      Press ] in the closed-loop terminal for CONTROL + streamed motion (or use STREAM_NOW=1)
#   3. deploy publishes g1_debug on STATE_ZMQ_PORT (default 5557) after control loop starts
#
# Default camera-source=live reads from tcp://CAMERA_HOST:5555 (ComposedCameraClientSensor).
# Set CAMERA_SOURCE=gt and GT_CAMERA_EP=447 to use dataset video instead.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/setup_env.sh"
cd "${PHI0_ROOT}"

CHECKPOINT="${CHECKPOINT:-${PHI0_ROOT}/experiments/pick_tissue_xperience_unified_3k_ddp4_fast/pick_tissue_xperience_unified_act_latest.pt}"
CONFIG_NAME="${CONFIG_NAME:-train_pick_tissue_xperience_unified_ddp4_3k}"
PROMPT="${PROMPT:-pick tissue}"
CAMERA_SOURCE="${CAMERA_SOURCE:-live}"
CAMERA_HOST="${CAMERA_HOST:-192.168.123.165}"
CAMERA_PORT="${CAMERA_PORT:-5555}"
GT_CAMERA_EP="${GT_CAMERA_EP:-447}"
ZMQ_HOST="${ZMQ_HOST:-127.0.0.1}"
ZMQ_PORT="${ZMQ_PORT:-5556}"
STATE_ZMQ_HOST="${STATE_ZMQ_HOST:-127.0.0.1}"
STATE_ZMQ_PORT="${STATE_ZMQ_PORT:-5557}"
PROPRIO_SOURCE="${PROPRIO_SOURCE:-robot}"
CONTROL_FPS="${CONTROL_FPS:-50}"
INFERENCE_RATE="${INFERENCE_RATE:-2.5}"
MOTION_SECONDS="${MOTION_SECONDS:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WAIT_DEPLOY_STATE="${WAIT_DEPLOY_STATE:-0}"
WAIT_ROBOT_PROPRIO="${WAIT_ROBOT_PROPRIO:-0}"
RECORD_DIR="${RECORD_DIR:-}"
STREAM_NOW="${STREAM_NOW:-1}"
DEPLOY_KEYBOARD="${DEPLOY_KEYBOARD:-1}"
NO_ZMQ="${NO_ZMQ:-0}"

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export PYTHONPATH="${GR00T_ROOT}:${PHI0_ROOT}/src:${PYTHONPATH:-}"

echo "[phi0_sonic_closed_loop] camera_source=${CAMERA_SOURCE}"
if [[ "${CAMERA_SOURCE}" == "gt" ]]; then
  echo "[phi0_sonic_closed_loop] camera=GT episode ${GT_CAMERA_EP}"
else
  echo "[phi0_sonic_closed_loop] camera=SONIC composed_camera tcp://${CAMERA_HOST}:${CAMERA_PORT}"
fi
echo "[phi0_sonic_closed_loop] zmq=tcp://${ZMQ_HOST}:${ZMQ_PORT}"
echo "[phi0_sonic_closed_loop] state=tcp://${STATE_ZMQ_HOST}:${STATE_ZMQ_PORT}"
echo "[phi0_sonic_closed_loop] proprio_source=${PROPRIO_SOURCE} (robot|hybrid|roll-forward)"
echo "[phi0_sonic_closed_loop] checkpoint=${CHECKPOINT}"
echo "[phi0_sonic_closed_loop] prompt=${PROMPT}"
echo "[phi0_sonic_closed_loop] control_fps=${CONTROL_FPS} inference_rate=${INFERENCE_RATE}Hz"
echo "[phi0_sonic_closed_loop] deploy_keyboard=${DEPLOY_KEYBOARD} (k control loop, ] stream, p planner, O stop, h help)"
echo "[phi0_sonic_closed_loop] stream_now=${STREAM_NOW} (auto start deploy after first chunk)"
echo "[phi0_sonic_closed_loop] no_zmq=${NO_ZMQ} (1=record npz only, no deploy ZMQ)"

EXTRA_ARGS=()
if [[ "${WAIT_DEPLOY_STATE}" == "1" ]]; then
  EXTRA_ARGS+=(--wait-deploy-state)
fi
if [[ "${WAIT_ROBOT_PROPRIO}" == "1" ]]; then
  EXTRA_ARGS+=(--wait-robot-proprio)
fi
if [[ "${STREAM_NOW}" == "1" && "${NO_ZMQ}" != "1" ]]; then
  EXTRA_ARGS+=(--stream-now)
fi
if [[ "${DEPLOY_KEYBOARD}" == "0" ]]; then
  EXTRA_ARGS+=(--no-deploy-keyboard)
fi
if [[ "${NO_ZMQ}" == "1" ]]; then
  EXTRA_ARGS+=(--no-zmq)
fi
if [[ -n "${RECORD_DIR}" ]]; then
  mkdir -p "${RECORD_DIR}"
  EXTRA_ARGS+=(--record-dir "${RECORD_DIR}")
  echo "[phi0_sonic_closed_loop] record_dir=${RECORD_DIR}"
fi

exec "${PHI0_PY}" "${PHI0_ROOT}/scripts/phi0_sonic_closed_loop_zmq.py" \
  --checkpoint "${CHECKPOINT}" \
  --config-name "${CONFIG_NAME}" \
  --prompt "${PROMPT}" \
  --camera-source "${CAMERA_SOURCE}" \
  --gt-camera-episode "${GT_CAMERA_EP}" \
  --camera-host "${CAMERA_HOST}" \
  --camera-port "${CAMERA_PORT}" \
  --zmq-host "${ZMQ_HOST}" \
  --zmq-port "${ZMQ_PORT}" \
  --state-zmq-host "${STATE_ZMQ_HOST}" \
  --state-zmq-port "${STATE_ZMQ_PORT}" \
  --proprio-source "${PROPRIO_SOURCE}" \
  --control-fps "${CONTROL_FPS}" \
  --inference-rate "${INFERENCE_RATE}" \
  --motion-seconds "${MOTION_SECONDS}" \
  "${EXTRA_ARGS[@]}" \
  "$@"

#!/usr/bin/env bash
# Phi-0 SONIC publisher -> existing deploy + composed_camera (no sim, no deploy start).
#
# Prerequisites (you start these):
#   1. composed_camera on CAMERA_PORT (default 5555)
#   2. g1_deploy_onnx_ref --input-type zmq_manager --zmq-port ZMQ_PORT (default 5556)
#      in CONTROL + ZMQ streaming enabled (press ENTER on deploy if needed)
#
# This script only runs Phi-0 inference + ZMQ v4 token publisher on ZMQ_PORT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/setup_env.sh"
cd "${PHI0_ROOT}"

CHECKPOINT="${CHECKPOINT:-${PHI0_ROOT}/experiments/pick_tissue_xperience_unified_3k_ddp4_fast/pick_tissue_xperience_unified_act_latest.pt}"
CONFIG_NAME="${CONFIG_NAME:-train_pick_tissue_xperience_unified_ddp4_3k}"
UNIFIED_EP="${UNIFIED_EP:-447}"
CAMERA_HOST="${CAMERA_HOST:-192.168.123.165}"
CAMERA_PORT="${CAMERA_PORT:-5555}"
ZMQ_HOST="${ZMQ_HOST:-127.0.0.1}"
ZMQ_PORT="${ZMQ_PORT:-5556}"
CONTROL_FPS="${CONTROL_FPS:-50}"
MOTION_SECONDS="${MOTION_SECONDS:-8}"
RECORD_FPS="${RECORD_FPS:-30}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WORK_DIR="${WORK_DIR:-${PHI0_ROOT}/logs/phi0_sonic_deploy_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${WORK_DIR}"
OUT_MP4="${OUT_MP4:-${WORK_DIR}/ego_deploy_ep${UNIFIED_EP}.mp4}"
MOTION_NPZ="${MOTION_NPZ:-${WORK_DIR}/deploy_motion.npz}"

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export PYTHONPATH="${GR00T_ROOT}:${PHI0_ROOT}/src:${PYTHONPATH:-}"

echo "[phi0_sonic_connect] camera=tcp://${CAMERA_HOST}:${CAMERA_PORT}"
echo "[phi0_sonic_connect] zmq=tcp://${ZMQ_HOST}:${ZMQ_PORT} deploy must SUB this port"
echo "[phi0_sonic_connect] checkpoint=${CHECKPOINT}"
echo "[phi0_sonic_connect] episode_idx=${UNIFIED_EP} (proprio GT LUT; ego from live camera)"
echo "[phi0_sonic_connect] record_mp4=${OUT_MP4}"
echo "[phi0_sonic_connect] motion_npz=${MOTION_NPZ} (64D tokens + hands sent to deploy)"

if [[ "${CAMERA_HOST}" == "127.0.0.1" || "${CAMERA_HOST}" == "localhost" ]]; then
  if ! ss -tlnp 2>/dev/null | grep -qE ":${CAMERA_PORT}\\b"; then
    echo "[warn] nothing listening locally on ${CAMERA_PORT} — start composed_camera first" >&2
  fi
else
  echo "[phi0_sonic_connect] remote camera — ensure tcp://${CAMERA_HOST}:${CAMERA_PORT} reachable"
fi

exec "${PHI0_PY}" "${PHI0_ROOT}/scripts/phi0_sonic_latent_zmq_publisher.py" \
  --checkpoint "${CHECKPOINT}" \
  --config-name "${CONFIG_NAME}" \
  --episode-idx "${UNIFIED_EP}" \
  --camera-host "${CAMERA_HOST}" \
  --camera-port "${CAMERA_PORT}" \
  --zmq-host "${ZMQ_HOST}" \
  --zmq-port "${ZMQ_PORT}" \
  --control-fps "${CONTROL_FPS}" \
  --motion-seconds "${MOTION_SECONDS}" \
  --record-mp4 "${OUT_MP4}" \
  --record-motion-npz "${MOTION_NPZ}" \
  --record-fps "${RECORD_FPS}" \
  --stream-now \
  "$@"

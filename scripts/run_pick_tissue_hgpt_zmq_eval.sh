#!/usr/bin/env bash
# Pick-tissue Phi-0 unified -> ZMQ -> Humanoid-GPT tracker sim mp4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHI0_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HGPT_ROOT="$(cd "${SCRIPT_DIR}/../../Humanoid-GPT-main" && pwd)"
WORK_DIR="${WORK_DIR:-${HGPT_ROOT}/experiments/phi0_hgpt_zmq/runs/pick_tissue_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${WORK_DIR}/logs"
mkdir -p "${LOG_DIR}"

CHECKPOINT="${CHECKPOINT:-${PHI0_ROOT}/experiments/pick_tissue_xperience_unified_3k_ddp4_fast/pick_tissue_xperience_unified_act_latest.pt}"
CONFIG_NAME="${CONFIG_NAME:-train_pick_tissue_xperience_unified_ddp4_8k}"
CLIP_IDX="${CLIP_IDX:-0}"
USE_GT="${USE_GT:-0}"
DEPLOY_MODE="${DEPLOY_MODE:-smpl}"
# qpos GT replay: default no EMA (faithful WBC joints); model deploy keeps HGPT default
if [[ "${DEPLOY_MODE}" == "qpos" && "${USE_GT}" == "1" && -z "${EMA_ALPHA+x}" ]]; then
  EMA_ALPHA=0
else
  EMA_ALPHA="${EMA_ALPHA:-0.55}"
fi
ZMQ_PORT="${ZMQ_PORT:-5560}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
STAND_SECONDS="${STAND_SECONDS:-2.0}"
# pick-tissue timeline is 50 Hz
MOTION_SECONDS="${MOTION_SECONDS:-8}"
CONTROL_FPS="${CONTROL_FPS:-50}"
# Publisher loads VLM + multi-chunk infer can exceed 120s for long clips
_motion_int="${MOTION_SECONDS%%.*}"
RECV_TIMEOUT_MS="${RECV_TIMEOUT_MS:-$((180000 + _motion_int * 45000))}"
CAM_DISTANCE="${CAM_DISTANCE:-2.0}"
CAM_AZIMUTH="${CAM_AZIMUTH:-90}"
SHOW_GT_VIEWS="${SHOW_GT_VIEWS:-1}"
OUT_MP4="${OUT_MP4:-${WORK_DIR}/pick_tissue_clip${CLIP_IDX}_tracker.mp4}"

PHI0_PY="${PHI0_PY:-/mnt/data/miniconda3/envs/Phi-0-wpy/bin/python}"
HGPT_PY="${HGPT_PY:-/mnt/data/miniconda3/envs/Humanoid-gpt-wpy/bin/python}"

# Optional: MANIFEST_SESSION + MANIFEST_EP -> unified episode_index (avoids clip-shuffle / filename gaps)
EPISODE_IDX="${EPISODE_IDX:-}"
if [[ -n "${MANIFEST_SESSION:-}" && -n "${MANIFEST_EP:-}" ]]; then
  EPISODE_IDX="$(
    PHI0_ROOT="${PHI0_ROOT}" MANIFEST_SESSION="${MANIFEST_SESSION}" MANIFEST_EP="${MANIFEST_EP}" \
    MANIFEST_PATH="${MANIFEST_PATH:-/mnt/data2/wpy/workspace/Isaac-GR00T/data/pick_tissues.json}" \
    PICK_TISSUE_VALID="${PICK_TISSUE_VALID:-/mnt/data2/wpy/workspace/Isaac-GR00T/data/pick_tissue_valid}" \
    "${PHI0_PY}" - <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.environ["PHI0_ROOT"], "src"))
from phi0.data.pick_tissue_episode_map import manifest_ep_to_unified_episode_index
print(manifest_ep_to_unified_episode_index(
    os.environ["MANIFEST_PATH"],
    os.environ["PICK_TISSUE_VALID"],
    os.environ["MANIFEST_SESSION"],
    int(os.environ["MANIFEST_EP"]),
))
PY
  )"
  echo "[pick_tissue_hgpt] manifest ${MANIFEST_SESSION} ep${MANIFEST_EP} -> unified episode_idx=${EPISODE_IDX}"
fi
EPISODE_ARGS=()
if [[ -n "${EPISODE_IDX}" ]]; then
  EPISODE_ARGS=(--episode-idx "${EPISODE_IDX}")
  if [[ -z "${OUT_MP4_SET:-}" ]]; then
    OUT_MP4="${WORK_DIR}/pick_tissue_ep${EPISODE_IDX}_tracker.mp4"
  fi
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

SUB_PID=""
cleanup() {
  [[ -n "${SUB_PID}" ]] && kill "${SUB_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[pick_tissue_hgpt] checkpoint=${CHECKPOINT} clip=${CLIP_IDX} episode_idx=${EPISODE_IDX:-clip} deploy_mode=${DEPLOY_MODE} out=${OUT_MP4}"

echo "[1/2] Starting Humanoid-GPT subscriber..."
cd "${HGPT_ROOT}"
GT_VIEW_ARGS=(--show-gt-views)
if [[ "${SHOW_GT_VIEWS}" == "0" ]]; then
  GT_VIEW_ARGS=(--no-show-gt-views)
fi
"${HGPT_PY}" -m experiments.phi0_hgpt_zmq.hgpt_zmq_tracker_sim \
  --connect "tcp://127.0.0.1:${ZMQ_PORT}" \
  --stand-seconds "${STAND_SECONDS}" \
  --recv-timeout-ms "${RECV_TIMEOUT_MS}" \
  --video-path "${OUT_MP4}" \
  --device cpu \
  --show-ref-ghost \
  --cam-distance "${CAM_DISTANCE}" \
  --cam-azimuth "${CAM_AZIMUTH}" \
  "${GT_VIEW_ARGS[@]}" \
  > "${LOG_DIR}/hgpt_sub.log" 2>&1 &
SUB_PID=$!
sleep 3

GT_ARGS=()
if [[ "${USE_GT}" == "1" ]]; then
  GT_ARGS=(--use-gt --gt-replay-only)
fi

echo "[2/2] Phi-0 publisher burst use_gt=${USE_GT} deploy_mode=${DEPLOY_MODE} motion_s=${MOTION_SECONDS} fps=${CONTROL_FPS}..."
cd "${PHI0_ROOT}"
"${PHI0_PY}" -m experiments.phi0_hgpt_zmq.phi0_zmq_publisher \
  --checkpoint "${CHECKPOINT}" \
  --config-name "${CONFIG_NAME}" \
  --clip-idx "${CLIP_IDX}" \
  "${EPISODE_ARGS[@]}" \
  --bind "tcp://*:${ZMQ_PORT}" \
  --control-fps "${CONTROL_FPS}" \
  --deploy-mode "${DEPLOY_MODE}" \
  --ema-alpha "${EMA_ALPHA}" \
  --burst \
  --device cuda \
  --motion-seconds "${MOTION_SECONDS}" \
  "${GT_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/phi0_pub.log"

wait "${SUB_PID}" || {
  echo "Subscriber failed; see ${LOG_DIR}/hgpt_sub.log"
  tail -40 "${LOG_DIR}/hgpt_sub.log"
  exit 1
}

echo "[done] video=${OUT_MP4}"

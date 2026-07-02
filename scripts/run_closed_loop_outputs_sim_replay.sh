#!/usr/bin/env bash
# Replay closed-loop (or any) motion npz -> MuJoCo sim + SONIC deploy -> mp4
#
# Usage:
#   bash scripts/run_closed_loop_outputs_sim_replay.sh logs/my_closed_loop_run/outputs.npz
#
# Closed-loop outputs.npz already includes hand ramp on left/right -> HAND_RAMP_FRAMES=0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/setup_env.sh"

MOTION_NPZ="${1:-${MOTION_NPZ:-${PHI0_ROOT}/logs/my_closed_loop_run/outputs.npz}}"
if [[ ! -f "${MOTION_NPZ}" ]]; then
  echo "[sim_replay] ERROR: motion npz not found: ${MOTION_NPZ}" >&2
  exit 1
fi

export MOTION_NPZ
export HAND_RAMP_FRAMES="${HAND_RAMP_FRAMES:-0}"
export UNIFIED_EP="${UNIFIED_EP:-447}"
export CONTROL_FPS="${CONTROL_FPS:-50}"
export MAX_FRAMES="${MAX_FRAMES:-0}"
export CHECKPOINT=""
export ROBOT_ONLY="${ROBOT_ONLY:-1}"
export GT_PANEL_LAYOUT="${GT_PANEL_LAYOUT:-robot}"
export ENABLE_G1_DEBUG_OVERLAY="${ENABLE_G1_DEBUG_OVERLAY:-0}"
export RECORD_FPS="${RECORD_FPS:-20}"
export SIM_WARMUP_S="${SIM_WARMUP_S:-10}"
export RECORD_SETTLE_S="${RECORD_SETTLE_S:-5}"
export WORK_DIR="${WORK_DIR:-${PHI0_ROOT}/logs/closed_loop_sim_replay_$(date +%Y%m%d_%H%M%S)}"
export OUT_MP4="${OUT_MP4:-${WORK_DIR}/closed_loop_replay.mp4}"

FRAMES="$("${PHI0_PY}" - <<PY
import numpy as np
print(int(np.load("${MOTION_NPZ}")["tokens"].shape[0]))
PY
)"
export MOTION_SECONDS="${MOTION_SECONDS:-$(python3 -c "print(round(${FRAMES}/${CONTROL_FPS}, 2))")}"

echo "[sim_replay] MOTION_NPZ=${MOTION_NPZ}"
echo "[sim_replay] frames=${FRAMES} (~${MOTION_SECONDS}s @ ${CONTROL_FPS}Hz)"
echo "[sim_replay] HAND_RAMP_FRAMES=${HAND_RAMP_FRAMES} (0 for closed-loop outputs)"
echo "[sim_replay] GT_PANEL_LAYOUT=${GT_PANEL_LAYOUT} (sim=MuJoCo camera, top=GT ego/wrist panels)"
echo "[sim_replay] OUT_MP4=${OUT_MP4}"

exec bash "${SCRIPT_DIR}/run_pick_tissue_sonic_latent_eval.sh"

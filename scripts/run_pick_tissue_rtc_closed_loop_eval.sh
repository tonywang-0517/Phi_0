#!/usr/bin/env bash: closed-loop rollout (GT ego + async re-infer + deploy RTC blend)
# -> MuJoCo sim + SONIC deploy mp4 replay.
#
# Usage:
#   bash scripts/run_pick_tissue_rtc_closed_loop_eval.sh
#   bash scripts/run_pick_tissue_rtc_closed_loop_eval.sh 447 logs/my_run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/setup_env.sh"
cd "${PHI0_ROOT}"

EPISODE_IDX="${1:-447}"
RUN_DIR="${2:-${PHI0_ROOT}/logs/ep${EPISODE_IDX}_rtc_closed_loop_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT="${CHECKPOINT:-${PHI0_ROOT}/experiments/pick_tissue_valid_wholebody_rtc_8k_ddp4/pick_tissue_valid_wholebody_rtc_act_latest.pt}"
CONFIG_NAME="${CONFIG_NAME:-train_pick_tissue_finetune_rtc_ddp4}"
INFERENCE_RATE="${INFERENCE_RATE:-2.5}"
MOTION_SECONDS="${MOTION_SECONDS:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export CUDA_VISIBLE_DEVICES
mkdir -p "${RUN_DIR}"
OUT_NPZ="${RUN_DIR}/outputs.npz"
OUT_MP4="${RUN_DIR}/closed_loop_replay.mp4"

echo "[rtc_eval] episode=${EPISODE_IDX} run_dir=${RUN_DIR}"
echo "[rtc_eval] checkpoint=${CHECKPOINT}"
echo "[rtc_eval] config=${CONFIG_NAME} (rtc.enabled from model cfg)"
echo "[rtc_eval] inference_rate=${INFERENCE_RATE}Hz motion_seconds=${MOTION_SECONDS}"

echo "[rtc_eval] stage 1/2: closed-loop rollout -> ${OUT_NPZ}"
CHECKPOINT="${CHECKPOINT}" CONFIG_NAME="${CONFIG_NAME}" \
  INFERENCE_RATE="${INFERENCE_RATE}" MOTION_SECONDS="${MOTION_SECONDS}" \
  bash "${SCRIPT_DIR}/run_closed_loop_episode_rollout.sh" "${EPISODE_IDX}" "${OUT_NPZ}"

echo "[rtc_eval] stage 2/2: sim + deploy replay -> ${OUT_MP4}"
MOTION_NPZ="${OUT_NPZ}" OUT_MP4="${OUT_MP4}" WORK_DIR="${RUN_DIR}/sim_replay" \
  GT_PANEL_LAYOUT="${GT_PANEL_LAYOUT:-sim}" HAND_RAMP_FRAMES=0 \
  bash "${SCRIPT_DIR}/run_closed_loop_outputs_sim_replay.sh" "${OUT_NPZ}"

echo "[rtc_eval] done"
echo "[rtc_eval] outputs.npz=${OUT_NPZ}"
echo "[rtc_eval] mp4=${OUT_MP4}"

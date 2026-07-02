#!/usr/bin/env bash
# Offline closed-loop rollout on a pick-tissue unified episode -> outputs.npz
#
# Usage:
#   bash scripts/run_closed_loop_episode_rollout.sh 447
#   bash scripts/run_closed_loop_episode_rollout.sh 447 logs/ep447_rollout/outputs.npz
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/setup_env.sh"
cd "${PHI0_ROOT}"

EPISODE_IDX="${1:-447}"
OUT_NPZ="${2:-${PHI0_ROOT}/logs/episode_${EPISODE_IDX}_closed_loop/outputs.npz}"
CHECKPOINT="${CHECKPOINT:-${PHI0_ROOT}/experiments/pick_tissue_valid_wholebody_rtc_8k_ddp4/pick_tissue_valid_wholebody_rtc_act_latest.pt}"
CONFIG_NAME="${CONFIG_NAME:-train_pick_tissue_finetune_rtc_ddp4}"
MOTION_SECONDS="${MOTION_SECONDS:-0}"
INFERENCE_RATE="${INFERENCE_RATE:-2.5}"

mkdir -p "$(dirname "${OUT_NPZ}")"

echo "[episode_rollout] episode_idx=${EPISODE_IDX}"
echo "[episode_rollout] output=${OUT_NPZ}"
echo "[episode_rollout] checkpoint=${CHECKPOINT}"
echo "[episode_rollout] motion_seconds=${MOTION_SECONDS} (0=full episode)"

exec "${PHI0_PY}" "${PHI0_ROOT}/scripts/phi0_sonic_closed_loop_episode_rollout.py" \
  --episode-idx "${EPISODE_IDX}" \
  --output "${OUT_NPZ}" \
  --checkpoint "${CHECKPOINT}" \
  --config-name "${CONFIG_NAME}" \
  --motion-seconds "${MOTION_SECONDS}" \
  --inference-rate "${INFERENCE_RATE}"

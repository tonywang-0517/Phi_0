#!/usr/bin/env bash
# 4-GPU DDP: pick-tissue xperience unified (512-d), 3k steps.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
CONFIG="${CONFIG:-train_pick_tissue_xperience_unified_ddp4_3k}"
EXP="${EXP:-experiments/pick_tissue_xperience_unified_3k_ddp4_fast}"
MAX_STEPS="${MAX_STEPS:-3000}"
SAVE_EVERY="${SAVE_EVERY:-3000}"
exec env CONFIG="${CONFIG}" EXP="${EXP}" MAX_STEPS="${MAX_STEPS}" SAVE_EVERY="${SAVE_EVERY}" \
  bash "${ROOT}/scripts/run_train_pick_tissue_xperience_unified_ddp4_8k.sh" "$@"

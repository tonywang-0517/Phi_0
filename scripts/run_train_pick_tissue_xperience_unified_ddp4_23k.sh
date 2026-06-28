#!/usr/bin/env bash
# 4-GPU DDP: resume pick-tissue 3k fast → 23k total (+20k steps).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
CONFIG="${CONFIG:-train_pick_tissue_xperience_unified_ddp4_23k}"
EXP="${EXP:-experiments/pick_tissue_xperience_unified_3k_ddp4_fast}"
MAX_STEPS="${MAX_STEPS:-23000}"
SAVE_EVERY="${SAVE_EVERY:-4000}"
AUTO_RESUME="${AUTO_RESUME:-true}"
exec env CONFIG="${CONFIG}" EXP="${EXP}" MAX_STEPS="${MAX_STEPS}" SAVE_EVERY="${SAVE_EVERY}" AUTO_RESUME="${AUTO_RESUME}" \
  bash "${ROOT}/scripts/run_train_pick_tissue_xperience_unified_ddp4_8k.sh" "$@"

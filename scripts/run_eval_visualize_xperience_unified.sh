#!/usr/bin/env bash
# Eval + FK skeleton viz for Xperience unified checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

CKPT="${CKPT:-experiments/xperience_unified_act_1k_ddp4/xperience_unified_act_latest.pt}"
OUT="${OUT:-}"
MAX_CLIPS="${MAX_CLIPS:-16}"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ARGS=(
  --checkpoint "${CKPT}"
  --config-name train_xperience_unified
  --device cuda
  --max-clips "${MAX_CLIPS}"
  --viz-clips 0 8 16 24
  --viz-stride 2
  --make-gif
)
if [[ -n "${OUT}" ]]; then
  ARGS+=(--output-dir "${OUT}")
fi

exec conda run --no-capture-output -n Phi-0-wpy python scripts/eval_visualize_xperience_unified.py "${ARGS[@]}" "$@"

#!/usr/bin/env bash
# LIBERO delta-EEF eval + rollout videos (same env as training).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EXP="${EXP:-experiments/libero_spatial_act_delta_8_30k}"
CKPT="${CKPT:-${EXP}/libero_spatial_act_delta_8_30k_latest.pt}"
DEVICE="${CUDA_VISIBLE_DEVICES:-1}"
MAX_TASKS="${LIBERO_MAX_TASKS:-5}"

export CUDA_VISIBLE_DEVICES="$DEVICE"
export PYTHONPATH="${ROOT}/src:${ROOT}/../FastWAM/src:${ROOT}/../vggt-omega"
export PYTHONUNBUFFERED=1

mkdir -p "${EXP}/eval_videos"

exec conda run --no-capture-output -n Phi-0-wpy python scripts/eval_vla_benchmark.py \
  --benchmark libero \
  --checkpoint "$CKPT" \
  --config-name train_libero_spatial_act_delta_30k \
  --device cuda \
  --min-free-gb 16 \
  --num-open-loop-steps 8 \
  --action-mode robot7d \
  --no-libero-osc-absolute \
  --libero-suite libero_spatial \
  --libero-trials-per-task 1 \
  --libero-max-tasks "$MAX_TASKS" \
  --save-videos \
  --video-dir "${EXP}/eval_videos" \
  --output "${EXP}/eval_report_step8000.json"

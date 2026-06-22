#!/usr/bin/env bash
# Eval new 35k ablations: wrist + act12 (10 tasks x 1 trial each).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

DEVICE="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES="${DEVICE}"
export USE_TF=0
export PYTHONPATH="${ROOT}/src:${ROOT}/../FastWAM/src:${ROOT}/../vggt-omega:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

run_eval() {
  local label="$1"
  local exp="$2"
  local ckpt_name="$3"
  local config="$4"
  local ckpt="${ROOT}/${exp}/${ckpt_name}_latest.pt"
  local out="${ROOT}/${exp}/eval_report_35k_1trial.json"
  local log="${ROOT}/${exp}/eval_35k_1trial.log"

  if [[ ! -f "${ckpt}" ]]; then
    echo "SKIP ${label}: missing ${ckpt}" >&2
    return 1
  fi

  echo "==> Eval ${label}"
  echo "    ckpt=${ckpt}"
  echo "    config=${config}"
  echo "    output=${out}"

  conda run --no-capture-output -n Phi-0-wpy python scripts/eval_vla_benchmark.py \
    --benchmark libero \
    --checkpoint "${ckpt}" \
    --config-name "${config}" \
    --device cuda \
    --min-free-gb 16 \
    --num-open-loop-steps 8 \
    --action-mode robot7d \
    --no-libero-osc-absolute \
    --libero-suite libero_spatial \
    --libero-trials-per-task 1 \
    --output "${out}" \
    2>&1 | tee "${log}"

  echo "==> Done ${label}: success_rate=$(python -c "import json; print(json.load(open('${out}'))['success_rate'])")"
}

run_eval "wrist (VLM+dual cam)" \
  "experiments/libero_spatial_vlm_wrist_35k_ddp4" \
  "libero_spatial_vlm_wrist_35k_ddp4" \
  "train_libero_spatial_vlm_wrist_35k_ddp4"

run_eval "act12 (VLM+12L ACT lr=5e-5)" \
  "experiments/libero_spatial_vlm_act12_35k_ddp4" \
  "libero_spatial_vlm_act12_35k_ddp4" \
  "train_libero_spatial_vlm_act12_35k_ddp4"

echo "==> Eval reports: experiments/*/eval_report_35k_1trial.json"

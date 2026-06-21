#!/usr/bin/env bash
# Eval all four LIBERO spatial ablations: 10 tasks x 1 trial each.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${DEVICE}"
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
  local out="${ROOT}/${exp}/eval_report_15k_1trial.json"

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
    2>&1 | tee "${ROOT}/${exp}/eval_15k_1trial.log"

  echo "==> Done ${label}: ${out}"
}

run_eval "dual (VLM+VGGT)" \
  "experiments/libero_spatial_vlm_dual_15k_single" \
  "libero_spatial_vlm_dual_15k_single" \
  "train_libero_spatial_act_delta_15k_single"

run_eval "vlm_only bs16" \
  "experiments/libero_spatial_vlm_only_15k_single" \
  "libero_spatial_vlm_only_15k_single" \
  "train_libero_spatial_act_delta_15k_single_vlm_only"

run_eval "vlm_only bs128" \
  "experiments/libero_spatial_vlm_only_15k_single_bs128" \
  "libero_spatial_vlm_only_15k_single_bs128" \
  "train_libero_spatial_act_delta_15k_single_vlm_only_bs128"

run_eval "action_only" \
  "experiments/libero_spatial_action_only_15k_single" \
  "libero_spatial_action_only_15k_single" \
  "train_libero_spatial_act_delta_15k_single_action_only"

echo "==> All eval reports written under experiments/*/eval_report_15k_1trial.json"

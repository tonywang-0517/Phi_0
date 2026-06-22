#!/usr/bin/env bash
# Re-verify eval success rates for main ablations (1 trial each).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
source "${ROOT}/.env" 2>/dev/null || true
export USE_TF=0
export PYTHONPATH="${ROOT}/src:${ROOT}/../FastWAM/src:${ROOT}/../vggt-omega:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

run_one() {
  local gpu="$1" exp="$2" ckpt_name="$3" config="$4"
  CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n Phi-0-wpy python scripts/eval_vla_benchmark.py \
    --benchmark libero \
    --checkpoint "${ROOT}/${exp}/${ckpt_name}_latest.pt" \
    --config-name "${config}" \
    --device cuda --min-free-gb 16 --num-open-loop-steps 8 \
    --action-mode robot7d --no-libero-osc-absolute \
    --libero-suite libero_spatial --libero-trials-per-task 1 \
    --output "${ROOT}/${exp}/eval_report_35k_1trial_verify.json" \
    > "${ROOT}/${exp}/eval_35k_1trial_verify.log" 2>&1
}

run_one 1 experiments/libero_spatial_vlm_only_35k_ddp4 libero_spatial_vlm_only_35k_ddp4 train_libero_spatial_vlm_only_35k_ddp4 &
run_one 2 experiments/libero_spatial_vlm_wrist_35k_ddp4 libero_spatial_vlm_wrist_35k_ddp4 train_libero_spatial_vlm_wrist_35k_ddp4 &
run_one 3 experiments/libero_spatial_vlm_dual_35k_ddp4 libero_spatial_vlm_dual_35k_ddp4 train_libero_spatial_vlm_dual_35k_ddp4 &
run_one 5 experiments/libero_spatial_vlm_act12_35k_ddp4 libero_spatial_vlm_act12_35k_ddp4 train_libero_spatial_vlm_act12_35k_ddp4 &
wait

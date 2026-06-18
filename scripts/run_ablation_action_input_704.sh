#!/usr/bin/env bash
# Action-input ablation @ 704x1280, 400 steps: baseline vs DiT4DiT prefix/query (4+29).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${ROOT}/../FastWAM/src:${ROOT}/../vggt-omega"

GPU_BASE="${GPU_BASE:-2}"
GPU_DIT4DIT="${GPU_DIT4DIT:-3}"
EXP_BASE="experiments/ablation_baseline_704_400step"
EXP_DIT="experiments/ablation_dit4dit_query_704_400step"

mkdir -p "${EXP_BASE}" "${EXP_DIT}"

run_train() {
  local gpu="$1" config="$2" log="$3"
  echo "GPU${gpu}: train ${config} -> ${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/train.py --config-name "${config}" \
    device=cuda mixed_precision=bf16 \
    2>&1 | tee "${log}"
}

run_post() {
  local gpu="$1" config="$2" exp="$3" name="$4"
  local ckpt="${exp}/${name}_latest.pt"
  echo "GPU${gpu}: post-train eval/viz ${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/benchmark_deploy.py \
    --checkpoint "${ckpt}" \
    --config-name "${config}" \
    --device cuda \
    --deploy-seconds 5 \
    --output "${exp}/benchmark_deploy.jsonl" \
    --benchmark-json "${exp}/inference_benchmark.json" \
    2>&1 | tee "${exp}/benchmark.log"
  python scripts/visualize_skeleton.py \
    --predictions "${exp}/benchmark_deploy.jsonl" \
    --output-dir "${exp}/viz_skeleton_5s" \
    --max-frames 100 --fps 15 \
    2>&1 | tee "${exp}/viz.log"
}

echo "=== Phase 1: training (parallel) ==="
run_train "${GPU_BASE}" train_ablation_baseline_704_400 "${EXP_BASE}/train.log" &
PID1=$!
run_train "${GPU_DIT4DIT}" train_ablation_dit4dit_query_704_400 "${EXP_DIT}/train.log" &
PID2=$!
wait "${PID1}" "${PID2}"

echo "=== Phase 2: deploy + skeleton viz ==="
run_post "${GPU_BASE}" train_ablation_baseline_704_400 "${EXP_BASE}" ablation_baseline_704_400step &
PID3=$!
run_post "${GPU_DIT4DIT}" train_ablation_dit4dit_query_704_400 "${EXP_DIT}" ablation_dit4dit_query_704_400step &
PID4=$!
wait "${PID3}" "${PID4}"

echo "Done."
echo "  baseline: ${EXP_BASE}/viz_skeleton_5s/skeleton_gt_vs_pred_sidebyside.gif"
echo "  dit4dit:  ${EXP_DIT}/viz_skeleton_5s/skeleton_gt_vs_pred_sidebyside.gif"

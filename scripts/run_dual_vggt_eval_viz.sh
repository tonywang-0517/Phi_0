#!/usr/bin/env bash
# Eval + deploy benchmark + skeleton viz for dual VGGT checkpoint; refresh loss plots.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EXP="${1:-experiments/phi0_act_dual_vggt_800step}"
CKPT="${EXP}/phi0_act_dual_vggt_800step_latest.pt"
DEVICE="${CUDA_VISIBLE_DEVICES:-2}"

export CUDA_VISIBLE_DEVICES="$DEVICE"
export PYTHONPATH="${ROOT}/src:${ROOT}/../FastWAM/src"

mkdir -p "$EXP"

echo "=== eval_action ==="
conda run -n Phi-0-wpy python scripts/eval_action.py \
  --checkpoint "$CKPT" \
  --config-name train_act_dual_vggt \
  --device cuda \
  --deploy-seconds 5 \
  --output "$EXP/eval_report_5s.json" \
  2>&1 | tee "$EXP/eval.log"

echo "=== benchmark_deploy ==="
conda run -n Phi-0-wpy python scripts/benchmark_deploy.py \
  --checkpoint "$CKPT" \
  --config-name train_act_dual_vggt \
  --device cuda \
  --deploy-seconds 5 \
  --output "$EXP/benchmark_deploy.jsonl" \
  2>&1 | tee "$EXP/benchmark.log"

echo "=== visualize_skeleton ==="
conda run -n Phi-0-wpy python scripts/visualize_skeleton.py \
  --predictions "$EXP/benchmark_deploy.jsonl" \
  --output-dir "$EXP/viz_skeleton_5s" \
  --max-frames 100 \
  --fps 15 \
  2>&1 | tee "$EXP/viz.log"

echo "=== extract dual train log ==="
grep -E 'step=[0-9]+ loss=' "$EXP/nohup.out" | sed 's/.*\[phi0.runtime\]\[INFO\] - //' > "$ROOT/experiments/phi0_act_dual_vggt_800step_train.log" || true

echo "=== plot_training_loss ==="
conda run -n Phi-0-wpy python scripts/plot_training_loss.py --y-max 1.0

echo "=== compare_dual_vggt_baseline ==="
conda run -n Phi-0-wpy python scripts/compare_dual_vggt_baseline.py

echo "Done. Compare with experiments/phi0_act_proprio_800step/"

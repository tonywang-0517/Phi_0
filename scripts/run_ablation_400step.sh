#!/usr/bin/env bash
# Launch 3 ablation runs (400 steps) on GPUs 2, 3, 5.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src

mkdir -p experiments

run_one() {
  local gpu="$1"
  local config="$2"
  local log="$3"
  echo "GPU${gpu}: ${config} -> ${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/train.py --config-name "${config}" \
    2>&1 | tee "${log}"
}

run_one 2 train_ablation1_vggt_full experiments/ablation1_vggt_full_400step_train.log &
PID1=$!
run_one 3 train_ablation2_dit4dit_query experiments/ablation2_dit4dit_query_400step_train.log &
PID2=$!
run_one 5 train_ablation3_both experiments/ablation3_both_400step_train.log &
PID3=$!

wait "$PID1" "$PID2" "$PID3"
python scripts/plot_ablation_loss.py

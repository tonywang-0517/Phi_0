#!/usr/bin/env bash
# Wait for dual VGGT 800-step training, then eval + viz + loss plot + baseline compare.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="experiments/phi0_act_dual_vggt_800step_train.log"
CKPT="experiments/phi0_act_dual_vggt_800step/phi0_act_dual_vggt_800step_latest.pt"

echo "Waiting for step=799 in $LOG ..."
while true; do
  if grep -q "step=799 " "$LOG" 2>/dev/null; then
    break
  fi
  if grep -qE "Error executing job|Traceback" "$LOG" 2>/dev/null; then
    echo "Training failed; see $LOG"
    exit 1
  fi
  sleep 60
done

echo "Waiting for checkpoint $CKPT ..."
while [ ! -f "$CKPT" ]; do
  sleep 30
done
sleep 10

bash scripts/run_dual_vggt_eval_viz.sh

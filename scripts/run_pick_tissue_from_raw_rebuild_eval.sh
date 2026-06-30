#!/usr/bin/env bash
# pick-tissue: raw sessions (Isaac-GR00T/data/2026-*) -> valid -> unified v2.8 -> eval ep2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHI0_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAAC_ROOT="$(cd "${PHI0_ROOT}/../Isaac-GR00T" && pwd)"
WORK_DIR="${WORK_DIR:-${PHI0_ROOT}/../logs/pick_tissue_finetune/from_raw_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${WORK_DIR}/logs"
mkdir -p "${LOG_DIR}"

PHI0_PY="${PHI0_PY:-/mnt/data/miniconda3/envs/Phi-0-wpy/bin/python}"
MANIFEST="${MANIFEST:-${ISAAC_ROOT}/data/pick_tissues.json}"
RAW_ROOT="${RAW_ROOT:-${ISAAC_ROOT}/data}"
VALID_ROOT="${VALID_ROOT:-${ISAAC_ROOT}/data/pick_tissue_valid}"
UNIFIED_OUT="${UNIFIED_OUT:-${ISAAC_ROOT}/data/pick_tissue_xperience_unified}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PYTHONUNBUFFERED=1

echo "[from_raw] manifest=${MANIFEST}"
echo "[from_raw] raw_root=${RAW_ROOT} -> valid=${VALID_ROOT} -> unified=${UNIFIED_OUT}"
echo "[from_raw] logs=${LOG_DIR}"

echo "[from_raw] (1/4) prepare_pick_tissue_dataset..."
"${PHI0_PY}" "${ISAAC_ROOT}/scripts/prepare_pick_tissue_dataset.py" \
  --manifest-path "${MANIFEST}" \
  --raw-root "${RAW_ROOT}" \
  --dst-root "${VALID_ROOT}" \
  2>&1 | tee "${LOG_DIR}/01_prepare.log"

echo "[from_raw] (2/4) check observation.base_trans in valid..."
"${PHI0_PY}" - <<PY | tee "${LOG_DIR}/02_base_trans_check.log"
import pandas as pd
from pathlib import Path
root = Path("${VALID_ROOT}") / "data" / "chunk-000"
parquets = sorted(root.glob("episode_*.parquet"))
sample = pd.read_parquet(parquets[0])
has = "observation.base_trans" in sample.columns
print(f"episodes={len(parquets)} sample={parquets[0].name} has_base_trans={has} ncols={len(sample.columns)}")
if has:
    import numpy as np
    bt = np.stack(sample["observation.base_trans"].values)
    print(f"sample xyz_span={bt.max(0)-bt.min(0)}")
PY

echo "[from_raw] (3/4) rebuild unified v2.8..."
"${PHI0_PY}" "${PHI0_ROOT}/scripts/data/isaac_groot_to_xperience_unified_lerobot.py" \
  --data-root "${VALID_ROOT}" \
  --out-dir "${UNIFIED_OUT}" \
  --num-workers 8 \
  2>&1 | tee "${LOG_DIR}/03_rebuild_v27.log"

echo "[from_raw] verify ep524..."
"${PHI0_PY}" "${PHI0_ROOT}/scripts/data/verify_pick_tissue_qpos_labels.py" --dst-ep 524 \
  2>&1 | tee "${LOG_DIR}/04_verify.log"

echo "[from_raw] (4/4) GT replay eval manifest ep2..."
EVAL_DIR="${WORK_DIR}/eval"
mkdir -p "${EVAL_DIR}"
WORK_DIR="${EVAL_DIR}" \
MANIFEST_SESSION=2026-06-25-16-09-43 MANIFEST_EP=2 \
DEPLOY_MODE=qpos USE_GT=1 SHOW_GT_VIEWS=0 MOTION_SECONDS=8 \
bash "${PHI0_ROOT}/scripts/run_pick_tissue_hgpt_zmq_eval.sh" \
  2>&1 | tee "${LOG_DIR}/05_eval.log"

echo "[from_raw] done work_dir=${WORK_DIR}"

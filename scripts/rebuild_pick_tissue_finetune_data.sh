#!/usr/bin/env bash
# Rebuild pick-tissue valid + sonic-unified (43s/100a, ego+left_wrist) training data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_GR00T="$(cd "${SCRIPT_DIR}/../../Isaac-GR00T" && pwd)"
PHI0_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-${PHI0_ROOT}/.venv-openpi/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="${PYTHON:-python3}"
fi

DATA_ROOT="${DATA_ROOT:-${ISAAC_GR00T}/data}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/pick_tissues.json}"
VALID_ROOT="${VALID_ROOT:-${DATA_ROOT}/pick_tissue_valid}"
SONIC_ROOT="${SONIC_ROOT:-${DATA_ROOT}/pick_tissue_sonic_unified}"
MODALITY_CONFIG="${MODALITY_CONFIG:-${ISAAC_GR00T}/examples/G1_SONIC/g1_sonic_ego_left_wrist_config.py}"
NUM_WORKERS="${NUM_WORKERS:-8}"

export PYTHONPATH="${PHI0_ROOT}/src:${PYTHONPATH:-}"

echo "[rebuild] step 1/3: merge valid episodes -> ${VALID_ROOT}"
"${PYTHON}" "${ISAAC_GR00T}/scripts/prepare_pick_tissue_dataset.py" \
  --manifest-path "${MANIFEST}" \
  --raw-root "${DATA_ROOT}" \
  --dst-root "${VALID_ROOT}"

echo "[rebuild] step 2/3: GR00T stats for ${VALID_ROOT}"
GR00T_PYTHON="${GR00T_PYTHON:-/mnt/data/miniconda3/envs/Phi-0-wpy/bin/python}"
if [[ -x "${GR00T_PYTHON}" && -f "${ISAAC_GR00T}/gr00t/data/stats.py" ]]; then
  (cd "${ISAAC_GR00T}" && "${GR00T_PYTHON}" gr00t/data/stats.py \
    --dataset-path "${VALID_ROOT}" \
    --embodiment-tag UNITREE_G1_SONIC \
    --modality-config-path "${MODALITY_CONFIG}") || echo "[rebuild] WARN: GR00T stats failed (non-fatal)"
else
  echo "[rebuild] skip GR00T stats (python or stats.py not found)"
fi

echo "[rebuild] step 3/3: sonic unified (43s/100a, ego+left_wrist) -> ${SONIC_ROOT}"
"${PYTHON}" "${PHI0_ROOT}/scripts/data/isaac_groot_to_sonic_unified_lerobot.py" \
  --data-root "${VALID_ROOT}" \
  --out-dir "${SONIC_ROOT}" \
  --num-workers "${NUM_WORKERS}"

# Psi0 loader expects stats_psi0.json (same layout as stats_sonic_unified.json)
# openpi norm_stats for pi05 pick_tissue_sonic_unified
"${PYTHON}" - <<'PY' "${SONIC_ROOT}"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
stats = json.loads((root / "meta" / "stats_sonic_unified.json").read_text())
norm = {"norm_stats": {"state": stats["states"], "actions": stats["action"]}}
(root / "norm_stats.json").write_text(json.dumps(norm, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {root / 'norm_stats.json'}")
PY

cp "${SONIC_ROOT}/meta/stats_sonic_unified.json" "${SONIC_ROOT}/meta/stats_psi0.json"
echo "[rebuild] wrote ${SONIC_ROOT}/meta/stats_psi0.json for Psi0 ik100 training"
if [[ -f "${SUMMARY}" ]]; then
  echo "[rebuild] summary:"
  "${PYTHON}" - <<'PY' "${SUMMARY}" "${SONIC_ROOT}/meta/stats_sonic_unified.json"
import json, sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text())
sonic = json.loads(Path(sys.argv[2]).read_text())
print(f"  valid episodes: {summary['episodes']}, frames: {summary['frames']}")
print(f"  skipped: {len(summary.get('skipped', []))}")
print(f"  sonic state dim: {len(sonic['states']['mean'])}, action dim: {len(sonic['action']['mean'])}")
if summary.get("skipped"):
    for row in summary["skipped"][:10]:
        print(f"    skipped {row}")
    if len(summary["skipped"]) > 10:
        print(f"    ... and {len(summary['skipped']) - 10} more")
PY
fi

echo "[rebuild] done."

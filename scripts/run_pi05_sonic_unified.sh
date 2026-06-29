#!/usr/bin/env bash
# OpenPI pi0.5 training on SONIC unified (43-d state, 100-d action).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
python -m openpi.train_pytorch --config pick_tissue_sonic_unified "$@"

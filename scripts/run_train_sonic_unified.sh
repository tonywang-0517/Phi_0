#!/usr/bin/env bash
# Phi0 ACT training on SONIC unified (43-d state, 100-d action).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
python -m phi0.train --config-name train_sonic_unified_act "$@"

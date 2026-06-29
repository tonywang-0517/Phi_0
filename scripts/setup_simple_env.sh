#!/usr/bin/env bash
# Install Python deps for Phi_0 SIMPLE training + HTTP eval (no Isaac Sim).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

echo "==> Installing Phi_0 [simple,serve] extras"
pip install -e ".[simple,serve]"

if [[ -d "${ROOT}/third_party/SIMPLE/src/simple" ]]; then
  echo "==> Adding SIMPLE to PYTHONPATH via editable install (no-deps)"
  pip install -e "${ROOT}/third_party/SIMPLE" --no-deps || {
    echo "WARN: full SIMPLE install failed (needs Python 3.10 + Isaac Sim for sim)."
    echo "      For closed-loop eval, use SIMPLE Docker or a py3.10 env."
  }
fi

export PYTHONPATH="${ROOT}/src:${ROOT}/third_party/SIMPLE/src:${PYTHONPATH:-}"
echo ""
echo "Add to your shell or .env:"
echo "  export PYTHONPATH=${ROOT}/src:${ROOT}/third_party/SIMPLE/src:\$PYTHONPATH"
echo "  export SIMPLE_DATA_ROOT=${ROOT}/data/simple"

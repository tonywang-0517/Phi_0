#!/usr/bin/env bash
# Agent (Qwen3-VL) -> skill -> Phi0 SONIC latent ZMQ sim + mp4. stay skips deploy.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/setup_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
cd "${PHI0_ROOT}"
PYTHONPATH=src "${PHI0_PY}" scripts/phi0_agent_zmq_sim_demo.py "$@"

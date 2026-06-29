#!/usr/bin/env bash
# Agent (Qwen3-VL) -> skill -> Phi0 SONIC latent ZMQ sim + mp4. stay skips deploy.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHI0_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PYTHONUNBUFFERED=1
cd "${PHI0_ROOT}"
PYTHONPATH=src python scripts/phi0_agent_zmq_sim_demo.py "$@"

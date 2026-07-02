#!/usr/bin/env bash
# Source before training / smoke tests: sets local workspace paths.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi
export PHI0_ROOT="${PHI0_ROOT:-$ROOT}"
export PHI0_WORKSPACE="${PHI0_WORKSPACE:-$(cd "${PHI0_ROOT}/.." && pwd)}"
export PHI0_PY="${PHI0_PY:-/mnt/data/miniconda3/envs/Phi-0-wpy/bin/python}"
export GR00T_ROOT="${GR00T_ROOT:-${PHI0_ROOT}/../GR00T-WholeBodyControl}"
export PYTHONPATH="${PHI0_ROOT}/src:${PHI0_WORKSPACE}/FastWAM/src:${PYTHONPATH:-}"
export TensorRT_ROOT="${TensorRT_ROOT:-/mnt/data2/TensorRT-10.13.3.9}"
export onnxruntime_ROOT="${onnxruntime_ROOT:-${PHI0_WORKSPACE}/deps/onnxruntime-linux-x64-1.16.3}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
# g1_deploy_onnx_ref links unitree_sdk2 DDS + TensorRT at runtime
_UNITREE_SDK2_LIB="${GR00T_ROOT}/gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty/lib/x86_64"
_VENV_SIM_CYCLONEDDS="${GR00T_ROOT}/.venv_sim/lib/python3.10/site-packages/cyclonedds/.libs"
export LD_LIBRARY_PATH="${TensorRT_ROOT}/lib:${onnxruntime_ROOT}/lib:${_UNITREE_SDK2_LIB}:${_VENV_SIM_CYCLONEDDS}:${LD_LIBRARY_PATH:-}"
# unified ep447 -> valid ep544; avoids syncing 1.1G pick_tissue_valid/
export VALID_EP="${VALID_EP:-544}"
# MuJoCo sim DDS bridge (editable install into .venv_sim)
export UNITREE_SDK2_PYTHON="${UNITREE_SDK2_PYTHON:-${HOME}/YZY/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python}"

#!/usr/bin/env bash
# Repoint rsync'd .venv_sim: cluster uv/python + VIRTUAL_ENV paths break local sim.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/setup_env.sh"

GR00T_ROOT="$(cd "${GR00T_ROOT:-${PHI0_ROOT}/../GR00T-WholeBodyControl}" && pwd)"
VENV_SIM="${VENV_SIM:-${GR00T_ROOT}/.venv_sim}"
CLUSTER_VENV="/mnt/data2/wpy/workspace/GR00T-WholeBodyControl/.venv_sim"

# Prefer uv-managed CPython (matches cluster venv layout); fall back to PHI0_PY.
UV_PY_DEFAULT="${HOME}/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10"
LOCAL_PY="${VENV_SIM_PYTHON:-}"
if [[ -z "${LOCAL_PY}" ]]; then
  if [[ -x "${UV_PY_DEFAULT}" ]]; then
    LOCAL_PY="${UV_PY_DEFAULT}"
  else
    LOCAL_PY="${PHI0_PY}"
  fi
fi

if [[ ! -d "${VENV_SIM}" ]]; then
  echo "ERROR: .venv_sim not found: ${VENV_SIM}" >&2
  exit 1
fi
if [[ ! -x "${LOCAL_PY}" ]]; then
  echo "ERROR: local Python not executable: ${LOCAL_PY}" >&2
  exit 1
fi

LOCAL_PY="$(readlink -f "${LOCAL_PY}")"
LOCAL_BIN="$(dirname "${LOCAL_PY}")"

echo "[fix_venv_sim] venv=${VENV_SIM}"
echo "[fix_venv_sim] python=${LOCAL_PY}"

ln -sf "${LOCAL_PY}" "${VENV_SIM}/bin/python"
ln -sf python "${VENV_SIM}/bin/python3"
ln -sf python "${VENV_SIM}/bin/python3.10"

if [[ -f "${VENV_SIM}/pyvenv.cfg" ]]; then
  sed -i "s|^home = .*|home = ${LOCAL_BIN}|" "${VENV_SIM}/pyvenv.cfg"
fi

for act in activate activate.csh activate.fish; do
  if [[ -f "${VENV_SIM}/bin/${act}" ]]; then
    sed -i "s|${CLUSTER_VENV}|${VENV_SIM}|g" "${VENV_SIM}/bin/${act}"
  fi
done

while IFS= read -r -d '' f; do
  sed -i "1s|^#!.*|#!${VENV_SIM}/bin/python3|" "${f}"
done < <(find "${VENV_SIM}/bin" -maxdepth 1 -type f -perm -u+x -print0 2>/dev/null || true)

_ensure_unitree_sdk2() {
  local src="${UNITREE_SDK2_PYTHON:-}"
  if [[ -z "${src}" || ! -d "${src}" ]]; then
    echo "WARN: unitree_sdk2_python not found at UNITREE_SDK2_PYTHON=${src:-<unset>}" >&2
    return 0
  fi
  src="$(cd "${src}" && pwd)"
  local link="${GR00T_ROOT}/external_dependencies/unitree_sdk2_python"
  mkdir -p "$(dirname "${link}")"
  if [[ ! -e "${link}" ]]; then
    ln -sfn "${src}" "${link}"
    echo "[fix_venv_sim] linked ${link} -> ${src}"
  fi
  if ! python -c "import unitree_sdk2py" 2>/dev/null; then
    echo "[fix_venv_sim] installing unitree_sdk2py editable from ${src}"
    if command -v uv >/dev/null 2>&1; then
      uv pip install -e "${src}" --python "${VENV_SIM}/bin/python" \
        -i https://pypi.tuna.tsinghua.edu.cn/simple
    elif python -m pip --version >/dev/null 2>&1; then
      python -m pip install -q -e "${src}" -i https://pypi.tuna.tsinghua.edu.cn/simple
    else
      python -m ensurepip --default-pip >/dev/null 2>&1 || true
      python -m pip install -q -e "${src}" -i https://pypi.tuna.tsinghua.edu.cn/simple
    fi
  fi
  python -c "import unitree_sdk2py; print('ok unitree_sdk2py', unitree_sdk2py.__file__)"
}

(
  cd "${GR00T_ROOT}"
  # shellcheck source=/dev/null
  source "${VENV_SIM}/bin/activate"
  if ! python -c "import tyro" 2>/dev/null; then
    echo "[fix_venv_sim] installing tyro into .venv_sim..."
    python -m pip install -q tyro -i https://pypi.tuna.tsinghua.edu.cn/simple
  fi
  _ensure_unitree_sdk2
  python -c "import tyro, mujoco, sys; print('ok tyro', tyro.__version__, 'mujoco', mujoco.__version__, 'via', sys.executable)"
  # sim scripts import gear_sonic.scripts.* + utils/zmq_*.py
  touch "${GR00T_ROOT}/gear_sonic/scripts/__init__.py" 2>/dev/null || true
  python -c "from gear_sonic.scripts.sim_mp4_recorder import SimMp4Recorder; print('ok gear_sonic.scripts.sim_mp4_recorder')" \
    || echo "WARN: gear_sonic sim imports failed — sync gear_sonic/utils/zmq_*.py" >&2
)

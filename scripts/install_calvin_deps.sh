#!/usr/bin/env bash
# Install CALVIN (calvin_env + calvin_models) without direct GitHub access.
# Usage:
#   conda activate Phi-0-wpy
#   export CALVIN_ROOT=/mnt/data2/wpy/workspace/Phi_0/third_party/calvin
#   bash scripts/install_calvin_deps.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALVIN_ROOT="${CALVIN_ROOT:-${ROOT_DIR}/third_party/calvin}"
VENDOR_DIR="${ROOT_DIR}/third_party/vendor"
PYOPENGL_COMMIT="76d1261adee2d3fd99b418e75b0416bb7d2865e6"
PYOPENGL_ZIP="${VENDOR_DIR}/pyopengl-${PYOPENGL_COMMIT}.zip"
PYOPENGL_DIR="${VENDOR_DIR}/pyopengl-${PYOPENGL_COMMIT}"

if [[ ! -d "${CALVIN_ROOT}/calvin_env" ]]; then
  echo "CALVIN_ROOT not found: ${CALVIN_ROOT}"
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
echo "Using python: $($PYTHON_BIN --version 2>&1) @ $(command -v "$PYTHON_BIN")"

mkdir -p "${VENDOR_DIR}"

# 1) cmake 3.18.x (MulticoreTSNE breaks with system cmake 3.22+)
$PYTHON_BIN -m pip install -U wheel "cmake>=3.18.4.post1,<3.19"
# Ensure pip's cmake shim is on PATH (not /usr/bin/cmake).
export PATH="$($PYTHON_BIN -c 'import sys; print(sys.prefix)')/bin:${PATH}"

# pyhash needs legacy setuptools; build without isolation.
$PYTHON_BIN -m pip install "setuptools==57.5.0"
$PYTHON_BIN -m pip install pyhash --no-build-isolation

# 2) pyopengl fork (tacto needs OSMesa symbols; avoid git clone from GitHub)
if [[ ! -d "${PYOPENGL_DIR}" ]]; then
  echo "Downloading pyopengl fork via mirror..."
  for url in \
    "https://mirror.ghproxy.com/https://github.com/mmatl/pyopengl/archive/${PYOPENGL_COMMIT}.zip" \
    "https://ghproxy.com/https://github.com/mmatl/pyopengl/archive/${PYOPENGL_COMMIT}.zip" \
    "https://github.com/mmatl/pyopengl/archive/${PYOPENGL_COMMIT}.zip"; do
    if wget -q --timeout=60 -O "${PYOPENGL_ZIP}" "${url}"; then
      echo "Fetched pyopengl from ${url}"
      break
    fi
  done
  if [[ ! -s "${PYOPENGL_ZIP}" ]]; then
    echo "Failed to download pyopengl zip from mirrors."
    exit 1
  fi
  unzip -q -o "${PYOPENGL_ZIP}" -d "${VENDOR_DIR}"
  # GitHub zip extracts to pyopengl-<short_sha>
  if [[ ! -d "${PYOPENGL_DIR}" ]]; then
    PYOPENGL_DIR="$(find "${VENDOR_DIR}" -maxdepth 1 -type d -name 'pyopengl-*' | head -1)"
  fi
fi

echo "Installing pyopengl from ${PYOPENGL_DIR}"
$PYTHON_BIN -m pip install "${PYOPENGL_DIR}"

# 3) tacto deps without re-resolving pyopengl git URL
TACTO_REQ="${CALVIN_ROOT}/calvin_env/tacto/requirements/requirements.txt"
TACTO_REQ_BAK="${TACTO_REQ}.bak_phi0"
if [[ ! -f "${TACTO_REQ_BAK}" ]]; then
  cp "${TACTO_REQ}" "${TACTO_REQ_BAK}"
fi
grep -v '^pyopengl @ git+' "${TACTO_REQ_BAK}" > "${TACTO_REQ}"

cd "${CALVIN_ROOT}/calvin_env/tacto"
$PYTHON_BIN -m pip install -e .

cd "${CALVIN_ROOT}/calvin_env"
$PYTHON_BIN -m pip install -e .

# MulticoreTSNE must use pip cmake 3.18 (see PATH above), not build isolation.
$PYTHON_BIN -m pip install MulticoreTSNE --no-build-isolation

cd "${CALVIN_ROOT}/calvin_models"
# --no-deps: Calvin pins torch==1.13.1 which conflicts with Phi_0 (torch>=2.5).
$PYTHON_BIN -m pip install . --no-deps --no-build-isolation

echo "CALVIN install done."
echo "CALVIN_ROOT=${CALVIN_ROOT}"

#!/usr/bin/env bash
# Bootstrap SIMPLE submodule + Phi_0/data/simple training dataset.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

TASK="${TASK:-G1WholebodyBendPick-v0-psi0}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-0}"
DATA_ROOT="${ROOT}/data/simple"
SIMPLE_DIR="${ROOT}/third_party/SIMPLE"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
  fi
fi

# HF zip name may differ from LeRobot repo_id (psi0 repack not always on Hub).
HF_ZIP="${HF_ZIP:-}"
if [[ -z "${HF_ZIP}" ]]; then
  case "${TASK}" in
    G1WholebodyBendPick-v0-psi0)
      HF_ZIP="simple/G1WholebodyBendPickTeleop-v0.zip"
      ;;
    *)
      HF_ZIP="simple/${TASK}.zip"
      ;;
  esac
fi
ZIP_PATH="${ROOT}/data/${HF_ZIP}"
ZIP_ALT="${DATA_ROOT}/${HF_ZIP##*/}"
EXTRACT_DIR="${DATA_ROOT}/${HF_ZIP##*/}"
EXTRACT_DIR="${EXTRACT_DIR%.zip}"

mkdir -p "${DATA_ROOT}"

echo "==> Phi_0 SIMPLE setup"
echo "    task=${TASK}"
echo "    data_root=${DATA_ROOT}"

# --- SIMPLE repo from workspace zip (optional) ---
if [[ ! -d "${SIMPLE_DIR}/src/simple" ]]; then
  if [[ -f "${ROOT}/../SIMPLE.zip" ]]; then
    echo "==> Extracting SIMPLE.zip -> third_party/SIMPLE"
    mkdir -p "${ROOT}/third_party"
    unzip -q -o "${ROOT}/../SIMPLE.zip" -d "${ROOT}/third_party"
    if [[ -d "${ROOT}/third_party/SIMPLE" ]]; then
      echo "    SIMPLE extracted."
    fi
  else
    echo "WARN: ${SIMPLE_DIR} missing and no ../SIMPLE.zip found."
    echo "      Clone: git clone https://github.com/physical-superintelligence-lab/SIMPLE.git third_party/SIMPLE"
  fi
fi

# --- Link SIMPLE container data dir to Phi_0/data/simple ---
if [[ -d "${SIMPLE_DIR}" ]]; then
  if [[ -L "${SIMPLE_DIR}/data" ]]; then
    rm -f "${SIMPLE_DIR}/data"
  elif [[ -d "${SIMPLE_DIR}/data" ]] && [[ ! -L "${SIMPLE_DIR}/data" ]]; then
    if [[ -z "$(ls -A "${SIMPLE_DIR}/data" 2>/dev/null | grep -v README.md || true)" ]]; then
      rmdir "${SIMPLE_DIR}/data" 2>/dev/null || rm -rf "${SIMPLE_DIR}/data"
    else
      echo "WARN: ${SIMPLE_DIR}/data is non-empty; not replacing with symlink."
    fi
  fi
  if [[ ! -e "${SIMPLE_DIR}/data" ]]; then
    ln -sfn "${DATA_ROOT}" "${SIMPLE_DIR}/data"
    echo "==> Linked third_party/SIMPLE/data -> data/simple"
  fi

  if [[ ! -f "${SIMPLE_DIR}/.env" ]] && [[ -f "${SIMPLE_DIR}/.env.sample" ]]; then
    cp "${SIMPLE_DIR}/.env.sample" "${SIMPLE_DIR}/.env"
    sed -i "s|^DATA_DIR=.*|DATA_DIR=${DATA_ROOT}|" "${SIMPLE_DIR}/.env"
    echo "==> Wrote ${SIMPLE_DIR}/.env (DATA_DIR=${DATA_ROOT})"
  fi
fi

# --- Training dataset ---
if [[ -d "${DATA_ROOT}/${TASK}/meta" ]] || [[ -L "${DATA_ROOT}/${TASK}" && -d "${DATA_ROOT}/${TASK}/meta" ]]; then
  echo "==> Dataset already present: ${DATA_ROOT}/${TASK}"
  exit 0
fi

if [[ "${DOWNLOAD_DATA}" != "1" ]]; then
  echo "==> Skipping HF download (set DOWNLOAD_DATA=1 to fetch ${TASK})."
  echo "    Place dataset under: ${DATA_ROOT}/${TASK}/"
  exit 0
fi

_resolve_zip() {
  if [[ -f "${ZIP_PATH}" ]]; then
    echo "${ZIP_PATH}"
  elif [[ -f "${ZIP_ALT}" ]]; then
    echo "${ZIP_ALT}"
  else
    echo ""
  fi
}

ZIP_FILE="$(_resolve_zip)"
if [[ -n "${ZIP_FILE}" ]]; then
  echo "==> Unzipping existing ${ZIP_FILE}"
  unzip -q -o "${ZIP_FILE}" -d "${DATA_ROOT}"
  if [[ "${TASK}" != "${EXTRACT_DIR}" && ! -e "${DATA_ROOT}/${TASK}" ]]; then
    ln -sfn "${EXTRACT_DIR}" "${DATA_ROOT}/${TASK}"
    echo "==> Linked ${DATA_ROOT}/${TASK} -> ${EXTRACT_DIR}"
  fi
  echo "    Done."
  exit 0
fi

if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "ERROR: huggingface-cli not found. Install huggingface_hub, then re-run."
  exit 1
fi

HF_CLI="$(command -v hf || command -v huggingface-cli)"
echo "==> Downloading USC-PSI-Lab/psi-data:${HF_ZIP}"
echo "    HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}"
mkdir -p "${ROOT}/data"
"${HF_CLI}" download USC-PSI-Lab/psi-data "${HF_ZIP}" \
  --local-dir "${ROOT}/data" \
  --repo-type=dataset

ZIP_FILE="$(_resolve_zip)"
if [[ -n "${ZIP_FILE}" ]]; then
  unzip -q -o "${ZIP_FILE}" -d "${DATA_ROOT}"
  if [[ "${TASK}" != "${EXTRACT_DIR}" && ! -e "${DATA_ROOT}/${TASK}" ]]; then
    ln -sfn "${EXTRACT_DIR}" "${DATA_ROOT}/${TASK}"
    echo "==> Linked ${DATA_ROOT}/${TASK} -> ${EXTRACT_DIR}"
  fi
  echo "==> Dataset ready at ${DATA_ROOT}/${TASK}"
else
  echo "ERROR: Download finished but zip not found at ${ZIP_PATH} or ${ZIP_ALT}."
  exit 1
fi

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALVIN_HOME="${ROOT_DIR}/data/calvin"
SPLIT="debug"
REPORT_PATH=""

usage() {
  cat <<'EOF'
Usage:
  scripts/download_calvin_assets.sh [--calvin-home PATH] [--split debug|D|ABC|ABCD] [--report PATH]

Source priority (hard-coded):
  1) ModelScope
  2) HF-mirror
  3) Official HuggingFace
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --calvin-home)
      CALVIN_HOME="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --report)
      REPORT_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${REPORT_PATH}" ]]; then
  REPORT_PATH="${CALVIN_HOME}/download_report.json"
fi

mkdir -p "${CALVIN_HOME}" "${CALVIN_HOME}/dataset" "${CALVIN_HOME}/benchmark" "${CALVIN_HOME}/tmp"

MODEL_SCOPE_DATASET_API="https://www.modelscope.cn/api/v1/datasets"
HF_MIRROR_ENDPOINT="https://hf-mirror.com"
HF_OFFICIAL_ENDPOINT="https://huggingface.co"

declare -a REPORT_ITEMS=()
append_report_item() {
  local resource="$1"
  local source="$2"
  local status="$3"
  local url="$4"
  local local_path="$5"
  local message="$6"
  REPORT_ITEMS+=("{\"resource\":\"${resource}\",\"source\":\"${source}\",\"status\":\"${status}\",\"url\":\"${url}\",\"local_path\":\"${local_path}\",\"message\":\"${message}\"}")
}

try_download_with_priority() {
  local resource="$1"
  local repo_id="$2"
  local file_path="$3"
  local out_path="$4"
  local tmp_err="${CALVIN_HOME}/tmp/${resource//\//_}.err"

  local ms_url="${MODEL_SCOPE_DATASET_API}/${repo_id}/repo?Revision=master&FilePath=${file_path}"
  local hfm_url="${HF_MIRROR_ENDPOINT}/datasets/${repo_id}/resolve/main/${file_path}"
  local hf_url="${HF_OFFICIAL_ENDPOINT}/datasets/${repo_id}/resolve/main/${file_path}"

  local chosen_source=""
  local chosen_url=""

  for pair in \
    "ModelScope|${ms_url}" \
    "HF-mirror|${hfm_url}" \
    "HuggingFace|${hf_url}"; do
    local source="${pair%%|*}"
    local url="${pair#*|}"
    rm -f "${tmp_err}"
    if wget -q --timeout=30 --tries=1 -O "${out_path}" "${url}" 2>"${tmp_err}"; then
      chosen_source="${source}"
      chosen_url="${url}"
      append_report_item "${resource}" "${source}" "success" "${url}" "${out_path}" "downloaded"
      break
    else
      local msg
      msg="$(tr '\n' ' ' < "${tmp_err}" | sed 's/"/'\''/g' | cut -c1-240)"
      append_report_item "${resource}" "${source}" "failed" "${url}" "${out_path}" "${msg}"
    fi
  done

  if [[ -z "${chosen_source}" ]]; then
    return 1
  fi
  return 0
}

SPLIT_REPO="fywang/calvin-task-ABC-D-lerobot"
if [[ "${SPLIT}" == "ABCD" ]]; then
  SPLIT_REPO="fywang/calvin-task-ABCD-D-lerobot"
fi

BENCHMARK_README="${CALVIN_HOME}/benchmark/calvin_dataset_README.md"
DATASET_README="${CALVIN_HOME}/dataset/calvin_dataset_README.md"

try_download_with_priority \
  "calvin_benchmark_readme" \
  "${SPLIT_REPO}" \
  "README.md" \
  "${BENCHMARK_README}" || true

try_download_with_priority \
  "calvin_dataset_readme" \
  "${SPLIT_REPO}" \
  "README.md" \
  "${DATASET_README}" || true

if [[ ! -d "${CALVIN_HOME}/calvin_models" && -d "/mnt/data2/wpy/workspace/VLA-Adapter/calvin/calvin_models" ]]; then
  cp -r "/mnt/data2/wpy/workspace/VLA-Adapter/calvin/calvin_models" "${CALVIN_HOME}/calvin_models"
  append_report_item "calvin_models_local_fallback" "local" "success" "local-copy" "${CALVIN_HOME}/calvin_models" "copied from local VLA-Adapter"
fi

{
  echo "{"
  echo "  \"split\": \"${SPLIT}\","
  echo "  \"calvin_home\": \"${CALVIN_HOME}\","
  echo "  \"source_priority\": [\"ModelScope\", \"HF-mirror\", \"HuggingFace\"],"
  echo "  \"items\": ["
  for i in "${!REPORT_ITEMS[@]}"; do
    if [[ "$i" -gt 0 ]]; then
      echo "    ,${REPORT_ITEMS[$i]}"
    else
      echo "    ${REPORT_ITEMS[$i]}"
    fi
  done
  echo "  ]"
  echo "}"
} > "${REPORT_PATH}"

echo "DONE"
echo "CALVIN_HOME=${CALVIN_HOME}"
echo "REPORT=${REPORT_PATH}"
du -sh "${CALVIN_HOME}" || true

#!/usr/bin/env bash
# Rsync Phi_0 assets from cluster_0 — minimal paths, resume (--partial), per-item logs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/scripts/setup_env.sh"

REMOTE="${CLUSTER0:-cluster_0}"
REMOTE_WS="/mnt/data2/wpy/workspace"
LOCAL_WS="${PHI0_WORKSPACE}"
LOCK_DIR="${ROOT}/.sync_locks"
LOG_DIR="${ROOT}/logs/sync"
RSYNC_OPTS=(-avh --partial --append-verify --info=progress2)
SSH_OPTS=(-o ServerAliveInterval=30 -o ServerAliveCountMax=10)
mkdir -p "${LOCK_DIR}" "${LOG_DIR}"

usage() {
  cat <<'EOF'
Usage: bash scripts/sync_from_cluster0.sh <profile|status>

Profiles (only what each workflow needs):
  smoke       demo HDF5 + egodex + smplh constants          ~1.9 GB
  psi0        Qwen3-VL Psi0 weights                         ~4.0 GB
  vggt        VGGT-Omega checkpoint (dual-tower only)        ~4.3 GB
  agent_sim   ZMQ sim demo (ep447) — see list below         ~9.5 GB

  agent_sim pulls:
    gear_sonic_deploy/          1.8 GB  TensorRT deploy + ONNX
    .venv_sim/                  5.4 GB  MuJoCo sim venv
    gear_sonic/scripts/           356 KB  sim runner scripts
    gear_sonic/utils/mujoco_sim/  155 KB  MuJoCo sim (set_standing_pose_on_ground)
    experiments/.../scripts/    140 KB  run_sim_loop_vla_record.py only
    sample_data/.../walk_*.pkl  344 KB  default robot motion
    pick_tissue unified meta+data 597 MB  LeRobot ep447 loader
    experiments/phi0_hgpt_zmq/   168 KB  ZMQ publisher (sonic + HGPT)
    ep447 ego+wrist mp4           3 MB  sim GT overlay
    TensorRT + onnxruntime       11 GB  deploy runtime

  NOT synced: sonic_vla_overfit/models (12G), pick_tissue videos (3.4G),
              pick_tissue_valid (1.1G — use VALID_EP=544 in setup_env.sh)

Logs:  logs/sync/<profile>_<item>.log
       logs/sync/<profile>.summary.log

Resume: rsync --partial --append-verify (safe to re-run).
Env: CLUSTER0=cluster_0  PHI0_WORKSPACE=/home/user
EOF
}

_rsync_busy() {
  local dest="$1"
  pgrep -af "rsync.*${dest}" >/dev/null 2>&1
}

_remote_bytes() {
  ssh "${SSH_OPTS[@]}" "${REMOTE}" "du -sb $(printf '%q' "$1") 2>/dev/null | cut -f1" 2>/dev/null || true
}

_local_bytes() {
  du -sb "$1" 2>/dev/null | cut -f1 || echo 0
}

_complete() {
  local remote="$1" local_path="$2"
  local rb lb
  rb="$(_remote_bytes "$remote")"
  lb="$(_local_bytes "$local_path")"
  [[ -n "${rb}" && "${rb}" != "0" && "${rb}" == "${lb}" ]]
}

_sync_item() {
  local remote_path="$1"
  local local_path="$2"
  local label="$3"
  local profile="${4:-misc}"
  local log="${LOG_DIR}/${profile}_$(echo -n "${label}" | tr ' /' '__').log"

  mkdir -p "$(dirname "$local_path")"
  if _complete "${remote_path}" "${local_path}"; then
    echo "[skip] ${label} — complete ($(_local_bytes "${local_path}" | numfmt --to=iec 2>/dev/null || _local_bytes "${local_path}") B)"
    echo "[skip] ${label}" >> "${LOG_DIR}/${profile}.summary.log"
    return 0
  fi
  if _rsync_busy "${local_path}"; then
    echo "[skip] ${label} — rsync already running -> ${local_path}"
    echo "  tail -f ${log}"
    return 0
  fi

  local lock="${LOCK_DIR}/$(echo -n "${local_path}" | tr '/' '_').lock"
  exec 9>"${lock}"
  if ! flock -n 9; then
    echo "[skip] ${label} — locked (${lock})"
    return 0
  fi

  {
    echo "======== $(date -Is) START ${label} ========"
    echo "remote: ${REMOTE}:${remote_path}"
    echo "local:  ${local_path}"
    echo "remote_bytes: $(_remote_bytes "${remote_path}")"
    echo "local_bytes:  $(_local_bytes "${local_path}")"
    rsync "${RSYNC_OPTS[@]}" \
      -e "ssh ${SSH_OPTS[*]}" \
      "${REMOTE}:${remote_path}" "${local_path}"
    echo "======== $(date -Is) OK ${label} ========"
  } 2>&1 | tee -a "${log}" | tee -a "${LOG_DIR}/${profile}.summary.log"
}

_print_status() {
  local -a items=(
    "${REMOTE_WS}/GR00T-WholeBodyControl/gear_sonic_deploy|${LOCAL_WS}/GR00T-WholeBodyControl/gear_sonic_deploy|gear_sonic_deploy"
    "${REMOTE_WS}/GR00T-WholeBodyControl/.venv_sim|${LOCAL_WS}/GR00T-WholeBodyControl/.venv_sim|venv_sim"
    "${REMOTE_WS}/GR00T-WholeBodyControl/gear_sonic/scripts|${GR00T_ROOT:-${LOCAL_WS}/GR00T-WholeBodyControl}/gear_sonic/scripts|gear_sonic_scripts"
    "${REMOTE_WS}/GR00T-WholeBodyControl/experiments/sonic_vla_overfit/scripts|${GR00T_ROOT:-${LOCAL_WS}/GR00T-WholeBodyControl}/experiments/sonic_vla_overfit/scripts|sim_scripts"
    "${REMOTE_WS}/GR00T-WholeBodyControl/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl|${LOCAL_WS}/GR00T-WholeBodyControl/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl|robot_motion_pkl"
    "${REMOTE_WS}/Isaac-GR00T/data/pick_tissue_xperience_unified/meta|${LOCAL_WS}/Isaac-GR00T/data/pick_tissue_xperience_unified/meta|unified_meta"
    "${REMOTE_WS}/Isaac-GR00T/data/pick_tissue_xperience_unified/data|${LOCAL_WS}/Isaac-GR00T/data/pick_tissue_xperience_unified/data|unified_data"
    "${REMOTE_WS}/Phi_0/experiments/phi0_hgpt_zmq|${LOCAL_WS}/Phi_0/experiments/phi0_hgpt_zmq|phi0_hgpt_zmq"
    "${REMOTE_WS}/Isaac-GR00T/data/pick_tissue_xperience_unified/videos/chunk-000/observation.images.ego_view/episode_000447.mp4|${LOCAL_WS}/Isaac-GR00T/data/pick_tissue_xperience_unified/videos/chunk-000/observation.images.ego_view/episode_000447.mp4|ep447_ego"
    "${REMOTE_WS}/Isaac-GR00T/data/pick_tissue_xperience_unified/videos/chunk-000/observation.images.left_wrist/episode_000447.mp4|${LOCAL_WS}/Isaac-GR00T/data/pick_tissue_xperience_unified/videos/chunk-000/observation.images.left_wrist/episode_000447.mp4|ep447_wrist"
    "/mnt/data2/TensorRT-10.13.3.9|${LOCAL_WS}/deps/TensorRT-10.13.3.9|TensorRT"
    "/mnt/data2/wpy/deps/onnxruntime-linux-x64-1.16.3|${LOCAL_WS}/deps/onnxruntime-linux-x64-1.16.3|onnxruntime"
  )
  printf "%-16s %12s %12s %s\n" "ITEM" "LOCAL" "REMOTE" "STATUS"
  for entry in "${items[@]}"; do
    IFS='|' read -r r l n <<<"${entry}"
    rb="$(_remote_bytes "$r")"; lb="$(_local_bytes "$l")"
    local_h=$(numfmt --to=iec "${lb:-0}" 2>/dev/null || echo "${lb:-0}")
    remote_h=$(numfmt --to=iec "${rb:-0}" 2>/dev/null || echo "${rb:-0}")
    st="pending"
    if _complete "$r" "$l"; then st="done"; elif [[ "${lb:-0}" != "0" ]]; then st="partial"; fi
    printf "%-16s %12s %12s %s\n" "${n}" "${local_h}" "${remote_h}" "${st}"
  done
  echo ""
  echo "Active rsync:"
  pgrep -af 'rsync.*cluster_0' 2>/dev/null || echo "  (none)"
  echo ""
  echo "Tail a log: tail -f ${LOG_DIR}/agent_sim_<item>.log"
}

_sync_smoke() {
  local p=smoke demo="${LOCAL_WS}/Isaac-GR00T/demo_data"
  : > "${LOG_DIR}/${p}.summary.log"
  _sync_item "${REMOTE_WS}/Isaac-GR00T/demo_data/xperience-10m-sample/annotation.hdf5" \
    "${demo}/xperience-10m-sample/annotation.hdf5" "xperience_hdf5" "${p}"
  _sync_item "${REMOTE_WS}/Isaac-GR00T/demo_data/xperience-10m-sample/stereo_left.mp4" \
    "${demo}/xperience-10m-sample/stereo_left.mp4" "xperience_mp4" "${p}"
  _sync_item "${REMOTE_WS}/Isaac-GR00T/demo_data/egodex/test/add_remove_lid/" \
    "${demo}/egodex/test/add_remove_lid/" "egodex_ep0" "${p}"
  _sync_item "${REMOTE_WS}/Phi_0/data/body_models/smplh_skeleton_constants.npz" \
    "${LOCAL_WS}/Phi_0/data/body_models/smplh_skeleton_constants.npz" "smplh_constants" "${p}"
}

_sync_psi0() {
  local p=psi0 dest="${LOCAL_WS}/Phi_0/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k"
  : > "${LOG_DIR}/${p}.summary.log"
  _sync_item "${REMOTE_WS}/Phi_0/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k/" \
    "${dest}/" "psi0_vlm" "${p}"
}

_sync_vggt() {
  local p=vggt
  : > "${LOG_DIR}/${p}.summary.log"
  _sync_item "${REMOTE_WS}/vggt-omega/checkpoints/vggt_omega_1b_512.pt" \
    "${LOCAL_WS}/vggt-omega/checkpoints/vggt_omega_1b_512.pt" "vggt_ckpt" "${p}"
}

_sync_agent_sim() {
  local p=agent_sim groot="${GR00T_ROOT:-${LOCAL_WS}/GR00T-WholeBodyControl}" data="${LOCAL_WS}/Isaac-GR00T/data"
  : > "${LOG_DIR}/${p}.summary.log"
  _sync_item "${REMOTE_WS}/GR00T-WholeBodyControl/gear_sonic_deploy/" \
    "${groot}/gear_sonic_deploy/" "gear_sonic_deploy" "${p}"
  _sync_item "${REMOTE_WS}/GR00T-WholeBodyControl/.venv_sim/" \
    "${groot}/.venv_sim/" "venv_sim" "${p}"
  _sync_item "${REMOTE_WS}/GR00T-WholeBodyControl/gear_sonic/scripts/" \
    "${groot}/gear_sonic/scripts/" "gear_sonic_scripts" "${p}"
  _sync_item "${REMOTE_WS}/GR00T-WholeBodyControl/gear_sonic/utils/mujoco_sim/" \
    "${groot}/gear_sonic/utils/mujoco_sim/" "gear_sonic_mujoco_sim" "${p}"
  for _zmq_util in zmq_pose_frames.py zmq_pose_unpack.py zmq_sim_diagnostics.py; do
    _sync_item "${REMOTE_WS}/GR00T-WholeBodyControl/gear_sonic/utils/${_zmq_util}" \
      "${groot}/gear_sonic/utils/${_zmq_util}" "gear_sonic_${_zmq_util%.py}" "${p}"
  done
  _sync_item "${REMOTE_WS}/GR00T-WholeBodyControl/experiments/sonic_vla_overfit/scripts/" \
    "${groot}/experiments/sonic_vla_overfit/scripts/" "sim_scripts" "${p}"
  _sync_item "${REMOTE_WS}/GR00T-WholeBodyControl/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl" \
    "${groot}/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl" "robot_motion_pkl" "${p}"
  _sync_item "${REMOTE_WS}/Isaac-GR00T/data/pick_tissue_xperience_unified/meta/" \
    "${data}/pick_tissue_xperience_unified/meta/" "unified_meta" "${p}"
  _sync_item "${REMOTE_WS}/Isaac-GR00T/data/pick_tissue_xperience_unified/data/" \
    "${data}/pick_tissue_xperience_unified/data/" "unified_data" "${p}"
  _sync_item "${REMOTE_WS}/Phi_0/experiments/phi0_hgpt_zmq/" \
    "${PHI0_ROOT}/experiments/phi0_hgpt_zmq/" "phi0_hgpt_zmq" "${p}"
  _sync_item "${REMOTE_WS}/Isaac-GR00T/data/pick_tissue_xperience_unified/videos/chunk-000/observation.images.ego_view/episode_000447.mp4" \
    "${data}/pick_tissue_xperience_unified/videos/chunk-000/observation.images.ego_view/episode_000447.mp4" "ep447_ego" "${p}"
  _sync_item "${REMOTE_WS}/Isaac-GR00T/data/pick_tissue_xperience_unified/videos/chunk-000/observation.images.left_wrist/episode_000447.mp4" \
    "${data}/pick_tissue_xperience_unified/videos/chunk-000/observation.images.left_wrist/episode_000447.mp4" "ep447_wrist" "${p}"
  _sync_item "/mnt/data2/TensorRT-10.13.3.9/" \
    "${LOCAL_WS}/deps/TensorRT-10.13.3.9/" "TensorRT" "${p}"
  _sync_item "/mnt/data2/wpy/deps/onnxruntime-linux-x64-1.16.3/" \
    "${LOCAL_WS}/deps/onnxruntime-linux-x64-1.16.3/" "onnxruntime" "${p}"
  echo "[fix] .venv_sim python symlinks (cluster uv -> local)"
  bash "${ROOT}/scripts/fix_venv_sim.sh" >> "${LOG_DIR}/${p}_fix_venv_sim.log" 2>&1 || true
  echo "[stub] LeRobot video placeholders (ep447 mp4 real, others empty)"
  "${PHI0_PY}" "${ROOT}/scripts/ensure_pick_tissue_video_stubs.py" \
    --dataset-root "${data}/pick_tissue_xperience_unified" \
    >> "${LOG_DIR}/${p}_video_stubs.log" 2>&1 || true
}

PROFILE="${1:-smoke}"
case "${PROFILE}" in
  smoke) _sync_smoke ;;
  psi0) _sync_psi0 ;;
  vggt) _sync_vggt ;;
  agent_sim) _sync_agent_sim ;;
  status) _print_status ;;
  all) _sync_smoke; _sync_psi0 ;;
  -h|--help|help) usage; exit 0 ;;
  *) echo "Unknown profile: ${PROFILE}"; usage; exit 1 ;;
esac

echo "[done] ${PROFILE} — logs in ${LOG_DIR}/"

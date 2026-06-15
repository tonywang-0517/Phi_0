#!/usr/bin/env bash
# Prepare SMPL-H body models for Phi_0 visualization (data/body_models).
#
# SMPL-H requires registration & manual download (non-commercial license):
#   http://mano.is.tue.mpg.de  -> Extended SMPL+H model + MANO v1.2
#
# After downloading, run merge (see smplx/tools/README.md):
#   python /mnt/data1/wpy/workspace/smplx/tools/merge_smplh_mano.py \
#     --smplh-fn /path/to/smplh/female/model.npz \
#     --mano-left-fn /path/to/mano_v1_2/models/MANO_LEFT.pkl \
#     --mano-right-fn /path/to/mano_v1_2/models/MANO_RIGHT.pkl \
#     --output-folder /tmp/smplh_merged
#   cp /tmp/smplh_merged/model.pkl data/body_models/smplh/SMPLH_FEMALE.pkl
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/data/body_models/smplh"
mkdir -p "${DEST}"

if [[ -f "${DEST}/SMPLH_FEMALE.pkl" ]] || [[ -f "${DEST}/SMPLH_MALE.pkl" ]]; then
  echo "SMPL-H model already present under ${DEST}"
  ls -lh "${DEST}"/*.pkl 2>/dev/null || true
  exit 0
fi

echo "SMPL-H models not found."
echo "Expected layout (smplx convention):"
echo "  ${ROOT}/data/body_models/"
echo "  ├── smplh/"
echo "  │   ├── SMPLH_FEMALE.pkl"
echo "  │   └── SMPLH_MALE.pkl   (optional)"
echo ""
echo "Download from http://mano.is.tue.mpg.de (license required), then merge with MANO"
echo "using smplx/tools/merge_smplh_mano.py — see scripts/setup_smplh_body_models.sh header."
exit 1

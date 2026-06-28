#!/usr/bin/env bash
# Parallel offline predecode for pick_tissue_xperience_unified (607 eps x 2 cams).
# ponytail: cv2 sequential decode >> torchcodec timestamp batch for full episodes.
set -euo pipefail

REPO="/mnt/data2/wpy/workspace/Phi_0"
PY="/mnt/data/miniconda3/envs/Phi-0-wpy/bin/python"
DATASET="/mnt/data2/wpy/workspace/Isaac-GR00T/data/pick_tissue_xperience_unified"
LOG_DIR="/mnt/data2/wpy/workspace/logs/predecode_pick_tissue"
BACKEND="${BACKEND:-cv2}"
NWORKERS="${NWORKERS:-16}"
OVERWRITE="${OVERWRITE:-1}"
TOTAL_EPISODES="${TOTAL_EPISODES:-607}"

mkdir -p "$LOG_DIR"
cd "$REPO"

per=$(( (TOTAL_EPISODES + NWORKERS - 1) / NWORKERS ))

echo "==> predecode backend=$BACKEND workers=$NWORKERS (~$per eps/worker)"

pids=()
for (( i=0; i<NWORKERS; i++ )); do
  start=$(( i * per ))
  if (( start >= TOTAL_EPISODES )); then
    break
  fi
  max=$per
  if (( start + max > TOTAL_EPISODES )); then
    max=$(( TOTAL_EPISODES - start ))
  fi
  log="$LOG_DIR/shard_w${i}_ep${start}.log"
  echo "    worker $i episodes [$start, $((start + max))) -> $log"
  PYTHONPATH=src "$PY" scripts/data/predecode_lerobot_videos.py \
    --dataset-root "$DATASET" \
    --backend "$BACKEND" \
    --start-episode "$start" \
    --max-episodes "$max" \
    --skip-store-meta \
    $( [[ "$OVERWRITE" == "1" ]] && echo --overwrite ) \
    >"$log" 2>&1 &
  pids+=("$!")
done

echo "==> waiting for ${#pids[@]} workers..."
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done
if (( fail )); then
  echo "ERROR: one or more workers failed; see $LOG_DIR"
  exit 1
fi

echo "==> writing meta.json + full validation"
PYTHONPATH=src "$PY" - <<PY
import json
from pathlib import Path

from phi0.data.predecoded_video import (
    PREDECODED_VERSION,
    PredecodedVideoMeta,
    predecoded_root,
    read_episodes_jsonl,
    validate_predecoded_store,
    write_store_meta,
)

dataset = Path("$DATASET")
from phi0.data.psi0_image import read_lerobot_video_hw
image_size = read_lerobot_video_hw(dataset, "observation.images.ego_view")
store = predecoded_root(dataset, image_size)
info = json.loads((dataset / "meta" / "info.json").read_text())
write_store_meta(
    store,
    PredecodedVideoMeta(
        version=PREDECODED_VERSION,
        image_size=image_size,
        layout="THWC",
        dtype="uint8",
        fps=float(info["fps"]),
        video_keys=(
            "observation.images.ego_view",
            "observation.images.left_wrist",
        ),
        total_episodes=len(read_episodes_jsonl(dataset / "meta")),
        backend="$BACKEND",
    ),
)
errs = validate_predecoded_store(dataset, store)
if errs:
    raise SystemExit(f"validation failed ({len(errs)}): {errs[0]}")
print(f"validation OK: {store}")
PY

echo "==> done. logs: $LOG_DIR"

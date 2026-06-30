#!/usr/bin/env bash
# Shard offline MP4→npy predecode; skips existing episode files unless OVERWRITE=1.
set -euo pipefail

REPO="${REPO:-/mnt/data2/wpy/workspace/Phi_0}"
PY="${PY:-/mnt/data/miniconda3/envs/Phi-0-wpy/bin/python}"
DATASET="${DATASET:?set DATASET to LeRobot root}"
LOG_DIR="${LOG_DIR:-/mnt/data2/wpy/workspace/logs/predecode_lerobot}"
BACKEND="${BACKEND:-cv2}"
NWORKERS="${NWORKERS:-16}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "$LOG_DIR"
cd "$REPO"

TOTAL_EPISODES="$("$PY" - <<PY
import json
from pathlib import Path
p = Path("$DATASET") / "meta" / "episodes.jsonl"
print(sum(1 for line in p.read_text().splitlines() if line.strip()))
PY
)"
per=$(( (TOTAL_EPISODES + NWORKERS - 1) / NWORKERS ))
ow_flag=()
[[ "$OVERWRITE" == "1" ]] && ow_flag=(--overwrite)

echo "==> predecode dataset=$DATASET backend=$BACKEND workers=$NWORKERS episodes=$TOTAL_EPISODES overwrite=$OVERWRITE"

pids=()
for (( i=0; i<NWORKERS; i++ )); do
  start=$(( i * per ))
  (( start >= TOTAL_EPISODES )) && break
  max=$per
  (( start + max > TOTAL_EPISODES )) && max=$(( TOTAL_EPISODES - start ))
  log="$LOG_DIR/shard_w${i}_ep${start}.log"
  echo "    worker $i episodes [$start, $((start + max))) -> $log"
  PYTHONPATH=src "$PY" scripts/data/predecode_lerobot_videos.py \
    --dataset-root "$DATASET" \
    --backend "$BACKEND" \
    --start-episode "$start" \
    --max-episodes "$max" \
    --skip-store-meta \
    "${ow_flag[@]}" \
    >"$log" 2>&1 &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
(( fail )) && { echo "ERROR: predecode workers failed; see $LOG_DIR"; exit 1; }

echo "==> writing meta.json + validation"
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
from phi0.data.psi0_image import read_lerobot_video_hw

dataset = Path("$DATASET")
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

echo "==> predecode done: $DATASET"

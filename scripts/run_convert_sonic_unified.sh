#!/usr/bin/env bash
# Convert Isaac-GR00T SONIC valid data to unified 43s/100a LeRobot.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/data/isaac_groot_to_sonic_unified_lerobot.py "$@"

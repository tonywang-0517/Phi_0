#!/usr/bin/env python3
"""Download Cosmos-Predict2.5-2B weights and verify local layout (no GPU load)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phi0.models.cosmos.loader import (  # noqa: E402
    DEFAULT_BASE_MODEL_NAME,
    DEFAULT_REVISION,
    verify_cosmos_weight_files,
)

REPO = "nvidia/Cosmos-Predict2.5-2B"
DEFAULT_DEST = ROOT / "checkpoints" / "nvidia" / DEFAULT_BASE_MODEL_NAME


def _load_env() -> None:
    env_file = ROOT / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


def download(dest: Path, revision: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        "hf",
        "download",
        REPO,
        "--revision",
        revision,
        "--local-dir",
        str(dest),
    ]
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"==> {REPO} ({revision})")
    print(f"    Destination: {dest}")
    env = {**os.environ, "HF_ENDPOINT": endpoint}
    try:
        subprocess.run(cmd, check=True, env=env, capture_output=True)
    except subprocess.CalledProcessError:
        print(f"Mirror failed for {REPO}; retrying huggingface.co ...")
        env["HF_ENDPOINT"] = "https://huggingface.co"
        subprocess.run(cmd, check=True, env=env)


def verify(dest: Path) -> int:
    if not (dest / "model_index.json").is_file():
        print(f"  [MISS] model_index.json under {dest}")
        return 1
    incomplete = verify_cosmos_weight_files(dest)
    if incomplete:
        for item in incomplete:
            print(f"  [MISS] {item}")
        return 1
    print(f"  [ok]   Cosmos weights verified under {dest}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and verify Cosmos-Predict2.5-2B weights")
    p.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Local directory (default: {DEFAULT_DEST})",
    )
    p.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help=f"HF revision (default: {DEFAULT_REVISION})",
    )
    p.add_argument("--skip-download", action="store_true", help="Only run verify")
    p.add_argument("--skip-verify", action="store_true", help="Only run download")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _load_env()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    if not args.skip_download:
        download(args.dest, args.revision)

    if args.skip_verify:
        print("Done. Run: python scripts/verify_weights.py")
        return 0

    return verify(args.dest)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Verify Wan2.2 weight files for legacy FastWAM stack (no GPU load)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
WAN22 = CKPT / "Wan-AI/Wan2.2-TI2V-5B"
WAN21 = CKPT / "Wan-AI/Wan2.1-T2V-1.3B"

EXPECTED = [
    WAN22 / "diffusion_pytorch_model-00001-of-00003.safetensors",
    WAN22 / "diffusion_pytorch_model-00002-of-00003.safetensors",
    WAN22 / "diffusion_pytorch_model-00003-of-00003.safetensors",
    WAN22 / "Wan2.2_VAE.pth",
    WAN22 / "models_t5_umt5-xxl-enc-bf16.pth",
    WAN21 / "google/umt5-xxl/tokenizer.json",
]


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main() -> int:
    ok = True
    print(f"Wan checkpoints: {CKPT}")
    for item in EXPECTED:
        if item.is_file():
            print(f"  [ok]   {item.relative_to(ROOT)} ({_fmt_size(item.stat().st_size)})")
        else:
            print(f"  [MISS] {item.relative_to(ROOT)}")
            ok = False
    smplh = ROOT / "data/body_models/smplh"
    pkls = list(smplh.glob("SMPLH_*.pkl")) if smplh.is_dir() else []
    if pkls:
        for p in pkls:
            print(f"  [smplh] {p.relative_to(ROOT)} ({_fmt_size(p.stat().st_size)})")
    else:
        print("  [smplh] data/body_models/smplh/SMPLH_*.pkl — manual license download")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

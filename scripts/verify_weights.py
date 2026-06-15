#!/usr/bin/env python3
"""Verify local Cosmos-Predict2.5-2B weight files (DiT4DiT layout, no GPU load)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phi0.models.cosmos.loader import (  # noqa: E402
    DEFAULT_BASE_MODEL_NAME,
    resolve_cosmos_base_model,
    verify_cosmos_weight_files,
)


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main() -> int:
    ckpt = ROOT / "checkpoints"
    print(f"Checkpoints root: {ckpt}")
    try:
        base = resolve_cosmos_base_model(checkpoints_dir=ckpt)
    except FileNotFoundError as exc:
        print(f"  [MISS] Cosmos base_model not found\n{exc}")
        return 1

    print(f"  [ok]   base_model: {base}")
    incomplete = verify_cosmos_weight_files(base)
    if incomplete:
        for item in incomplete:
            print(f"  [MISS] {item}")
        return 1

    for sub in ("vae", "transformer", "text_encoder", "tokenizer"):
        subdir = base / sub
        if not subdir.is_dir():
            continue
        weights = list(subdir.glob("*.safetensors")) + list(subdir.glob("*.bin"))
        configs = list(subdir.glob("config.json"))
        total = sum(p.stat().st_size for p in weights)
        label = f"{sub}/"
        if weights:
            print(f"  [ok]   {label} ({len(weights)} weight file(s), {_fmt_size(total)})")
        elif configs:
            print(f"  [warn] {label} config only (no weights)")
        else:
            print(f"  [MISS] {label}")

    idx = base / "model_index.json"
    if idx.is_file():
        print(f"  [ok]   model_index.json ({_fmt_size(idx.stat().st_size)})")

    print(f"\nExpected layout (DiT4DiT): .../{DEFAULT_BASE_MODEL_NAME}/{{vae,transformer,text_encoder,tokenizer}}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

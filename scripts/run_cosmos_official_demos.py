#!/usr/bin/env python3
"""Run cosmos-predict2.5 official Diffusers demos with Phi0 local Cosmos weights."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
COSMOS_ROOT = ROOT.parent / "cosmos-predict2.5"
DEFAULT_MODEL = ROOT / "checkpoints/nvidia/Cosmos-Predict2.5-2B"
DEFAULT_OUT = ROOT / "experiments/cosmos_official_demos"


def _patch_safety_checker() -> None:
    import diffusers.pipelines.cosmos.pipeline_cosmos2_5_predict as pmod

    class DummySafetyChecker(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self._dev = torch.device("cpu")
            self._dtype = torch.float32

        @property
        def dtype(self):
            return self._dtype

        @property
        def device(self):
            return self._dev

        def to(self, *args, **kwargs):
            device = kwargs.get("device", args[0] if args else None)
            dtype = kwargs.get("dtype", args[1] if len(args) > 1 else None)
            if device is not None:
                self._dev = torch.device(device)
            if dtype is not None:
                self._dtype = dtype
            return self

        def check_text_safety(self, text):
            return True

        def check_video_safety(self, video):
            return video

    pmod.CosmosSafetyChecker = DummySafetyChecker


def _load_prompt_and_image(json_path: Path) -> tuple[str, object | None, object | None]:
    from diffusers.utils import load_image, load_video

    cfg = json.loads(json_path.read_text())
    base = json_path.parent
    prompt = None
    if cfg.get("prompt_path"):
        prompt = (base / cfg["prompt_path"]).read_text().strip()
    elif cfg.get("prompt"):
        prompt = str(cfg["prompt"]).strip()

    image = None
    video = None
    media = cfg.get("input_path")
    if media:
        media_path = base / media
        suffix = media_path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png"}:
            image = load_image(str(media_path))
        elif suffix == ".mp4":
            video = load_video(str(media_path))
    return prompt, image, video


def run_demo(
    *,
    name: str,
    json_path: Path,
    model_id: Path,
    revision: str,
    output_dir: Path,
    num_frames: int,
    num_steps: int,
    seed: int,
    device: str,
) -> dict:
    from diffusers import Cosmos2_5_PredictBasePipeline
    from diffusers.utils import export_to_video

    _patch_safety_checker()
    prompt, image, video = _load_prompt_and_image(json_path)
    if not prompt:
        raise ValueError(f"No prompt found in {json_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = output_dir / f"{name}_{num_frames}f_s{seed}.mp4"

    t0 = time.perf_counter()
    pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
        str(model_id),
        revision=revision,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe = pipe.to(device)
    frames = pipe(
        image=image,
        video=video,
        prompt=prompt,
        num_frames=num_frames,
        num_inference_steps=num_steps,
        generator=torch.Generator(device=device).manual_seed(seed),
    ).frames[0]
    export_to_video(frames, str(out_mp4), fps=16)
    elapsed = time.perf_counter() - t0

    return {
        "name": name,
        "json": str(json_path),
        "prompt_preview": prompt[:160] + ("..." if len(prompt) > 160 else ""),
        "num_frames": num_frames,
        "num_steps": num_steps,
        "seed": seed,
        "output_mp4": str(out_mp4),
        "elapsed_sec": round(elapsed, 1),
        "model_id": str(model_id),
        "revision": revision,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Run official Cosmos demos with local Phi0 weights")
    p.add_argument("--model-id", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--revision", type=str, default="diffusers/base/post-trained")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--num-frames", type=int, default=93)
    p.add_argument("--num-steps", type=int, default=36)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--demos",
        nargs="*",
        default=["robot_welding", "robot_pouring", "bus_terminal"],
        help="Demo names under cosmos-predict2.5/assets/base/{name}.json",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not COSMOS_ROOT.is_dir():
        print(f"Missing cosmos-predict2.5 repo at {COSMOS_ROOT}", file=sys.stderr)
        return 1
    if not args.model_id.is_dir():
        print(f"Missing local model at {args.model_id}", file=sys.stderr)
        return 1

    results = []
    for name in args.demos:
        json_path = COSMOS_ROOT / "assets/base" / f"{name}.json"
        if not json_path.is_file():
            print(f"Skip missing demo config: {json_path}")
            continue
        print(f"==> {name} ({json_path.name})")
        meta = run_demo(
            name=name,
            json_path=json_path,
            model_id=args.model_id,
            revision=args.revision,
            output_dir=args.output_dir,
            num_frames=args.num_frames,
            num_steps=args.num_steps,
            seed=args.seed,
            device=args.device,
        )
        results.append(meta)
        print(json.dumps(meta, indent=2))

    summary_path = args.output_dir / "demo_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

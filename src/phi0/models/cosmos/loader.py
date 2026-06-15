"""Load Cosmos-Predict2.5-2B from a local directory (DiT4DiT ``base_model`` pattern)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL_NAME = "Cosmos-Predict2.5-2B"
DEFAULT_REVISION = "diffusers/base/post-trained"


class _DefaultDummySafetyChecker:
    """Construct Cosmos pipeline without importing/downloading guardrail (DiT4DiT)."""

    def __init__(self, *args, **kwargs):
        pass

    def to(self, device):
        return self

    def check_text_safety(self, text):
        return True

    def check_video_safety(self, video):
        return video


@dataclass
class CosmosComponents:
    vae: torch.nn.Module | None
    transformer: torch.nn.Module | None
    text_encoder: torch.nn.Module | None
    tokenizer: object | None
    latents_mean: torch.Tensor | None
    latents_std: torch.Tensor | None
    vae_scale_factor_spatial: int
    vae_scale_factor_temporal: int
    text_embed_dim: int
    transformer_in_channels: int
    base_model_path: Path


def _is_valid_cosmos_dir(path: Path) -> bool:
    return path.is_dir() and (path / "model_index.json").is_file()


def resolve_cosmos_base_model(
    *,
    base_model: str | Path | None = None,
    checkpoints_dir: Path | None = None,
) -> Path:
    """Resolve local Cosmos-Predict2.5-2B directory (DiT4DiT ``framework.cosmos25.base_model``)."""
    if checkpoints_dir is None:
        checkpoints_dir = Path(__file__).resolve().parents[3] / "checkpoints"
    checkpoints_dir = Path(checkpoints_dir)
    workspace = checkpoints_dir.parent

    candidates: list[Path] = []
    env_path = os.environ.get("COSMOS25_BASE_MODEL")
    if env_path:
        candidates.append(Path(env_path))
    if base_model is not None and str(base_model).lower() not in {"", "null", "none"}:
        candidates.append(Path(str(base_model)))
    candidates.extend(
        [
            checkpoints_dir / DEFAULT_BASE_MODEL_NAME,
            checkpoints_dir / "nvidia" / DEFAULT_BASE_MODEL_NAME,
            workspace.parent / "checkpoints" / DEFAULT_BASE_MODEL_NAME,
        ]
    )

    seen: set[str] = set()
    for raw in candidates:
        p = raw.expanduser().resolve()
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if _is_valid_cosmos_dir(p):
            logger.info("Resolved Cosmos base_model: %s", p)
            return p

    checked = "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Cosmos-Predict2.5-2B not found locally. Checked:\n  - "
        f"{checked}\n\n"
        "Download once (same as DiT4DiT README):\n"
        "  huggingface-cli download nvidia/Cosmos-Predict2.5-2B "
        f"--revision {DEFAULT_REVISION} --local-dir /path/to/{DEFAULT_BASE_MODEL_NAME}\n"
        "Then set `model.base_model` in phi0_full.yaml or export COSMOS25_BASE_MODEL.\n"
        "See: /mnt/data1/wpy/workspace/DiT4DiT/README.md"
    )


def verify_cosmos_weight_files(base: Path) -> list[str]:
    """Return list of missing weight subfolders (empty if complete)."""
    missing: list[str] = []
    for sub in ("vae", "transformer", "text_encoder"):
        subdir = base / sub
        if not subdir.is_dir():
            missing.append(f"{sub}/ (directory missing)")
            continue
        has_weights = any(subdir.glob("*.safetensors")) or any(subdir.glob("*.bin"))
        if not has_weights:
            missing.append(f"{sub}/ (no *.safetensors or *.bin)")
    return missing


def load_cosmos_predict25_2b(
    *,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    base_model: str | Path | None = None,
    revision: str = DEFAULT_REVISION,
    checkpoints_dir: str | Path | None = None,
    load_text_encoder: bool = True,
    load_transformer: bool = True,
    local_files_only: bool = True,
) -> CosmosComponents:
    """Load Cosmos Predict2.5-2B via ``Cosmos2_5_PredictBasePipeline`` (DiT4DiT-local only)."""
    from diffusers import Cosmos2_5_PredictBasePipeline

    if checkpoints_dir is None:
        checkpoints_dir = Path(__file__).resolve().parents[3] / "checkpoints"
    base_path = resolve_cosmos_base_model(
        base_model=base_model,
        checkpoints_dir=Path(checkpoints_dir),
    )
    incomplete = verify_cosmos_weight_files(base_path)
    if incomplete:
        raise FileNotFoundError(
            f"Incomplete Cosmos weights under {base_path}:\n  - "
            + "\n  - ".join(incomplete)
            + "\nRe-download following DiT4DiT README (see DiT4DiT/README.md)."
        )

    pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
        str(base_path),
        revision=revision,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
        safety_checker=_DefaultDummySafetyChecker(),
    )

    vae = pipe.vae.to(device=device, dtype=torch_dtype)
    transformer = (
        pipe.transformer.to(device=device, dtype=torch_dtype) if load_transformer else None
    )
    text_encoder = (
        pipe.text_encoder.to(device=device, dtype=torch_dtype) if load_text_encoder else None
    )
    tokenizer = pipe.tokenizer if load_text_encoder else None

    latents_mean = getattr(pipe, "latents_mean", None)
    latents_std = getattr(pipe, "latents_std", None)
    if latents_mean is not None and not isinstance(latents_mean, torch.Tensor):
        latents_mean = torch.tensor(latents_mean, dtype=torch_dtype).view(1, vae.config.z_dim, 1, 1, 1).to(device)
    elif latents_mean is not None:
        latents_mean = latents_mean.to(device=device, dtype=torch_dtype)
    if latents_std is not None and not isinstance(latents_std, torch.Tensor):
        latents_std = torch.tensor(latents_std, dtype=torch_dtype).view(1, vae.config.z_dim, 1, 1, 1).to(device)
    elif latents_std is not None:
        latents_std = latents_std.to(device=device, dtype=torch_dtype)

    if latents_mean is None and getattr(vae.config, "latents_mean", None) is not None:
        latents_mean = (
            torch.tensor(vae.config.latents_mean, dtype=torch_dtype)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(device)
        )
        latents_std = (
            torch.tensor(vae.config.latents_std, dtype=torch_dtype)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(device)
        )

    vae_scale_factor_temporal = int(getattr(pipe, "vae_scale_factor_temporal", 2 ** sum(vae.temperal_downsample)))
    vae_scale_factor_spatial = int(getattr(pipe, "vae_scale_factor_spatial", 2 ** len(vae.temperal_downsample)))

    if text_encoder is not None:
        enc_cfg = text_encoder.config
        hidden = int(getattr(enc_cfg, "hidden_size", 3584))
        n_layers = int(getattr(enc_cfg, "num_hidden_layers", 28))
        # ``encode_prompt`` concatenates normalized hidden states from layers 1..L.
        text_embed_dim = hidden * n_layers
    elif transformer is not None:
        text_embed_dim = int(transformer.config.crossattn_proj_in_channels)
    else:
        text_embed_dim = 1024
    in_channels = int(transformer.config.in_channels) if transformer is not None else 17

    del pipe

    return CosmosComponents(
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        latents_mean=latents_mean,
        latents_std=latents_std,
        vae_scale_factor_spatial=vae_scale_factor_spatial,
        vae_scale_factor_temporal=vae_scale_factor_temporal,
        text_embed_dim=text_embed_dim,
        transformer_in_channels=in_channels,
        base_model_path=base_path,
    )


def configure_hf_env(checkpoints_dir: str | Path | None = None) -> Path:
    """Return checkpoints root (legacy .env parsing kept for optional HF utilities)."""
    root = Path(__file__).resolve().parents[3]
    env_file = root / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

    ckpt = Path(checkpoints_dir) if checkpoints_dir else root / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    return ckpt

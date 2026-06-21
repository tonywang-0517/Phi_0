from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BridgeHeadConfig:
    input_dim: int
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    head_type: str = "mlp"


class MLPBridgeHead(nn.Module):
    """Lightweight trainable bridge: Phi_0 features -> 7D action logits."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(num_layers)):
            layers.extend(
                [
                    nn.Linear(in_dim, int(hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, 7))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBridgeHead(nn.Module):
    """Optional sequence bridge head over per-step Phi_0 features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(int(input_dim), int(hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=int(hidden_dim) * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.out_proj = nn.Linear(int(hidden_dim), 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Transformer bridge expects [T,D], got {tuple(x.shape)}")
        y = self.in_proj(x).unsqueeze(0)
        y = self.encoder(y).squeeze(0)
        return self.out_proj(y)


def build_bridge_head(cfg: BridgeHeadConfig) -> nn.Module:
    head_type = str(cfg.head_type).strip().lower()
    if head_type == "mlp":
        return MLPBridgeHead(
            input_dim=int(cfg.input_dim),
            hidden_dim=int(cfg.hidden_dim),
            num_layers=int(cfg.num_layers),
            dropout=float(cfg.dropout),
        )
    if head_type == "transformer":
        return TransformerBridgeHead(
            input_dim=int(cfg.input_dim),
            hidden_dim=int(cfg.hidden_dim),
            num_layers=int(cfg.num_layers),
            dropout=float(cfg.dropout),
        )
    raise ValueError(f"Unsupported bridge head type: {cfg.head_type}")


def bridge_loss(
    pred_logits: torch.Tensor,
    target_action: torch.Tensor,
    *,
    gripper_loss_weight: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pose regression + gripper BCE loss."""
    if pred_logits.shape != target_action.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred_logits.shape)} target={tuple(target_action.shape)}")
    pose_loss = F.mse_loss(pred_logits[:, :6], target_action[:, :6])
    grip_target = (target_action[:, 6] > 0).to(dtype=pred_logits.dtype)
    grip_loss = F.binary_cross_entropy_with_logits(pred_logits[:, 6], grip_target)
    total = pose_loss + float(gripper_loss_weight) * grip_loss
    return total, {
        "loss_total": float(total.detach().item()),
        "loss_pose": float(pose_loss.detach().item()),
        "loss_gripper": float(grip_loss.detach().item()),
    }


@torch.no_grad()
def bridge_logits_to_action(pred_logits: torch.Tensor) -> torch.Tensor:
    """Convert raw logits to 7D action with gripper in [0, 1]."""
    out = pred_logits.clone()
    out[..., 6] = torch.sigmoid(out[..., 6])
    return out


def save_bridge_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    config: BridgeHeadConfig,
    input_mode: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "input_dim": int(config.input_dim),
            "hidden_dim": int(config.hidden_dim),
            "num_layers": int(config.num_layers),
            "dropout": float(config.dropout),
            "head_type": str(config.head_type),
        },
        "input_mode": str(input_mode),
        "extra": dict(extra or {}),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)


def load_bridge_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    cfg_raw = payload.get("config", {})
    cfg = BridgeHeadConfig(
        input_dim=int(cfg_raw["input_dim"]),
        hidden_dim=int(cfg_raw.get("hidden_dim", 512)),
        num_layers=int(cfg_raw.get("num_layers", 2)),
        dropout=float(cfg_raw.get("dropout", 0.1)),
        head_type=str(cfg_raw.get("head_type", "mlp")),
    )
    model = build_bridge_head(cfg)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload

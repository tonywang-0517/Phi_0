#!/usr/bin/env python3
"""HTTP policy server for SIMPLE G1 whole-body eval (Psi0-compatible /act API)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from hydra import compose, initialize_config_dir
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from phi0.checkpoint_utils import merge_saved_cfg
from phi0.data.simple_action_norm import SIMPLE_G1_DIM
from phi0.deploy.helpers import RequestMessage, ResponseMessage
from phi0.inference.session import ActionInferenceSession
from phi0.models.vlm.preprocess import normalize_vlm_instruction
from phi0.runtime import (
    activate_cuda_device,
    apply_processor_stats_from_checkpoint,
    build_processor,
    create_phi0,
    resolve_inference_device,
    sync_model_action_norm,
)

logger = logging.getLogger(__name__)


def _pick_egocentric_image(image_dict: Dict[str, Any]) -> np.ndarray:
    for key in (
        "observation.images.egocentric",
        "egocentric",
        "ego_view",
        "image",
    ):
        if key in image_dict:
            return np.asarray(image_dict[key], dtype=np.uint8)
    if image_dict:
        return np.asarray(next(iter(image_dict.values())), dtype=np.uint8)
    raise KeyError("Request image dict is empty.")


class SimpleG1Server:
    def __init__(
        self,
        *,
        checkpoint: str,
        config_dir: str,
        config_name: str,
        device: str,
        min_free_gb: float,
        action_exec_horizon: Optional[int],
    ):
        self.device = resolve_inference_device(device, min_free_gb=min_free_gb)
        activate_cuda_device(self.device)

        with initialize_config_dir(version_base="1.3", config_dir=config_dir):
            cfg = compose(config_name=config_name)
        cfg.device = self.device

        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        saved_cfg = payload.get("cfg") if isinstance(payload, dict) else None
        if saved_cfg:
            cfg = merge_saved_cfg(cfg, saved_cfg)

        self.cfg = cfg
        self.model = create_phi0(cfg)
        if isinstance(payload, dict) and ("model" in payload or "action_expert" in payload):
            self.model.load_checkpoint(checkpoint)
        self.model.eval()

        self.processor = build_processor(cfg).eval()
        if isinstance(payload, dict):
            apply_processor_stats_from_checkpoint(self.processor, payload, cfg)
        sync_model_action_norm(self.model, self.processor)

        self.session = ActionInferenceSession(self.model, processor=self.processor)
        self.Da = int(getattr(self.model.action_expert, "raw_action_dim", SIMPLE_G1_DIM))
        future_steps = int(cfg.data.get("future_action_steps", 30))
        self.Tp = future_steps
        self.Ta = int(action_exec_horizon or future_steps)
        if self.Ta > self.Tp:
            raise ValueError(f"action_exec_horizon={self.Ta} exceeds chunk size {self.Tp}")
        logger.info(
            "Phi_0 SIMPLE server ready: Da=%d Tp=%d Ta=%d device=%s",
            self.Da,
            self.Tp,
            self.Ta,
            self.device,
        )

    def _normalized_proprio(self, states: np.ndarray) -> torch.Tensor:
        states = np.asarray(states, dtype=np.float32).reshape(-1)
        if states.size < SIMPLE_G1_DIM:
            pad = np.zeros(SIMPLE_G1_DIM - states.size, dtype=np.float32)
            states = np.concatenate([states, pad], axis=0)
        else:
            states = states[:SIMPLE_G1_DIM]
        tensor = torch.from_numpy(states).view(1, 1, SIMPLE_G1_DIM)
        normed = self.processor.normalize_robot_nd_tensor(
            tensor, dim=SIMPLE_G1_DIM, proprio=True
        )
        return normed.to(device=self.model.device, dtype=self.model.torch_dtype)

    def _image_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        h, w = rgb.shape[:2]
        target_h, target_w = self.processor.vlm_image_size
        if (h, w) != (target_h, target_w):
            pil = Image.fromarray(rgb).resize((target_w, target_h), Image.BILINEAR)
            rgb = np.asarray(pil, dtype=np.uint8)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor * 2.0 - 1.0
        return tensor.unsqueeze(0).to(device=self.model.device, dtype=self.model.torch_dtype)

    def predict_action(self, payload: Dict[str, Any]) -> JSONResponse:
        try:
            request = RequestMessage.deserialize(payload)
            rgb = _pick_egocentric_image(request.image)
            instruction = normalize_vlm_instruction(str(request.instruction).lower())
            states = request.state.get("states", request.state.get("state"))
            if states is None:
                raise KeyError("state dict missing 'states'")

            proprio = self._normalized_proprio(np.asarray(states))
            image_t = self._image_tensor(rgb)

            if self.session.action_ctx is None:
                self.session.prefill_from_image(image_t, instruction)
            else:
                self.session.refresh_video_context(image_t, prompt=instruction)
            self.session.set_proprio_gt(proprio)

            pred_norm = self.session.predict(self.Tp, denormalize=False)
            if pred_norm.ndim == 3:
                pred_norm = pred_norm.squeeze(0)
            pred_phys = self.processor.denormalize_robot_nd_future(
                pred_norm.unsqueeze(0), dim=SIMPLE_G1_DIM
            ).squeeze(0)
            pred_actions = pred_phys[: self.Ta].detach().cpu().numpy().astype(np.float32)
            response = ResponseMessage(pred_actions, 0.0)
            return JSONResponse(content=response.serialize())
        except Exception as exc:
            logger.exception("predict_action failed")
            return JSONResponse(content={"status": str(exc)}, status_code=500)

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        app = FastAPI()
        app.post("/act")(self.predict_action)
        app.get("/health")(lambda: JSONResponse(content={"status": "ok"}))
        logger.info("Listening on %s:%d", host, port)
        uvicorn.run(app, host=host, port=port)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve Phi_0 for SIMPLE G1 eval")
    p.add_argument("--checkpoint", required=True, help="Phi_0 training checkpoint (.pt)")
    p.add_argument("--config-dir", default=str(ROOT / "configs"))
    p.add_argument("--config-name", default="train_simple_g1_act")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--min-free-gb", type=float, default=8.0)
    p.add_argument("--action-exec-horizon", type=int, default=None)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    server = SimpleG1Server(
        checkpoint=args.checkpoint,
        config_dir=args.config_dir,
        config_name=args.config_name,
        device=args.device,
        min_free_gb=args.min_free_gb,
        action_exec_horizon=args.action_exec_horizon,
    )
    server.run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

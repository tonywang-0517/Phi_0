"""Inference session + eval caches (VLM context / chunk action predict)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch

from phi0.schema.draw_schema import zero_unsupervised_action_dims


def resolve_deploy_action_chunk_size(
    model,
    *,
    seq_len: int | None = None,
) -> int:
    """Training-aligned deploy horizon: ``seq_len - prefix window``."""
    if getattr(model, "uses_history_action_input", lambda: False)():
        w = int(getattr(model, "action_history_window", 0) or 0)
    else:
        w = int(getattr(model, "past_action_window_size", 1))
    if seq_len is None:
        seq_len = 9 if w <= 1 else (24 if w == 5 else w * 2)
    return max(1, int(seq_len) - w)


class VLMContextCache:
    """Cache VLM action context keyed by instruction string (single-task eval)."""

    def __init__(self) -> None:
        self._store: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def get(self, model, vlm_inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        key = vlm_inputs.get("_cache_key", "")
        if key and key in self._store:
            return self._store[key]
        if not model.uses_vlm_tower():
            batch_size = int(vlm_inputs.get("input_ids", torch.zeros(1)).shape[0])
            ctx, mask = model._dummy_action_context(
                batch_size,
                device=model.device,
                dtype=model.torch_dtype,
                text_dim=model.text_dim,
            )
        else:
            ctx, mask = model.vlm_tower.extract_action_context(
                vlm_inputs["input_ids"],
                vlm_inputs["attention_mask"],
                vlm_inputs["pixel_values"],
                vlm_inputs["image_grid_thw"],
                vlm_inputs.get("mm_token_type_ids"),
            )
        if key:
            self._store[key] = (ctx, mask)
        return ctx, mask

    def clear(self) -> None:
        self._store.clear()


# Backward-compatible alias
PromptEmbedCache = VLMContextCache


@dataclass
class AgentTurnResult:
    """Eval-only: frozen agent utterance from the first input (not per action chunk)."""

    text: str


@dataclass
class ClipInputsCache:
    """Lazy cache for ``build_inputs`` + VLM context per clip index."""

    _store: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get_or_build(
        self,
        model,
        processor,
        seq_dataset,
        clip_idx: int,
        *,
        prompt_cache: Optional[VLMContextCache] = None,
        cache_action_context: bool = True,
        collate_fn=None,
    ) -> Dict[str, Any]:
        if clip_idx in self._store:
            self.hits += 1
            return self._store[clip_idx]

        from phi0.data.sequence import SequenceDataset
        from phi0.runtime import prepare_model_batch

        if collate_fn is None:
            collate_fn = SequenceDataset.collate_fn

        self.misses += 1
        batch = collate_fn([seq_dataset[clip_idx]])
        mb = prepare_model_batch(model, processor, batch)
        mb = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in mb.items()}
        inputs = model.build_inputs(mb)
        if cache_action_context and "action_ctx" not in inputs:
            inputs["action_ctx"], inputs["action_ctx_mask"] = model._resolve_action_context(
                inputs=inputs
            )
        if model.uses_dual_vggt_cross_attn() and "vggt_ctx" not in inputs:
            if mb.get("vggt_video") is not None:
                vggt_ctx, vggt_ctx_mask = model._resolve_vggt_context(
                    mb["vggt_video"], inputs=mb
                )
                inputs["vggt_ctx"] = vggt_ctx
                inputs["vggt_ctx_mask"] = vggt_ctx_mask
        if "action_ctx" in inputs:
            if model.uses_cross_attn_context():
                context_emb, vggt_context_emb = model._embed_action_contexts(
                    inputs["action_ctx"],
                    inputs.get("vggt_ctx"),
                )
                inputs["context_emb"] = context_emb
                if vggt_context_emb is not None:
                    inputs["vggt_context_emb"] = vggt_context_emb
        self._store[clip_idx] = inputs
        return inputs

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "size": len(self._store)}


class ActionInferenceSession:
    """Stateful deploy: Qwen3-VL context + current-frame proprio + chunk predict."""

    def __init__(
        self,
        model,
        processor=None,
        *,
        deploy_seq_len: int = 9,
        action_video_freq_ratio: int = 2,
        use_gt_proprio: bool = False,
        use_gt_history: bool | None = None,
        max_rgb_frames: int = 33,
        use_wrist_view: bool = False,
        agent_speech_model_path: str | None = None,
    ) -> None:
        del max_rgb_frames, action_video_freq_ratio
        self.model = model
        self.processor = processor
        self.use_wrist_view = bool(use_wrist_view)
        self.deploy_seq_len = int(deploy_seq_len)
        if use_gt_history is not None:
            use_gt_proprio = bool(use_gt_history)
        self.use_gt_proprio = bool(use_gt_proprio)
        self.action_ctx: Optional[torch.Tensor] = None
        self.action_ctx_mask: Optional[torch.Tensor] = None
        self.context_emb: Optional[torch.Tensor] = None
        self.vggt_ctx: Optional[torch.Tensor] = None
        self.vggt_ctx_mask: Optional[torch.Tensor] = None
        self.vggt_context_emb: Optional[torch.Tensor] = None
        self._vlm_inputs: Optional[Dict[str, torch.Tensor]] = None
        self._video_clip: Optional[torch.Tensor] = None
        # Eval-only VLM agent speech (default off; never used in training).
        self._agent_speech_enabled: bool = False
        self._agent_speech_done: bool = False
        self._agent_text: str = ""
        self._agent_vlm_inputs: Optional[Dict[str, torch.Tensor]] = None
        self._agent_speech_model_path: str | None = (
            str(agent_speech_model_path).strip() if agent_speech_model_path else None
        )
        self._agent_speech_tower: Any | None = None
        self._agent_instruction: str = ""
        self._wrist_video_clip: Optional[torch.Tensor] = None
        w = int(getattr(model, "past_action_window_size", 1) or 1)
        if getattr(model, "uses_history_action_input", lambda: False)():
            history_window = int(getattr(model, "action_history_window", 0) or 0)
            self._proprio_history: Deque[torch.Tensor] = deque(maxlen=history_window or None)
        else:
            self._proprio_history = deque(maxlen=w or None)
        self._proprio_hold: Optional[torch.Tensor] = None
        self.video_refresh_count: int = 0

    @property
    def use_gt_history(self) -> bool:
        return self.use_gt_proprio

    def reset(self) -> None:
        self.action_ctx = None
        self.action_ctx_mask = None
        self.context_emb = None
        self.vggt_ctx = None
        self.vggt_ctx_mask = None
        self.vggt_context_emb = None
        self._vlm_inputs = None
        self._video_clip = None
        self._agent_speech_done = False
        self._agent_text = ""
        self._agent_vlm_inputs = None
        self._agent_instruction = ""
        self._wrist_video_clip = None
        self._proprio_history.clear()
        self._proprio_hold = None
        self.video_refresh_count = 0

    def seed_proprio_from_normalized(self, action: torch.Tensor) -> None:
        step = action.reshape(-1).detach().to(
            device=self.model.device, dtype=self.model.torch_dtype
        )
        self._proprio_hold = step
        self._proprio_history.clear()

    def seed_history_from_normalized(self, action: torch.Tensor) -> None:
        self.seed_proprio_from_normalized(action)

    def set_proprio_gt(self, proprio: torch.Tensor) -> None:
        w = int(getattr(self.model, "past_action_window_size", 1) or 1)
        if getattr(self.model, "uses_history_action_input", lambda: False)():
            w = int(getattr(self.model, "action_history_window", 0) or 0)
        if w <= 0:
            return
        if proprio.ndim == 2:
            proprio = proprio.unsqueeze(0)
        if proprio.shape[1] != w:
            raise ValueError(f"proprio must have {w} steps, got shape {tuple(proprio.shape)}")
        steps = [
            proprio[0, i].reshape(-1).detach().to(
                device=self.model.device, dtype=self.model.torch_dtype
            )
            for i in range(w)
        ]
        self._proprio_history.clear()
        for step in steps:
            self._proprio_history.append(step)
        self._proprio_hold = steps[-1]

    def set_history_gt(self, history: torch.Tensor) -> None:
        self.set_proprio_gt(history)

    def enable_agent_speech_for_eval(
        self,
        enabled: bool = True,
        *,
        model_path: str | None = None,
    ) -> None:
        """Opt-in eval-only agent AR. Call before the first ``prefill``; default off."""
        self._agent_speech_enabled = bool(enabled)
        if model_path is not None:
            path = str(model_path).strip()
            self._agent_speech_model_path = path or None
            self._agent_speech_tower = None
        if not enabled:
            self._agent_speech_done = False
            self._agent_text = ""
            self._agent_vlm_inputs = None
            self._agent_instruction = ""
            self._wrist_video_clip = None

    @property
    def agent_speech_enabled(self) -> bool:
        return self._agent_speech_enabled

    @property
    def agent_text(self) -> str:
        return self._agent_text

    def _snapshot_agent_vlm_inputs(self, vlm_inputs: Dict[str, torch.Tensor]) -> None:
        """Freeze first-eval VLM batch; later ``refresh_*`` must not re-run agent AR."""
        if not self._agent_speech_enabled or self._agent_vlm_inputs is not None:
            return
        self._agent_vlm_inputs = {
            k: v.detach().clone() if torch.is_tensor(v) else v
            for k, v in vlm_inputs.items()
        }

    def _resolve_agent_speech_tower(self):
        action_tower = self.model.vlm_tower
        path = self._agent_speech_model_path
        if not path:
            return action_tower
        if self._agent_speech_tower is None:
            from phi0.models.vlm.tower import load_agent_speech_tower

            attn = "sdpa"
            if action_tower is not None and hasattr(action_tower, "attn_implementation"):
                attn = str(action_tower.attn_implementation)
            self._agent_speech_tower = load_agent_speech_tower(
                path,
                device=str(self.model.device),
                torch_dtype=self.model.torch_dtype,
                attn_implementation=attn,
                local_files_only=False,
            )
        return self._agent_speech_tower

    def _build_vlm_inputs_from_video(
        self,
        video: torch.Tensor,
        instruction: str,
        *,
        wrist_video: torch.Tensor | None = None,
        vlm_tower=None,
    ) -> Dict[str, torch.Tensor]:
        from phi0.models.vlm.preprocess import (
            build_deploy_vlm_inputs_from_pixels,
            video_bcthw_to_pixel_batch,
        )

        if video.ndim != 5:
            raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video.shape)}")
        pixel = video_bcthw_to_pixel_batch(video)
        wrist_pixel = None
        if wrist_video is not None:
            if wrist_video.ndim != 5:
                raise ValueError(
                    f"wrist_video must be [B,3,T,H,W], got {tuple(wrist_video.shape)}"
                )
            wrist_pixel = video_bcthw_to_pixel_batch(wrist_video)
        tower = vlm_tower if vlm_tower is not None else self.model.vlm_tower
        processor_obj = getattr(tower, "processor", None)
        if processor_obj is None:
            batch = int(video.shape[0])
            seq = int(getattr(tower, "num_context_tokens", 16))
            device = video.device
            pixel_stat = float(pixel.mean())
            return {
                "input_ids": torch.ones(batch, seq, dtype=torch.long, device=device),
                "attention_mask": torch.ones(batch, seq, dtype=torch.bool, device=device),
                "pixel_values": torch.full((batch, 1), pixel_stat, device=device),
                "image_grid_thw": torch.ones(batch, 3, dtype=torch.long, device=device),
            }
        return build_deploy_vlm_inputs_from_pixels(
            processor_obj,
            self.processor,
            pixel.float(),
            [instruction],
            model_max_length=int(getattr(self.model, "prompt_max_length", 512)),
            wrist_pixel=wrist_pixel,
        )

    def _agent_speech_vlm_inputs(self) -> Dict[str, torch.Tensor] | None:
        speech_tower = self._resolve_agent_speech_tower()
        action_tower = self.model.vlm_tower
        if speech_tower is action_tower:
            return self._agent_vlm_inputs or self._vlm_inputs
        if self._video_clip is None:
            return self._agent_vlm_inputs or self._vlm_inputs
        vlm_inputs = self._build_vlm_inputs_from_video(
            self._video_clip,
            self._agent_instruction,
            wrist_video=self._wrist_video_clip,
            vlm_tower=speech_tower,
        )
        return {
            k: v.to(device=self.model.device) if torch.is_tensor(v) else v
            for k, v in vlm_inputs.items()
        }

    def _set_action_context(
        self,
        action_ctx: torch.Tensor,
        action_ctx_mask: torch.Tensor,
        *,
        vggt_ctx: torch.Tensor | None = None,
        vggt_ctx_mask: torch.Tensor | None = None,
        context_emb: torch.Tensor | None = None,
        vggt_context_emb: torch.Tensor | None = None,
    ) -> None:
        self.action_ctx = action_ctx
        self.action_ctx_mask = action_ctx_mask
        if self.model.uses_cross_attn_context():
            if context_emb is None or (
                vggt_context_emb is None and vggt_ctx is not None
            ):
                context_emb, vggt_context_emb = self.model._embed_action_contexts(
                    action_ctx,
                    vggt_ctx,
                    context_emb=context_emb,
                    vggt_context_emb=vggt_context_emb,
                )
            self.context_emb = context_emb
            self.vggt_context_emb = vggt_context_emb
        else:
            self.context_emb = None
            self.vggt_context_emb = None
        self.vggt_ctx = vggt_ctx
        self.vggt_ctx_mask = vggt_ctx_mask

    def _update_action_context_from_video(
        self,
        video: torch.Tensor,
        instruction: str,
        *,
        vggt_video: torch.Tensor | None = None,
        wrist_video: torch.Tensor | None = None,
    ) -> None:
        if video.ndim != 5:
            raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video.shape)}")
        if self.use_wrist_view and wrist_video is None:
            raise ValueError("use_wrist_view=True but wrist_video is missing.")
        video = video.to(device=self.model.device, dtype=self.model.torch_dtype)
        if wrist_video is not None:
            wrist_video = wrist_video.to(device=self.model.device, dtype=self.model.torch_dtype)
        self._video_clip = video.detach()
        if wrist_video is not None:
            self._wrist_video_clip = wrist_video.detach()
        else:
            self._wrist_video_clip = None
        if self._agent_speech_enabled:
            self._agent_instruction = str(instruction)
        batch_size = int(video.shape[0])

        if not self.model.uses_vlm_tower():
            self._vlm_inputs = None
            action_ctx, action_ctx_mask = self.model._dummy_action_context(
                batch_size,
                device=self.model.device,
                dtype=self.model.torch_dtype,
                text_dim=self.model.text_dim,
            )
            vggt_ctx, vggt_ctx_mask = None, None
        else:
            vlm_inputs = self._build_vlm_inputs_from_video(
                video, instruction, wrist_video=wrist_video
            )
            for key, value in vlm_inputs.items():
                vlm_inputs[key] = value.to(device=self.model.device)
            self._vlm_inputs = vlm_inputs
            self._snapshot_agent_vlm_inputs(vlm_inputs)
            action_ctx, action_ctx_mask = self.model.vlm_tower.extract_action_context(
                vlm_inputs["input_ids"],
                vlm_inputs["attention_mask"],
                vlm_inputs["pixel_values"],
                vlm_inputs["image_grid_thw"],
                vlm_inputs.get("mm_token_type_ids"),
            )
            vggt_src = vggt_video if vggt_video is not None else video[:, :, -1:, :, :]
            if self.model.uses_dual_vggt_cross_attn():
                vggt_ctx, vggt_ctx_mask = self.model._resolve_vggt_context(
                    vggt_src, inputs={"vggt_video": vggt_src}
                )
            else:
                vggt_ctx, vggt_ctx_mask = None, None

        self._set_action_context(action_ctx, action_ctx_mask, vggt_ctx=vggt_ctx, vggt_ctx_mask=vggt_ctx_mask)
        self.video_refresh_count += 1

    def _proprio_tensor(self) -> Optional[torch.Tensor]:
        w = int(getattr(self.model, "past_action_window_size", 1) or 1)
        if getattr(self.model, "uses_history_action_input", lambda: False)():
            w = int(getattr(self.model, "action_history_window", 0) or 0)
        if w <= 0:
            return None
        hist = list(self._proprio_history)
        anchor = hist[-1] if hist else self._proprio_hold
        if anchor is None:
            raise RuntimeError(
                "Proprio deploy requires set_proprio_gt() or seed_proprio_from_normalized() "
                "before predict()."
            )
        if len(hist) >= w:
            steps = hist[-w:]
        else:
            steps = [anchor] * w
        return torch.stack(steps, dim=0).unsqueeze(0)

    def _history_tensor(self) -> torch.Tensor:
        tensor = self._proprio_tensor()
        if tensor is None:
            raise RuntimeError("proprio/history window must be positive for deploy.")
        return tensor

    def _update_proprio_history(self, pred_norm: torch.Tensor) -> None:
        if self.use_gt_proprio:
            return
        w = int(getattr(self.model, "past_action_window_size", 1) or 1)
        if getattr(self.model, "uses_history_action_input", lambda: False)():
            w = int(getattr(self.model, "action_history_window", 0) or 0)
        if w <= 0:
            return
        if pred_norm.ndim == 3:
            steps = [pred_norm[0, i] for i in range(pred_norm.shape[1])]
        else:
            steps = [pred_norm[i] for i in range(pred_norm.shape[0])]
        for step in steps:
            self._proprio_history.append(step.detach())
        if steps:
            self._proprio_hold = steps[-1].detach()

    def _update_action_history(self, pred_norm: torch.Tensor) -> None:
        self._update_proprio_history(pred_norm)

    @torch.no_grad()
    def refresh_video_context_from_clip(
        self,
        video: torch.Tensor,
        *,
        prompt: Optional[str] = None,
        vggt_video: Optional[torch.Tensor] = None,
        wrist_video: Optional[torch.Tensor] = None,
    ) -> None:
        if self.action_ctx is None:
            raise RuntimeError("Call prefill before refresh.")
        if prompt is None:
            raise ValueError("refresh_video_context_from_clip requires `prompt`.")
        self._update_action_context_from_video(
            video, prompt, vggt_video=vggt_video, wrist_video=wrist_video
        )

    @torch.no_grad()
    def refresh_video_context(
        self,
        input_image: torch.Tensor,
        *,
        prompt: Optional[str] = None,
    ) -> None:
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        clip = input_image.unsqueeze(2)
        self.refresh_video_context_from_clip(clip, prompt=prompt)

    @torch.no_grad()
    def prefill_from_video_clip(
        self,
        video: torch.Tensor,
        prompt: Optional[str] = None,
        *,
        prompt_cache: Optional[VLMContextCache] = None,
        vggt_video: Optional[torch.Tensor] = None,
        wrist_video: Optional[torch.Tensor] = None,
    ) -> None:
        del prompt_cache
        if prompt is None:
            raise ValueError("prefill_from_video_clip requires `prompt`.")
        self._update_action_context_from_video(
            video, prompt, vggt_video=vggt_video, wrist_video=wrist_video
        )

    @torch.no_grad()
    def prefill_from_image(
        self,
        input_image: torch.Tensor,
        prompt: Optional[str] = None,
        *,
        prompt_cache: Optional[VLMContextCache] = None,
    ) -> None:
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        clip = input_image.unsqueeze(2)
        self.prefill_from_video_clip(clip, prompt, prompt_cache=prompt_cache)

    @torch.no_grad()
    def prefill_from_clip_inputs(self, inputs: Dict[str, Any]) -> None:
        if "action_ctx" in inputs and "action_ctx_mask" in inputs:
            action_ctx = inputs["action_ctx"]
            action_ctx_mask = inputs["action_ctx_mask"]
        elif all(k in inputs for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")):
            action_ctx, action_ctx_mask = self.model._resolve_action_context(inputs=inputs)
        elif not self.model.uses_vlm_tower():
            batch_size = int(inputs.get("action", torch.zeros(1, 1, 1)).shape[0])
            action_ctx, action_ctx_mask = self.model._dummy_action_context(
                batch_size,
                device=self.model.device,
                dtype=self.model.torch_dtype,
                text_dim=self.model.text_dim,
            )
        else:
            raise ValueError("Cannot prefill: missing VLM inputs or precomputed action_ctx.")
        vggt_ctx = None
        vggt_ctx_mask = None
        if self.model.uses_dual_vggt_cross_attn():
            if "vggt_ctx" in inputs and "vggt_ctx_mask" in inputs:
                vggt_ctx = inputs["vggt_ctx"]
                vggt_ctx_mask = inputs["vggt_ctx_mask"]
            elif inputs.get("vggt_video") is not None:
                vggt_ctx, vggt_ctx_mask = self.model._resolve_vggt_context(
                    inputs["vggt_video"], inputs=inputs
                )
            else:
                raise ValueError("dual_vlm_vggt prefill requires vggt_video or vggt_ctx.")
        self._set_action_context(
            action_ctx,
            action_ctx_mask,
            vggt_ctx=vggt_ctx,
            vggt_ctx_mask=vggt_ctx_mask,
            context_emb=inputs.get("context_emb"),
            vggt_context_emb=inputs.get("vggt_context_emb"),
        )

    @torch.no_grad()
    def predict(self, num_frames: int, *, denormalize: bool = False) -> torch.Tensor:
        if self.action_ctx is None:
            raise RuntimeError("Call prefill before predict().")
        batch_size = int(self.action_ctx.shape[0])
        prefix = self._proprio_tensor()
        pred = self.model.predict_action(
            self.action_ctx,
            self.action_ctx_mask,
            int(num_frames),
            batch_size=batch_size,
            context_emb=self.context_emb,
            vggt_context=self.vggt_ctx,
            vggt_context_mask=self.vggt_ctx_mask,
            vggt_context_emb=self.vggt_context_emb,
            proprio_tokens=None if self.model.uses_history_action_input() else prefix,
            history=prefix if self.model.uses_history_action_input() else None,
        )
        pred = zero_unsupervised_action_dims(pred)
        self._update_proprio_history(pred)
        out = pred.squeeze(0) if batch_size == 1 else pred
        if denormalize and self.processor is not None:
            if out.ndim == 2:
                return self.processor.postprocess(out.unsqueeze(0)).squeeze(0)
            return self.processor.postprocess(out)
        return out

    @torch.no_grad()
    def run_agent_speech_once(
        self,
        *,
        gen_cfg: Any | None = None,
        batch_index: int = 0,
        **generate_kwargs: Any,
    ) -> str:
        """Eval-only: one HF ``generate`` on the first input, then no-op until ``reset()``."""
        if not self._agent_speech_enabled:
            return ""
        if self._agent_speech_done:
            return self._agent_text
        vlm_inputs = self._agent_speech_vlm_inputs()
        if vlm_inputs is None or not self.model.uses_vlm_tower():
            return ""
        tower = self._resolve_agent_speech_tower()
        gen_fn = getattr(tower, "generate_text_from_vlm_batch", None)
        if gen_fn is None:
            return ""
        texts = gen_fn(vlm_inputs, gen_cfg=gen_cfg, **generate_kwargs)
        idx = int(batch_index)
        if idx < 0 or idx >= len(texts):
            raise IndexError(f"batch_index {idx} out of range for {len(texts)} replies")
        self._agent_text = str(texts[idx])
        self._agent_speech_done = True
        return self._agent_text

    @torch.no_grad()
    def predict_segments(
        self,
        num_frames: int,
        segment_len: int,
        *,
        denormalize: bool = False,
    ) -> torch.Tensor:
        segment_len = max(1, int(segment_len))
        chunks: List[torch.Tensor] = []
        remaining = int(num_frames)
        while remaining > 0:
            cur = min(segment_len, remaining)
            chunks.append(self.predict(cur, denormalize=False))
            remaining -= cur
        out = torch.cat(chunks, dim=0)
        if denormalize and self.processor is not None:
            return self.processor.postprocess(out.unsqueeze(0)).squeeze(0)
        return out

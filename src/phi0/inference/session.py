"""Inference session + eval caches (prompt / clip vision / chunk action predict)."""

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
        w = int(getattr(model, "past_action_window_size", 4))
    if seq_len is None:
        seq_len = 33 if w <= 4 else w * 2
    return max(1, int(seq_len) - w)


def _cosmos_video_input(video: torch.Tensor) -> torch.Tensor:
    """Single-frame Cosmos I2V conditioning (current control step)."""
    if video.ndim != 5:
        raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video.shape)}")
    return video[:, :, -1:, :, :]


class PromptEmbedCache:
    """Cache Cosmos text encoder outputs keyed by instruction string."""

    def __init__(self) -> None:
        self._store: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def get(self, model, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if prompt not in self._store:
            embeds, mask = model.encode_prompt(prompt)
            self._store[prompt] = (embeds, mask)
        return self._store[prompt]

    def get_batch(self, model, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(set(prompts)) == 1:
            e, m = self.get(model, prompts[0])
            if e.shape[0] == 1:
                return e.expand(len(prompts), -1, -1), m.expand(len(prompts), -1)
            return e, m
        embeds, mask = model.encode_prompt(prompts)
        return embeds, mask

    def clear(self) -> None:
        self._store.clear()


@dataclass
class ClipInputsCache:
    """Lazy cache for ``build_inputs`` + Cosmos hook per clip index."""

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
        prompt_cache: Optional[PromptEmbedCache] = None,
        cache_action_context: bool = True,
    ) -> Dict[str, Any]:
        if clip_idx in self._store:
            self.hits += 1
            return self._store[clip_idx]

        from phi0.data.sequence import SequenceDataset
        from phi0.runtime import prepare_model_batch

        self.misses += 1
        batch = SequenceDataset.collate_fn([seq_dataset[clip_idx]])
        mb = prepare_model_batch(model, processor, batch, prompt_cache=prompt_cache)
        mb = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in mb.items()}
        inputs = model.build_inputs(mb)
        if cache_action_context and getattr(model.video_tower, "transformer", None) is not None:
            _, action_ctx, action_ctx_mask = model.video_tower.forward_joint_step(
                _cosmos_video_input(inputs["video"]),
                inputs["context"],
                compute_video_loss=False,
            )
            inputs["action_ctx"] = action_ctx
            inputs["action_ctx_mask"] = action_ctx_mask
        if model.uses_dual_vggt_cross_attn() and "video" in inputs:
            vggt_ctx, vggt_ctx_mask = model._resolve_vggt_context(inputs["video"], inputs=inputs)
            inputs["vggt_ctx"] = vggt_ctx
            inputs["vggt_ctx_mask"] = vggt_ctx_mask
        if "action_ctx" in inputs:
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
    """Stateful deploy: Cosmos single-frame hook + proprio prefix + chunk predict."""

    def __init__(
        self,
        model,
        processor=None,
        *,
        deploy_seq_len: int = 33,
        action_video_freq_ratio: int = 2,
        use_gt_proprio: bool = False,
        use_gt_history: bool | None = None,
        max_rgb_frames: int = 33,
    ) -> None:
        del max_rgb_frames  # legacy arg; clip is passed explicitly on refresh
        self.model = model
        self.processor = processor
        self.deploy_seq_len = int(deploy_seq_len)
        self.action_video_freq_ratio = max(1, int(action_video_freq_ratio))
        if use_gt_history is not None:
            use_gt_proprio = bool(use_gt_history)
        self.use_gt_proprio = bool(use_gt_proprio)
        self.action_ctx: Optional[torch.Tensor] = None
        self.action_ctx_mask: Optional[torch.Tensor] = None
        self.context_emb: Optional[torch.Tensor] = None
        self.vggt_ctx: Optional[torch.Tensor] = None
        self.vggt_ctx_mask: Optional[torch.Tensor] = None
        self.vggt_context_emb: Optional[torch.Tensor] = None
        self._prompt_embeds: Optional[torch.Tensor] = None
        self._video_clip: Optional[torch.Tensor] = None
        if getattr(model, "uses_history_action_input", lambda: False)():
            history_window = int(getattr(model, "action_history_window", 0) or 0)
            self._proprio_history: Deque[torch.Tensor] = deque(maxlen=history_window or None)
        else:
            self._proprio_history = deque(
                maxlen=int(getattr(model, "past_action_window_size", 0)) or None
            )
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
        self._prompt_embeds = None
        self._video_clip = None
        self._proprio_history.clear()
        self._proprio_hold = None
        self.video_refresh_count = 0

    def seed_proprio_from_normalized(self, action: torch.Tensor) -> None:
        """Cold-start proprio with the current frame (replicated), not all-zero."""
        step = action.reshape(-1).detach().to(
            device=self.model.device, dtype=self.model.torch_dtype
        )
        self._proprio_hold = step
        self._proprio_history.clear()

    def seed_history_from_normalized(self, action: torch.Tensor) -> None:
        """Deprecated alias for ``seed_proprio_from_normalized``."""
        self.seed_proprio_from_normalized(action)

    def set_proprio_gt(self, proprio: torch.Tensor) -> None:
        """Set proprio prefix from GT normalized actions ``[W,D]`` or ``[1,W,D]``."""
        if getattr(self.model, "uses_history_action_input", lambda: False)():
            w = int(getattr(self.model, "action_history_window", 0) or 0)
        else:
            w = int(getattr(self.model, "past_action_window_size", 0))
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
        """Deprecated alias for ``set_proprio_gt``."""
        self.set_proprio_gt(history)

    def _update_action_context_from_video(self, video: torch.Tensor) -> None:
        if self._prompt_embeds is None:
            raise RuntimeError("Missing prompt embeds; call prefill before refresh.")
        if video.ndim != 5:
            raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video.shape)}")
        video = video.to(device=self.model.device, dtype=self.model.torch_dtype)
        self._video_clip = video.detach()
        tower = self.model.video_tower
        video_cosmos = _cosmos_video_input(video)
        if getattr(tower, "transformer", None) is not None:
            _, self.action_ctx, self.action_ctx_mask = tower.forward_joint_step(
                video_cosmos,
                self._prompt_embeds,
                compute_video_loss=False,
            )
        else:
            self.action_ctx, self.action_ctx_mask = tower.extract_action_context(
                video_cosmos, self._prompt_embeds
            )
        if self.model.uses_dual_vggt_cross_attn():
            self.vggt_ctx, self.vggt_ctx_mask = self.model._resolve_vggt_context(
                video, inputs={"vggt_video": video}
            )
        else:
            self.vggt_ctx = None
            self.vggt_ctx_mask = None
        self.context_emb, self.vggt_context_emb = self.model._embed_action_contexts(
            self.action_ctx,
            self.vggt_ctx,
        )
        self.video_refresh_count += 1

    def _proprio_tensor(self) -> Optional[torch.Tensor]:
        if getattr(self.model, "uses_history_action_input", lambda: False)():
            w = int(getattr(self.model, "action_history_window", 0) or 0)
        else:
            w = int(getattr(self.model, "past_action_window_size", 0))
        if w <= 0:
            return None
        hist = list(self._proprio_history)
        anchor = hist[-1] if hist else self._proprio_hold
        if anchor is None:
            raise RuntimeError(
                "Proprio deploy requires set_proprio_gt() or seed_proprio_from_normalized() "
                "before predict(); all-zero proprio is not used."
            )
        if len(hist) >= w:
            steps = hist[-w:]
        else:
            steps = [anchor] * w
        return torch.stack(steps, dim=0).unsqueeze(0)

    def _history_tensor(self) -> torch.Tensor:
        """Deprecated alias for ``_proprio_tensor`` (history-mode ablations)."""
        tensor = self._proprio_tensor()
        if tensor is None:
            raise RuntimeError("proprio/history window must be positive for deploy.")
        return tensor

    def _update_proprio_history(self, pred_norm: torch.Tensor) -> None:
        if self.use_gt_proprio:
            return
        if getattr(self.model, "uses_history_action_input", lambda: False)():
            w = int(getattr(self.model, "action_history_window", 0) or 0)
        else:
            w = int(getattr(self.model, "past_action_window_size", 0))
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
        prompt_embeds: Optional[torch.Tensor] = None,
    ) -> None:
        """Refresh Cosmos hook from training-aligned multi-frame clip ``[1,3,T,H,W]``."""
        if self.action_ctx is None:
            raise RuntimeError("Call prefill_from_video_clip or prefill_from_clip_inputs before refresh.")
        if prompt_embeds is not None:
            self._prompt_embeds = prompt_embeds
        self._update_action_context_from_video(video)

    @torch.no_grad()
    def refresh_video_context(
        self,
        input_image: torch.Tensor,
        *,
        prompt_embeds: Optional[torch.Tensor] = None,
    ) -> None:
        """Legacy single-frame refresh (wraps as 1-frame clip)."""
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        clip = input_image.unsqueeze(2)
        self.refresh_video_context_from_clip(clip, prompt_embeds=prompt_embeds)

    @torch.no_grad()
    def prefill_from_video_clip(
        self,
        video: torch.Tensor,
        prompt: Optional[str] = None,
        *,
        prompt_cache: Optional[PromptEmbedCache] = None,
    ) -> None:
        """Encode prompt once and capture hook from multi-frame clip ``[1,3,T,H,W]``."""
        if prompt is not None and self.model.video_tower.text_encoder is not None:
            if prompt_cache is not None:
                prompt_embeds, _ = prompt_cache.get(self.model, prompt)
            else:
                prompt_embeds, _ = self.model.encode_prompt(prompt)
            self._prompt_embeds = prompt_embeds
            self._update_action_context_from_video(video)
        else:
            raise ValueError("prefill_from_video_clip requires `prompt` when text encoder is loaded.")

    @torch.no_grad()
    def prefill_from_image(
        self,
        input_image: torch.Tensor,
        prompt: Optional[str] = None,
        *,
        prompt_cache: Optional[PromptEmbedCache] = None,
    ) -> None:
        """Legacy single-frame prefill (wraps as 1-frame clip)."""
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        clip = input_image.unsqueeze(2)
        self.prefill_from_video_clip(clip, prompt, prompt_cache=prompt_cache)

    @torch.no_grad()
    def prefill_from_clip_inputs(self, inputs: Dict[str, Any]) -> None:
        if "action_ctx" in inputs and "action_ctx_mask" in inputs:
            self.action_ctx = inputs["action_ctx"]
            self.action_ctx_mask = inputs["action_ctx_mask"]
        elif getattr(self.model.video_tower, "transformer", None) is not None:
            self._prompt_embeds = inputs["context"]
            _, self.action_ctx, self.action_ctx_mask = self.model.video_tower.forward_joint_step(
                _cosmos_video_input(inputs["video"]),
                inputs["context"],
                compute_video_loss=False,
            )
        else:
            raise ValueError("Cannot prefill: missing action context and no transformer.")
        if self.model.uses_dual_vggt_cross_attn():
            if "vggt_ctx" in inputs and "vggt_ctx_mask" in inputs:
                self.vggt_ctx = inputs["vggt_ctx"]
                self.vggt_ctx_mask = inputs["vggt_ctx_mask"]
            elif "vggt_video" in inputs or "video" in inputs:
                self.vggt_ctx, self.vggt_ctx_mask = self.model._resolve_vggt_context(
                    inputs.get("video"), inputs=inputs
                )
            else:
                raise ValueError("dual_cosmos_vggt prefill requires video or vggt_ctx in inputs.")
        else:
            self.vggt_ctx = None
            self.vggt_ctx_mask = None
        self.context_emb, self.vggt_context_emb = self.model._embed_action_contexts(
            self.action_ctx,
            self.vggt_ctx,
            context_emb=inputs.get("context_emb"),
            vggt_context_emb=inputs.get("vggt_context_emb"),
        )

    @torch.no_grad()
    def predict(self, num_frames: int, *, denormalize: bool = False) -> torch.Tensor:
        """Predict future action chunk [T, D] (or [B,T,D] if B>1)."""
        if self.action_ctx is None:
            raise RuntimeError("Call prefill_from_video_clip or prefill_from_clip_inputs before predict().")
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
        if batch_size == 1:
            out = pred.squeeze(0)
        else:
            out = pred
        if denormalize and self.processor is not None:
            if out.ndim == 2:
                return self.processor.postprocess(out.unsqueeze(0)).squeeze(0)
            return self.processor.postprocess(out)
        return out

    @torch.no_grad()
    def predict_segments(
        self,
        num_frames: int,
        segment_len: int,
        *,
        denormalize: bool = False,
    ) -> torch.Tensor:
        """Predict in segments (caller refreshes video context between segments)."""
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

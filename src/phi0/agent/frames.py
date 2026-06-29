"""Load eval frames for agent demos (ep447 ego + wrist)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from phi0.models.vlm.preprocess import tensor_frame_to_pil


def _to_pil(frame) -> Image.Image:
    import torch

    t = frame.detach().float() if torch.is_tensor(frame) else torch.as_tensor(frame).float()
    if t.max() > 1.0:
        t = t / 255.0
    return tensor_frame_to_pil(t)


def load_pick_tissue_episode_images(
    episode_idx: int,
    *,
    config_name: str = "train_pick_tissue_xperience_unified_ddp4_3k",
    config_dir: str | None = None,
) -> tuple[Image.Image, Image.Image | None, int, str]:
    from hydra import compose, initialize_config_dir

    from phi0.data.pick_tissue_unified import PickTissueUnifiedClipDataset
    from phi0.deploy.pick_tissue_gt import clip_dataset_index_for_episode
    from phi0.runtime import build_base_dataset

    root = Path(__file__).resolve().parents[3]
    cfg_dir = config_dir or str(root / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(config_name=config_name)
    base = build_base_dataset(cfg)
    clip_row = clip_dataset_index_for_episode(base, episode_idx, data_cfg=cfg.data)
    batch = PickTissueUnifiedClipDataset.collate_fn([base[clip_row]])
    ego = _to_pil(batch["images"]["ego_view"][0, 0])
    wrist = None
    if "wrist_view" in batch["images"]:
        wrist = _to_pil(batch["images"]["wrist_view"][0, 0])
    return ego, wrist, clip_row, str(batch["task"][0])

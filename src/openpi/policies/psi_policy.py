import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

def make_psi_policy(model_config: _model.BaseModelConfig):
    """Creates a random input example for the Droid policy."""
    return {
        "base_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        # TODO
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class HfmInputs(transforms.DataTransformFn):
    
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        head_image = _parse_image(data["observation/image"])

        proprio_state = np.asarray(data["states"], dtype=np.float32).reshape(-1)

        left_wrist_image = None
        if "observation/wrist_image" in data:
            left_wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": proprio_state,
            "image": {
                "base_0_rgb": head_image,
                "left_wrist_0_rgb": left_wrist_image
                if left_wrist_image is not None
                else np.zeros_like(head_image),
                "right_wrist_0_rgb": np.zeros_like(head_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if left_wrist_image is not None else np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        def nice(x):
            return " ".join([y.lower() for y in x.split("/")[-1].split("_")]) 
        
        if "actions" in data:
            # inputs["actions"] = np.concatenate([
            #     data["actions"][:, :4], # H + RPY (4): 1 dof base height + 3 torso rpy
            #     data["actions"][:, 4:18], # ARM (14): 7 dof left arm + 7 dof right arm
            #     data["actions"][:, 18:32], # HAND (14): 7 dof left hand + 7 dof right hand
            #     # data["actions"][:, 32:], # LEG (15): 6 dof left leg + 6 dof right leg + 3 dof waist
            # ], axis=1)
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = nice(data["prompt"]) # "pick up dumpling toy and squat to put on the chair" # data["prompt"] # FIXME use natual language prompt later
        
        return inputs

@dataclasses.dataclass(frozen=True)
class HfmOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # return {"actions": np.asarray(data["actions"][:, :8])}
        return data
"""Official SMPL-X (SMPL+H hands) mesh forward + matplotlib rendering for Phi_0.

Xperience / Phi_0 unified actions store root pose in **Isaac Z-up world** (``Ts_world_root``).
Mesh viz therefore:

1. ``smplx.forward`` with world ``transl`` (matches FK numerically).
2. Pelvis-relative offsets in that world frame.
3. ``_isaac_world_delta_to_standing`` → matplotlib display (+Z up, feet @ z=0).

Pure SMPL ``transl=0`` + SMPL Y-up→Z-up is only valid for native SMPL/HMR params,
not for Isaac-world Xperience quats (see ``ref.md`` vs GR00T conventions).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from phi0.schema.unified_action_schema import (
    body_rot6d_to_smpl_pose_aa,
    rot6d_to_axis_angle,
    unpack_body_rot6d_local,
    unpack_left_hand_rot6d_local,
    unpack_right_hand_rot6d_local,
    unpack_root_rot6d,
)
from phi0.viz.skeleton import SMPLX_TO_XPERIENCE_JOINT_REMAP


def resolve_smplx_models_root() -> Path:
    """Directory passed to ``smplx.create`` (contains ``smplx/SMPLX_NEUTRAL.npz``)."""
    root = Path(__file__).resolve().parents[3]
    for base in (
        root / "data/body_models",
        root / "data/body_models/_hf_cache/models",
    ):
        if (base / "smplx" / "SMPLX_NEUTRAL.npz").is_file():
            return base
        if (base / "SMPLX_NEUTRAL.npz").is_file():
            return base.parent if base.name == "smplx" else base
    raise FileNotFoundError(
        "SMPLX_NEUTRAL.npz not found under Phi_0/data/body_models. "
        "Place models per scripts/setup_smplh_body_models.sh or HF cache layout."
    )


@lru_cache(maxsize=1)
def load_smplx_neutral():
    import smplx

    models_root = resolve_smplx_models_root()
    return smplx.create(
        str(models_root),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        num_betas=16,
        flat_hand_mean=True,
    )


def _as_batch_f32(x: np.ndarray, dim: int) -> "Any":
    import torch

    t = torch.from_numpy(np.asarray(x, dtype=np.float32).reshape(-1)[:dim][None])
    return t


def unified_action_to_smplx_inputs(
    d: np.ndarray,
    *,
    betas: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Map unified action to SMPL-X pose parameters (no world translation)."""
    global_orient = rot6d_to_axis_angle(unpack_root_rot6d(d)).astype(np.float32).reshape(3)
    body_pose = body_rot6d_to_smpl_pose_aa(unpack_body_rot6d_local(d)).reshape(-1).astype(np.float32)
    left_hand = rot6d_to_axis_angle(unpack_left_hand_rot6d_local(d)).reshape(-1).astype(np.float32)
    right_hand = rot6d_to_axis_angle(unpack_right_hand_rot6d_local(d)).reshape(-1).astype(np.float32)
    if betas is None:
        betas_np = np.zeros(16, dtype=np.float32)
    else:
        betas_np = np.asarray(betas, dtype=np.float32).reshape(-1)[:16]
    return {
        "global_orient": global_orient,
        "body_pose": body_pose,
        "left_hand_pose": left_hand,
        "right_hand_pose": right_hand,
        "betas": betas_np,
    }


def hdf5_quat_frame_to_smplx_inputs(f: Mapping, t: int, betas: np.ndarray) -> dict[str, np.ndarray]:
    """Independent SMPL-X inputs from Xperience HDF5 quats (dataset reference)."""
    from scipy.spatial.transform import Rotation as R

    root = f["full_body_mocap/Ts_world_root"][t].astype(np.float32)
    body_q = f["full_body_mocap/body_quats"][t].astype(np.float32)
    lh_q = f["full_body_mocap/left_hand_quats"][t].astype(np.float32)
    rh_q = f["full_body_mocap/right_hand_quats"][t].astype(np.float32)

    def quat_to_aa(q_wxyz: np.ndarray) -> np.ndarray:
        q = np.asarray(q_wxyz, dtype=np.float32).reshape(-1, 4)
        return R.from_quat(q[:, [1, 2, 3, 0]]).as_rotvec().astype(np.float32)

    return {
        "global_orient": quat_to_aa(root[3:7]).reshape(3),
        "body_pose": quat_to_aa(body_q).reshape(-1),
        "left_hand_pose": quat_to_aa(lh_q).reshape(-1),
        "right_hand_pose": quat_to_aa(rh_q).reshape(-1),
        "betas": np.asarray(betas, dtype=np.float32).reshape(-1)[:16],
    }


def _smpl_yup_to_display_zup(points: np.ndarray) -> np.ndarray:
    """SMPL body model is Y-up; Rx(+90°) about X maps to matplotlib Z-up.

    Matches GR00T ``smpl_root_ytoz_up``: ``(x, y, z) -> (x, -z, y)``.
    Do **not** use Isaac standing ``(x, z, -y)`` here — that maps Isaac Z-up to
  display Y-up, and inverts SMPL +Y to -Z.
    """
    p = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=-1).astype(np.float32)


def smplx_forward(
    inputs: dict[str, np.ndarray],
    *,
    transl: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run SMPL-X forward; return (vertices, joints_52, faces)."""
    import torch

    model = load_smplx_neutral()
    if transl is None:
        transl = np.zeros(3, dtype=np.float32)
    with torch.no_grad():
        out = model(
            betas=_as_batch_f32(inputs["betas"], 16),
            global_orient=_as_batch_f32(inputs["global_orient"], 3),
            body_pose=_as_batch_f32(inputs["body_pose"], 63),
            left_hand_pose=_as_batch_f32(inputs["left_hand_pose"], 45),
            right_hand_pose=_as_batch_f32(inputs["right_hand_pose"], 45),
            transl=_as_batch_f32(transl, 3),
        )
    verts = out.vertices[0].detach().cpu().numpy().astype(np.float32)
    joints = out.joints[0].detach().cpu().numpy().astype(np.float32)[SMPLX_TO_XPERIENCE_JOINT_REMAP]
    faces = np.asarray(model.faces, dtype=np.int64)
    if hasattr(faces, "detach"):
        faces = faces.detach().cpu().numpy()
    return verts, joints, faces


def hdf5_quat_transl_world(f: Mapping, t: int) -> np.ndarray:
    """World root translation from Xperience ``Ts_world_root``."""
    return f["full_body_mocap/Ts_world_root"][t][:3].astype(np.float32)


def smplx_forward_mesh_viz(
    inputs: dict[str, np.ndarray],
    *,
    transl_world: np.ndarray,
    upright_display: bool = True,
    ground_feet: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Pelvis-relative SMPL-X mesh in Isaac standing display frame."""
    from phi0.viz.skeleton import FOOT_JOINT_INDICES, _isaac_world_delta_to_standing

    transl = np.asarray(transl_world, dtype=np.float32).reshape(3)
    verts, joints, faces = smplx_forward(inputs, transl=transl)
    pelvis = joints[0]
    verts = verts - pelvis
    joints = joints - pelvis
    if upright_display:
        verts = _isaac_world_delta_to_standing(verts)
        joints = _isaac_world_delta_to_standing(joints)
    if ground_feet:
        foot_z = float(np.min(joints[FOOT_JOINT_INDICES, 2]))
        verts[:, 2] -= foot_z
    return verts, faces


def smplx_forward_mesh(
    inputs: dict[str, np.ndarray],
    *,
    transl_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Standing-display mesh (alias for eval viz)."""
    return smplx_forward_mesh_viz(inputs, transl_world=transl_world)


def compute_mesh_viz_bounds(
    *vertex_sets: np.ndarray,
    margin: float = 1.12,
) -> tuple[np.ndarray, float]:
    """Fixed cubic bounds over mesh vertex clouds."""
    if not vertex_sets:
        raise ValueError("compute_mesh_viz_bounds needs at least one vertex array")
    pts = np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1, 3) for v in vertex_sets], axis=0)
    center = pts.mean(axis=0)
    radius = float(np.max(np.linalg.norm(pts - center, axis=1))) * margin + 1e-3
    return center.astype(np.float32), radius


def draw_smplx_mesh(
    ax,
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    color: str = "tab:blue",
    alpha: float = 0.45,
    edgecolor: str | None = None,
) -> None:
    """Draw one SMPL-X mesh on a matplotlib 3D axis."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    v = np.asarray(verts, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    tris = v[f]
    kw: dict[str, Any] = {"alpha": alpha, "linewidths": 0.0}
    if edgecolor is not None:
        kw["edgecolors"] = edgecolor
        kw["linewidths"] = 0.05
    else:
        kw["edgecolors"] = "none"
    coll = Poly3DCollection(tris, **kw)
    coll.set_facecolor(color)
    ax.add_collection3d(coll)

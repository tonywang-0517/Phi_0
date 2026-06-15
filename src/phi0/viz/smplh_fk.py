"""Forward kinematics for SMPL-H 52-joint skeleton from Phi_0 d_raw (no mesh / full pkl).

Uses minimal skeleton constants (J_template + J_shapedirs) extracted from SMPL-X neutral
model with ``SMPLX_TO_XPERIENCE_JOINT_REMAP`` — see ``data/body_models/smplh_skeleton_constants.npz``.
FK parent tree is ``SMPLH_PARENTS`` (GR00T / Xperience 52-joint layout), not raw SMPL-X kintree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from phi0.schema.action_schema import D_RAW
from phi0.schema.draw_schema import neutral_betas
from phi0.viz.skeleton import SMPLH_PARENTS, NUM_JOINTS


def unpack_smplh_for_viz(d_raw: np.ndarray, *, use_neutral_betas: bool = False) -> dict[str, np.ndarray]:
    """Legacy quat layout unpack for FK audit scripts only."""
    betas = neutral_betas(1)[0] if use_neutral_betas else d_raw[211:227].copy()
    return {
        "root_trans": d_raw[0:3].copy(),
        "root_quat_wxyz": d_raw[3:7].copy(),
        "body_quats": d_raw[7:91].reshape(21, 4).copy(),
        "left_hand_quats": d_raw[91:151].reshape(15, 4).copy(),
        "right_hand_quats": d_raw[151:211].reshape(15, 4).copy(),
        "betas": betas,
    }


def pack_xperience_frame_quat(
    ts_world_root: np.ndarray,
    body_quats: np.ndarray,
    left_hand_quats: np.ndarray,
    right_hand_quats: np.ndarray,
    betas: np.ndarray,
    tactile: np.ndarray,
) -> np.ndarray:
    """Legacy quat layout packer for FK audit scripts only."""
    out = np.zeros(D_RAW, dtype=np.float32)
    out[0:3] = ts_world_root[:3]
    out[3:7] = ts_world_root[3:7]
    out[7:91] = body_quats.reshape(-1)
    out[91:151] = left_hand_quats.reshape(-1)
    out[151:211] = right_hand_quats.reshape(-1)
    out[211:227] = betas
    out[227:237] = tactile
    return out


def load_gt_quat_d_raw_from_hdf5(hdf5_path, start: int, count: int) -> np.ndarray:
    """Load legacy quat-packed GT rows from Xperience HDF5 (FK audit only)."""
    import h5py

    rows: list[np.ndarray] = []
    with h5py.File(hdf5_path, "r") as f:
        end = start + count
        for t in range(start, end):
            root = f["full_body_mocap/Ts_world_root"][t].astype(np.float32)
            body = f["full_body_mocap/body_quats"][t].astype(np.float32)
            lh = f["full_body_mocap/left_hand_quats"][t].astype(np.float32)
            rh = f["full_body_mocap/right_hand_quats"][t].astype(np.float32)
            betas = f["full_body_mocap/betas"][t].astype(np.float32)
            tactile = np.zeros(10, dtype=np.float32)
            rows.append(pack_xperience_frame_quat(root, body, lh, rh, betas, tactile))
    return np.stack(rows, axis=0)


_DEFAULT_CONSTANTS = (
    Path(__file__).resolve().parents[3] / "data" / "body_models" / "smplh_skeleton_constants.npz"
)


def _default_constants_path() -> Path:
    return _DEFAULT_CONSTANTS


def load_skeleton_constants(path: Path | None = None) -> dict[str, np.ndarray]:
    """Load T-pose joint template and beta blend shapes for FK."""
    path = Path(path or _default_constants_path())
    if not path.is_file():
        raise FileNotFoundError(
            f"SMPL-H skeleton constants not found at {path}. "
            "Run scripts/extract_smplh_skeleton_constants.py or place SMPLH pkl under data/body_models/smplh/."
        )
    data = np.load(path)
    if len(data["J_template"]) != NUM_JOINTS:
        raise ValueError(f"Expected {NUM_JOINTS} joints in J_template, got {len(data['J_template'])}")
    parents = data["parents"].astype(np.int32)
    if not np.array_equal(parents, SMPLH_PARENTS):
        # Always FK with GR00T / Xperience parent tree (matches keypoints drawing).
        parents = SMPLH_PARENTS.copy()
    return {
        "J_template": data["J_template"].astype(np.float32),
        "J_shapedirs": data["J_shapedirs"].astype(np.float32),
        "parents": parents,
    }


def get_skeleton(betas: np.ndarray, constants: dict[str, np.ndarray]) -> np.ndarray:
    """T-pose joint positions shaped by betas: (J, 3) or (T, J, 3)."""
    J_template = constants["J_template"]
    J_shapedirs = constants["J_shapedirs"]
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim == 1:
        return J_template + np.einsum("k,jck->jc", betas[:J_shapedirs.shape[-1]], J_shapedirs)
    return J_template + np.einsum("tk,jck->tjc", betas[:, :J_shapedirs.shape[-1]], J_shapedirs)


def stack_local_quats_wxyz(d_raw_row: np.ndarray) -> np.ndarray:
    """Stack 52 local quaternions (wxyz) from one D_raw row."""
    parts = unpack_smplh_for_viz(d_raw_row)
    quats = np.concatenate(
        [
            parts["root_quat_wxyz"],
            parts["body_quats"].reshape(-1),
            parts["left_hand_quats"].reshape(-1),
            parts["right_hand_quats"].reshape(-1),
        ],
        axis=0,
    )
    return quats.reshape(NUM_JOINTS, 4).astype(np.float32)


def quat_wxyz_to_matrix(quats: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 3, 3) rotation matrices."""
    q = np.asarray(quats, dtype=np.float32)
    w, x, y, z = np.moveaxis(q, -1, 0)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    norm = np.maximum(norm, 1e-8)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    o = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
    o[..., 0, 0] = 1 - 2 * (yy + zz)
    o[..., 0, 1] = 2 * (xy - wz)
    o[..., 0, 2] = 2 * (xz + wy)
    o[..., 1, 0] = 2 * (xy + wz)
    o[..., 1, 1] = 1 - 2 * (xx + zz)
    o[..., 1, 2] = 2 * (yz - wx)
    o[..., 2, 0] = 2 * (xz - wy)
    o[..., 2, 1] = 2 * (yz + wx)
    o[..., 2, 2] = 1 - 2 * (xx + yy)
    return o


def batch_rigid_transform(
    rot_mats: np.ndarray,
    joints: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    """
    SMPL-style FK: local rotations + rest skeleton -> global joint positions.

    Args:
        rot_mats: (..., J, 3, 3)
        joints: (..., J, 3) rest pose joint positions
        parents: (J,) parent indices, root parent = -1
    Returns:
        posed_joints: (..., J, 3)
    """
    rot_mats = np.asarray(rot_mats, dtype=np.float32)
    joints = np.asarray(joints, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int32)

    prefix = rot_mats.shape[:-3]
    j_count = rot_mats.shape[-3]
    if joints.shape[:-2] != prefix:
        joints = np.broadcast_to(joints, prefix + (j_count, 3))

    rel_joints = joints.copy()
    rel_joints[..., 1:, :] -= joints[..., parents[1:], :]

    transforms = np.zeros(prefix + (j_count, 4, 4), dtype=np.float32)
    transforms[..., :, :3, :3] = rot_mats
    transforms[..., :, :3, 3] = rel_joints
    transforms[..., :, 3, 3] = 1.0

    chain = [transforms[..., 0, :, :]]
    for i in range(1, j_count):
        parent = int(parents[i])
        chain.append(chain[parent] @ transforms[..., i, :, :])

    posed = np.stack(chain, axis=-3)[..., :3, 3].copy()
    return posed


def _resolve_fk_betas(d_raw: np.ndarray, *, use_d_raw_betas: bool) -> np.ndarray:
    if use_d_raw_betas:
        return d_raw[:, 211:227]
    return neutral_betas(d_raw.shape[0] if d_raw.ndim > 1 else 1)


def joints_from_d_raw(
    d_raw: np.ndarray,
    constants: dict[str, np.ndarray] | None = None,
    constants_path: Path | None = None,
    *,
    use_d_raw_betas: bool = False,
) -> np.ndarray:
    """
    Recover global 52-joint positions from D_raw rows.

    Uses the same rigid-chain FK as SMPL-X ``lbs`` (verified numerically against
    ``smplx`` forward on GT d_raw). Output is in the SMPL world frame
    (``posed + root_trans``). It does **not** match Xperience HDF5 ``keypoints``
    directly — apply ``phi0.viz.xperience_viz_frame.fk_joints_to_keypoints_frame``
    for skeleton overlay with GT keypoints.

    Args:
        d_raw: (T, D_RAW) or (D_RAW,)
    Returns:
        joints: (T, 52, 3) or (52, 3) SMPL-style world frame (not HDF5 keypoints frame).
    """
    d_raw = np.asarray(d_raw, dtype=np.float32)
    single = d_raw.ndim == 1
    if single:
        d_raw = d_raw[None]

    if constants is None:
        constants = load_skeleton_constants(constants_path)

    parents = constants["parents"]
    betas = _resolve_fk_betas(d_raw, use_d_raw_betas=use_d_raw_betas)
    skeleton = get_skeleton(betas, constants)

    quats = np.stack([stack_local_quats_wxyz(row) for row in d_raw], axis=0)
    rot_mats = quat_wxyz_to_matrix(quats)
    posed = batch_rigid_transform(rot_mats, skeleton, parents)

    # SMPL adds global translation to all joints (root_trans is world translation).
    root_trans = d_raw[:, 0:3]
    posed = posed + root_trans[:, None, :]
    return posed[0] if single else posed


def joints_from_d_raw_batch(
    d_raw: np.ndarray,
    constants: dict[str, np.ndarray] | None = None,
    *,
    use_d_raw_betas: bool = False,
) -> np.ndarray:
    """Vectorized FK for (T, D_RAW) — same as joints_from_d_raw but avoids Python loop on quats.

    Default ``use_d_raw_betas=False``: neutral (zero) betas for model deploy preds.
    Pass ``use_d_raw_betas=True`` when ``d_raw`` is GT-packed from HDF5 (211:227 valid).
    """
    d_raw = np.asarray(d_raw, dtype=np.float32)
    if constants is None:
        constants = load_skeleton_constants()

    parents = constants["parents"]
    betas = _resolve_fk_betas(d_raw, use_d_raw_betas=use_d_raw_betas)
    skeleton = get_skeleton(betas, constants)

    root_q = d_raw[:, 3:7]
    body_q = d_raw[:, 7:91].reshape(-1, 21, 4)
    lh_q = d_raw[:, 91:151].reshape(-1, 15, 4)
    rh_q = d_raw[:, 151:211].reshape(-1, 15, 4)
    quats = np.concatenate([root_q[:, None], body_q, lh_q, rh_q], axis=1)
    rot_mats = quat_wxyz_to_matrix(quats)
    posed = batch_rigid_transform(rot_mats, skeleton, parents)
    posed = posed + d_raw[:, None, 0:3]
    return posed

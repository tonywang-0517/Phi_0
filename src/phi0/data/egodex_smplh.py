"""EgoDex ARKit transforms -> sparse Phi_0 D_raw (256-d) + per-dim availability mask."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np

from phi0.schema.action_schema import D_RAW, LEGACY_QUAT_SLICES as SLICES
from phi0.schema.draw_schema import compute_tactile_proxy_from_tip_positions

# SMPL-H body_quats index (0..20) -> EgoDex transform key (upper-body sparse map).
# body_quats[i] corresponds to skeleton joint (i + 1); see phi0.viz.skeleton.SMPLH_PARENTS.
EGODEX_TO_BODY_QUAT: Dict[int, str] = {
    2: "spine1",
    5: "spine4",
    8: "spine7",
    11: "neck2",
    12: "leftShoulder",
    16: "leftArm",
    17: "leftForearm",
    19: "rightShoulder",
    20: "rightForearm",
}

# SMPL-H left/right hand local joint order (15 each) -> EgoDex transform keys.
# One EgoDex joint per SMPL-H hand quaternion slot (sparse fingertip / knuckle coverage).
LEFT_HAND_EGODEX_KEYS: Tuple[str, ...] = (
    "leftIndexFingerMetacarpal",
    "leftIndexFingerKnuckle",
    "leftIndexFingerIntermediateBase",
    "leftMiddleFingerMetacarpal",
    "leftMiddleFingerKnuckle",
    "leftMiddleFingerIntermediateBase",
    "leftRingFingerMetacarpal",
    "leftRingFingerKnuckle",
    "leftRingFingerIntermediateBase",
    "leftLittleFingerMetacarpal",
    "leftLittleFingerKnuckle",
    "leftLittleFingerIntermediateBase",
    "leftThumbKnuckle",
    "leftThumbIntermediateBase",
    "leftThumbTip",
)

RIGHT_HAND_EGODEX_KEYS: Tuple[str, ...] = tuple(k.replace("left", "right") for k in LEFT_HAND_EGODEX_KEYS)

ROOT_EGODEX_KEY = "hip"
MIN_CONFIDENCE = 0.25


def _identity4() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def mat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """Rotation matrix (3,3) -> unit quaternion (w,x,y,z)."""
    m = np.asarray(rot, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def invert_se3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = _identity4()
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def world_to_camera(T_world: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    return invert_se3(T_world_cam) @ T_world


@dataclass
class EgoDexFrameSMPLH:
    d_raw: np.ndarray
    dim_available: np.ndarray


def _read_transform(f: h5py.File, key: str, t: int) -> np.ndarray:
    return f[f"transforms/{key}"][t].astype(np.float64)


def _read_confidence(f: h5py.File, key: str, t: int, default: float = 1.0) -> float:
    path = f"confidences/{key}"
    if path not in f:
        return default
    return float(f[path][t])


def _skeleton_joint_to_body_slice(joint_idx: int) -> slice:
    """Skeleton joint 1..21 -> d_raw body_quats slice."""
    body_i = joint_idx - 1
    s = SLICES["body_quats"][0] + body_i * 4
    return slice(s, s + 4)


def _hand_slice(hand: str, local_i: int) -> slice:
    base = SLICES["left_hand_quats" if hand == "left" else "right_hand_quats"][0]
    s = base + local_i * 4
    return slice(s, s + 4)


def convert_egodex_frame(
    f: h5py.File,
    t: int,
    *,
    camera_frame: bool = True,
    min_confidence: float = MIN_CONFIDENCE,
    include_tactile_proxy: bool = False,
) -> EgoDexFrameSMPLH:
    """Convert one EgoDex frame to sparse D_raw + dim_available (256,)."""
    d_raw = np.zeros(D_RAW, dtype=np.float32)
    dim_available = np.zeros(D_RAW, dtype=bool)

    T_cam = _read_transform(f, "camera", t)
    T_root = _read_transform(f, ROOT_EGODEX_KEY, t)
    if camera_frame:
        T_root = world_to_camera(T_root, T_cam)

    root_conf = _read_confidence(f, ROOT_EGODEX_KEY, t)
    if root_conf >= min_confidence:
        d_raw[0:3] = T_root[:3, 3].astype(np.float32)
        d_raw[3:7] = mat_to_quat_wxyz(T_root[:3, :3])
        dim_available[0:7] = True

    # Global rotations in camera frame for mapped joints (skeleton index -> R_global).
    global_rots: Dict[int, np.ndarray] = {}
    if root_conf >= min_confidence:
        global_rots[0] = T_root[:3, :3]

    for body_i, eg_key in EGODEX_TO_BODY_QUAT.items():
        skel_j = body_i + 1
        conf = _read_confidence(f, eg_key, t)
        if conf < min_confidence:
            continue
        T_j = _read_transform(f, eg_key, t)
        if camera_frame:
            T_j = world_to_camera(T_j, T_cam)
        global_rots[skel_j] = T_j[:3, :3]

    # Wrist / hand roots attach to body joints 19/20 (left/right wrist in SMPL body_quats).
    for side, wrist_skel_j, hand_keys in (
        ("left", 20, LEFT_HAND_EGODEX_KEYS),
        ("right", 21, RIGHT_HAND_EGODEX_KEYS),
    ):
        wrist_key = "leftHand" if side == "left" else "rightHand"
        wrist_conf = _read_confidence(f, wrist_key, t)
        if wrist_conf >= min_confidence:
            T_w = _read_transform(f, wrist_key, t)
            if camera_frame:
                T_w = world_to_camera(T_w, T_cam)
            global_rots[wrist_skel_j] = T_w[:3, :3]

        for local_i, eg_key in enumerate(hand_keys):
            conf = _read_confidence(f, eg_key, t)
            if conf < min_confidence:
                continue
            T_h = _read_transform(f, eg_key, t)
            if camera_frame:
                T_h = world_to_camera(T_h, T_cam)
            skel_j = 22 + local_i if side == "left" else 37 + local_i
            global_rots[skel_j] = T_h[:3, :3]

    # Local quaternions along SMPL-H parent tree (phi0.viz.skeleton.SMPLH_PARENTS).
    from phi0.viz.skeleton import SMPLH_PARENTS

    parents = SMPLH_PARENTS
    for j in range(1, len(parents)):
        if j not in global_rots:
            continue
        p = int(parents[j])
        if p < 0:
            R_local = global_rots[j]
        elif p in global_rots:
            R_local = global_rots[p].T @ global_rots[j]
        else:
            continue
        quat = mat_to_quat_wxyz(R_local)
        if j <= 21:
            sl = _skeleton_joint_to_body_slice(j)
        elif j <= 36:
            sl = _hand_slice("left", j - 22)
        else:
            sl = _hand_slice("right", j - 37)
        d_raw[sl] = quat
        dim_available[sl] = True

    if include_tactile_proxy:
        tips: Dict[str, np.ndarray] = {}
        for key in (
            "leftHand",
            "rightHand",
            *LEFT_HAND_EGODEX_KEYS,
            *RIGHT_HAND_EGODEX_KEYS,
        ):
            if f"transforms/{key}" not in f:
                continue
            T = _read_transform(f, key, t)
            if camera_frame:
                T = world_to_camera(T, T_cam)
            tips[key] = T[:3, 3].astype(np.float32)
        if tips:
            left_t = compute_tactile_proxy_from_tip_positions(tips, "left")
            right_t = compute_tactile_proxy_from_tip_positions(tips, "right")
            d_raw[227:237] = np.concatenate([left_t, right_t], axis=0)
            dim_available[227:237] = True

    return EgoDexFrameSMPLH(d_raw=d_raw, dim_available=dim_available)


def default_processed_path(source_hdf5: Path) -> Path:
    return source_hdf5.with_name(f"{source_hdf5.stem}_smplh.hdf5")


def preprocess_egodex_file(
    source_hdf5: Path,
    output_hdf5: Path,
    *,
    camera_frame: bool = True,
    min_confidence: float = MIN_CONFIDENCE,
    include_tactile_proxy: bool = False,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
) -> Dict[str, object]:
    """Write processed sparse SMPL+H HDF5 next to (or under) the raw EgoDex episode."""
    source_hdf5 = Path(source_hdf5)
    output_hdf5 = Path(output_hdf5)
    output_hdf5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_hdf5, "r") as src:
        n_total = int(src["transforms/camera"].shape[0])
        end = n_total if max_frames is None else min(n_total, max_frames)
        frame_indices = list(range(0, end, max(1, frame_stride)))
        d_raw = np.zeros((len(frame_indices), D_RAW), dtype=np.float32)
        dim_frame = np.zeros((len(frame_indices), D_RAW), dtype=bool)
        dim_union = np.zeros(D_RAW, dtype=bool)

        for out_i, t in enumerate(frame_indices):
            frame = convert_egodex_frame(
                src,
                t,
                camera_frame=camera_frame,
                min_confidence=min_confidence,
                include_tactile_proxy=include_tactile_proxy,
            )
            d_raw[out_i] = frame.d_raw
            dim_frame[out_i] = frame.dim_available
            dim_union |= frame.dim_available

        with h5py.File(output_hdf5, "w") as dst:
            dst.create_dataset("d_raw", data=d_raw, compression="gzip")
            dst.create_dataset("dim_available", data=dim_union.astype(np.uint8))
            dst.create_dataset("dim_available_frame", data=dim_frame.astype(np.uint8))
            dst.create_dataset("frame_indices", data=np.asarray(frame_indices, dtype=np.int32))
            dst.attrs["source_hdf5"] = str(source_hdf5)
            dst.attrs["camera_frame"] = int(camera_frame)
            dst.attrs["min_confidence"] = float(min_confidence)
            dst.attrs["include_tactile_proxy"] = int(include_tactile_proxy)
            for k, v in src.attrs.items():
                try:
                    dst.attrs[k] = v
                except Exception:
                    pass

    return {
        "source": str(source_hdf5),
        "output": str(output_hdf5),
        "num_frames": len(frame_indices),
        "dim_available_count": int(dim_union.sum()),
        "dim_available_ratio": float(dim_union.sum() / D_RAW),
    }


def iter_raw_egodex_hdf5(root: Path) -> Iterable[Path]:
    root = Path(root)
    for path in sorted(root.rglob("*.hdf5")):
        if path.name.endswith("_smplh.hdf5"):
            continue
        if path.with_name(f"{path.stem}_smplh.hdf5").exists():
            # still allow re-processing via explicit output path
            pass
        yield path


def resolve_processed_hdf5(
    hdf5_path: Path,
    processed_path: Optional[Path] = None,
) -> Optional[Path]:
    if processed_path is not None:
        p = Path(processed_path)
        return p if p.is_file() else None
    default = default_processed_path(hdf5_path)
    return default if default.is_file() else None

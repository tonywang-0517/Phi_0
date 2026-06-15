#!/usr/bin/env python3
"""Audit Phi_0 FK pipeline: GT self-consistency, pred vs GT, SMPL-X reference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phi0.data.xperience import XperienceDataset
from phi0.schema.draw_schema import zero_unsupervised_action_dims_np
from phi0.viz.skeleton import iter_bone_segments, load_gt_from_hdf5, load_jsonl_predictions
from phi0.viz.smplh_fk import (
    batch_rigid_transform,
    get_skeleton,
    joints_from_d_raw_batch,
    load_gt_quat_d_raw_from_hdf5,
    load_skeleton_constants,
    quat_wxyz_to_matrix,
)
from phi0.schema.action_schema import unpack_keypoints_52
from phi0.viz.xperience_viz_frame import fk_joints_to_keypoints_frame

HDF5 = Path("/mnt/data2/wpy/workspace/Isaac-GR00T/demo_data/xperience-10m-sample/annotation.hdf5")
PRED = ROOT / "experiments/phi0_full/smplh_predictions_2k.jsonl"


def bone_stats(keypoints: np.ndarray) -> dict:
    lens = [float(np.linalg.norm(c - p)) for p, c in iter_bone_segments(keypoints)]
    return {"mean": float(np.mean(lens)), "max": float(np.max(lens)), "min": float(np.min(lens))}


def fk_variants(d_raw: np.ndarray, constants: dict) -> dict[str, np.ndarray]:
    """FK with different root-translation conventions."""
    parents = constants["parents"]
    betas = d_raw[:, 211:227]
    skel = get_skeleton(betas, constants)
    root_q = d_raw[:, 3:7]
    body_q = d_raw[:, 7:91].reshape(-1, 21, 4)
    lh_q = d_raw[:, 91:151].reshape(-1, 15, 4)
    rh_q = d_raw[:, 151:211].reshape(-1, 15, 4)
    quats = np.concatenate([root_q[:, None], body_q, lh_q, rh_q], axis=1)
    rot = quat_wxyz_to_matrix(quats)
    posed = batch_rigid_transform(rot, skel, parents)
    rt = d_raw[:, None, 0:3]
    return {
        "add_trans": posed + rt,
        "recenter_pelvis": posed - posed[:, :1, :] + rt,
        "no_trans": posed,
        "trans_only_root_rot": posed + rt,  # alias
    }


def smplx_joints_if_available(d_raw_row: np.ndarray) -> np.ndarray | None:
    try:
        import torch
        import smplx
        from scipy.spatial.transform import Rotation as R
    except ImportError:
        return None

    model_dir = ROOT / "data/body_models/_hf_cache/models/smplx"
    if not (model_dir / "SMPLX_NEUTRAL.npz").is_file() and not (model_dir / "smplx").is_dir():
        return None
    from phi0.viz.skeleton import SMPLX_TO_XPERIENCE_JOINT_REMAP

    model = smplx.create(
        str(model_dir.parent),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        num_betas=16,
        flat_hand_mean=True,
    )

    def quat_to_aa(q_wxyz: np.ndarray) -> np.ndarray:
        return R.from_quat(q_wxyz.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_rotvec().reshape(q_wxyz.shape[0], 3)

    betas = torch.from_numpy(d_raw_row[211:227][None].astype(np.float32))
    global_orient = torch.from_numpy(
        R.from_quat(d_raw_row[3:7][[1, 2, 3, 0]]).as_rotvec()[None].astype(np.float32)
    )
    body = d_raw_row[7:91].reshape(21, 4)
    lh = d_raw_row[91:151].reshape(15, 4)
    rh = d_raw_row[151:211].reshape(15, 4)
    body_aa = torch.from_numpy(quat_to_aa(body)[None].astype(np.float32))
    lh_aa = torch.from_numpy(quat_to_aa(lh)[None].astype(np.float32))
    rh_aa = torch.from_numpy(quat_to_aa(rh)[None].astype(np.float32))
    transl = torch.from_numpy(d_raw_row[0:3][None].astype(np.float32))
    with torch.no_grad():
        out = model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_aa,
            left_hand_pose=lh_aa,
            right_hand_pose=rh_aa,
            transl=transl,
        )
    return out.joints.numpy()[0, SMPLX_TO_XPERIENCE_JOINT_REMAP, :3]


def main():
    n = 33
    gt_d_quat = load_gt_quat_d_raw_from_hdf5(HDF5, 0, n)
    xp = XperienceDataset(hdf5_path=HDF5, max_frames=n)
    gt_d_kp = np.stack([xp._load_frame_action(t) for t in range(n)], axis=0)
    gt_h5 = load_gt_from_hdf5(HDF5, 0, n)
    pred_d, _ = load_jsonl_predictions(PRED)
    pred_d = zero_unsupervised_action_dims_np(pred_d[:n])
    constants = load_skeleton_constants()

    print("=" * 60)
    print("1) GT FK self-check (quat d_raw from HDF5 → FK → viz frame vs keypoints)")
    gt_fk = joints_from_d_raw_batch(gt_d_quat, constants, use_d_raw_betas=True)
    gt_viz = fk_joints_to_keypoints_frame(gt_fk)
    kp = gt_h5["keypoints_hdf5"]
    err_viz = np.linalg.norm(gt_viz - kp, axis=-1).mean(axis=1)
    print(f"   GT FK→viz vs HDF5 keypoints: mean joint L2 = {err_viz.mean():.4f} m (max frame {err_viz.max():.4f})")

    gt_fk_neutral = joints_from_d_raw_batch(gt_d_quat, constants, use_d_raw_betas=False)
    err_neutral = np.linalg.norm(
        fk_joints_to_keypoints_frame(gt_fk_neutral) - kp, axis=-1
    ).mean(axis=1)
    print(f"   GT FK (neutral betas) vs keypoints: {err_neutral.mean():.4f} m")

    print("\n2) Root translation conventions (GT frame 0)")
    variants = fk_variants(gt_d_quat[:1], constants)
    for name, joints in variants.items():
        j = fk_joints_to_keypoints_frame(joints[0])
        e = np.linalg.norm(j - kp[0], axis=-1).mean()
        bs = bone_stats(j)
        print(f"   {name:20s} → keypoints L2={e:.4f}  bones mean={bs['mean']:.3f} max={bs['max']:.3f}")

    print("\n3) Pred vs GT keypoints (deploy output, d_raw[0:156])")
    pred_kp = unpack_keypoints_52(pred_d)
    gt_kp = unpack_keypoints_52(gt_d_kp)
    err_kp = np.linalg.norm(pred_kp - gt_kp, axis=-1).mean(axis=1)
    print(f"   keypoints mean joint L2: {err_kp.mean():.4f} m")

    print("\n4) Pred keypoints vs GT HDF5 keypoints")
    err_pred = np.linalg.norm(pred_kp - kp, axis=-1).mean(axis=1)
    print(f"   pred vs GT keypoints: {err_pred.mean():.4f} m")
    print(f"   pred vs GT FK viz: {np.linalg.norm(pred_kp - gt_viz, axis=-1).mean():.4f} m")

    print("\n5) Bone length sanity (frame 0)")
    for label, pts in [("GT keypoints", kp[0]), ("GT FK viz", gt_viz[0]), ("Pred keypoints", pred_kp[0])]:
        bs = bone_stats(pts)
        print(f"   {label}: mean={bs['mean']:.3f} max={bs['max']:.3f}")

    print("\n6) SMPL-X reference (frame 0 quat d_raw)")
    ours_fk = joints_from_d_raw_batch(gt_d_quat[:1], constants, use_d_raw_betas=True)[0]
    j_smplx = smplx_joints_if_available(gt_d_quat[0])
    if j_smplx is None:
        print("   smplx not available or model missing — skip")
    else:
        e_smpl = np.linalg.norm(ours_fk - j_smplx, axis=-1).mean()
        print(f"   our FK vs SMPL-X forward: {e_smpl:.2e} m (should be ~0)")
        e_smpl_viz = np.linalg.norm(fk_joints_to_keypoints_frame(j_smplx[None])[0] - kp[0], axis=-1).mean()
        print(f"   SMPL-X→viz vs keypoints: {e_smpl_viz:.4f} m")

    print("\n7) body_quats layout check (GT frame 0 norms)")
    bq = gt_d_quat[0, 7:91].reshape(21, 4)
    print(f"   body quat norms: min={np.linalg.norm(bq,axis=1).min():.3f} max={np.linalg.norm(bq,axis=1).max():.3f}")

    out = {
        "gt_fk_viz_vs_keypoints_mean_m": float(err_viz.mean()),
        "pred_viz_vs_keypoints_mean_m": float(err_pred.mean()),
        "pred_viz_vs_gt_fk_viz_mean_m": float(np.linalg.norm(pred_viz - gt_viz, axis=-1).mean()),
        "pred_pose_l2_mean": float(np.linalg.norm(diff, axis=1).mean()),
        "pred_root_trans_l2_mean_m": float(np.linalg.norm(pred_d[:, :3] - gt_d[:, :3], axis=1).mean()),
    }
    out_path = ROOT / "experiments/phi0_full/fk_audit.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

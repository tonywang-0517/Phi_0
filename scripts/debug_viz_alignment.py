#!/usr/bin/env python3
"""Debug pred vs GT skeleton alignment for viz issues."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phi0.viz.skeleton import load_gt_from_hdf5, load_jsonl_predictions
from phi0.viz.smplh_fk import joints_from_d_raw_batch, load_skeleton_constants


def main():
    pred_path = ROOT / "experiments/phi0_full/smplh_predictions_2k.jsonl"
    hdf5 = Path("/mnt/data2/wpy/workspace/Isaac-GR00T/demo_data/xperience-10m-sample/annotation.hdf5")

    d_raw, frames = load_jsonl_predictions(pred_path)
    gt = load_gt_from_hdf5(hdf5, 0, len(frames))
    constants = load_skeleton_constants()

    pred_kp = joints_from_d_raw_batch(d_raw, constants, use_d_raw_betas=False)
    gt_d_raw = gt["d_raw"]
    gt_fk = joints_from_d_raw_batch(gt_d_raw, constants, use_d_raw_betas=True)
    gt_kp_hdf5 = gt["keypoints_hdf5"]

    print("=== frame 0 root ===")
    print("pred d_raw root_trans:", d_raw[0, :3])
    print("gt d_raw root_trans:", gt_d_raw[0, :3])
    print("gt hdf5 root_trans:", gt["root_trans"][0])
    print("pred FK root (neutral betas):", pred_kp[0, 0])
    print("gt FK from gt d_raw:", gt_fk[0, 0])
    print("gt hdf5 keypoints[0] (NOT translation):", gt_kp_hdf5[0, 0])
    print("gt hdf5 root quat xyz:", gt_d_raw[0, 4:7])

    print("\n=== L2 errors (all frames mean unless noted) ===")
    root_trans_l2 = np.linalg.norm(d_raw[:, :3] - gt["root_trans"], axis=1)
    root_fk_l2 = np.linalg.norm(pred_kp[:, 0] - gt_fk[:, 0], axis=1)
    skel_fk_l2 = np.linalg.norm(pred_kp - gt_fk, axis=-1).mean(axis=1)
    skel_hdf5_l2 = np.linalg.norm(pred_kp - gt_kp_hdf5, axis=-1).mean(axis=1)
    gt_fk_vs_hdf5 = np.linalg.norm(gt_fk - gt_kp_hdf5, axis=-1).mean(axis=1)

    print("pred root_trans vs gt root_trans:", root_trans_l2.mean())
    print("pred FK root vs gt FK root:", root_fk_l2.mean())
    print("pred FK vs gt FK mean joint L2:", skel_fk_l2.mean())
    print("pred FK vs hdf5 keypoints mean joint L2 (legacy/wrong):", skel_hdf5_l2.mean())
    print("gt FK vs hdf5 keypoints mean joint L2 (proves hdf5 unusable):", gt_fk_vs_hdf5.mean())

    print("\n=== frame 0 detail ===")
    print("pred FK vs gt FK mean joint L2:", skel_fk_l2[0])
    print("gt FK vs hdf5 keypoints mean joint L2:", gt_fk_vs_hdf5[0])
    print("hdf5 keypoints[0] == root quat xyz:", np.allclose(gt_kp_hdf5[0, 0], gt_d_raw[0, 4:7]))

    out = {
        "root_trans_l2_mean": float(root_trans_l2.mean()),
        "root_fk_l2_mean": float(root_fk_l2.mean()),
        "skeleton_l2_pred_fk_vs_gt_fk_mean": float(skel_fk_l2.mean()),
        "skeleton_l2_pred_fk_vs_gt_kp_mean": float(skel_hdf5_l2.mean()),
        "skeleton_l2_gt_fk_vs_gt_kp_mean": float(gt_fk_vs_hdf5.mean()),
        "pred_fk_root_vs_d_raw_root_mean": float(
            np.linalg.norm(pred_kp[:, 0] - d_raw[:, :3], axis=1).mean()
        ),
        "hdf5_keypoints_joint0_is_root_quat_xyz": bool(np.allclose(gt_kp_hdf5[0, 0], gt_d_raw[0, 4:7])),
    }
    out_path = ROOT / "experiments/phi0_full/debug_viz_alignment.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("\nWrote", out_path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

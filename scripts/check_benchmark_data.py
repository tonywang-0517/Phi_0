#!/usr/bin/env python3
"""Report LIBERO/CALVIN data + eval readiness under Phi_0/data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phi0.benchmark.paths import (  # noqa: E402
    CALVIN_RLDS_ROOT,
    LIBERO_RLDS_ROOT,
    LIBERO_SUITES,
    calvin_eval_root,
    calvin_models_dir,
    calvin_validation_dir,
    libero_rlds_dir,
)


def _count_glob(root: Path, pattern: str) -> int:
    return len(list(root.glob(pattern))) if root.is_dir() else 0


def check_libero() -> dict:
    suites = {}
    for suite in LIBERO_SUITES:
        d = libero_rlds_dir(suite)
        base = suite.replace("_no_noops", "")
        patterns = [f"{base}-train.tfrecord-*", "liber_o10-train.tfrecord-*"]
        n_shards = 0
        for pat in patterns:
            n_shards = max(n_shards, _count_glob(d, pat))
        meta_ok = (d / "dataset_info.json").is_file() and (d / "features.json").is_file()
        suites[suite] = {
            "path": str(d),
            "train_shards": n_shards,
            "metadata_ok": meta_ok,
            "ready_for_bridge_train": meta_ok and n_shards > 0,
        }
    libero_pkg = False
    try:
        import libero  # noqa: F401

        libero_pkg = True
    except Exception:
        pass
    return {
        "rlds_root": str(LIBERO_RLDS_ROOT),
        "suites": suites,
        "libero_package_installed": libero_pkg,
        "ready_for_eval": libero_pkg and any(s["train_shards"] > 0 for s in suites.values()),
    }


def check_calvin() -> dict:
    n_train = _count_glob(CALVIN_RLDS_ROOT, "calvin_abc-train.tfrecord-*")
    n_valid = _count_glob(CALVIN_RLDS_ROOT, "calvin_abc-validation.tfrecord-*")
    val_dir = calvin_validation_dir()
    models = calvin_models_dir()
    agent_ok = env_ok = False
    try:
        import calvin_agent  # noqa: F401

        agent_ok = True
    except Exception:
        pass
    try:
        import calvin_env  # noqa: F401

        env_ok = True
    except Exception:
        pass
    return {
        "rlds_root": str(CALVIN_RLDS_ROOT),
        "rlds_train_shards": n_train,
        "rlds_valid_shards": n_valid,
        "rlds_train_complete": n_train >= 512,
        "ready_for_bridge_train": n_train > 0,
        "eval_root": str(calvin_eval_root()),
        "validation_dir": str(val_dir),
        "validation_ready": val_dir.is_dir() and any(val_dir.glob("episode_*.npz")),
        "calvin_models": str(models),
        "calvin_models_ready": models.is_dir(),
        "calvin_agent_installed": agent_ok,
        "calvin_env_installed": env_ok,
        "ready_for_eval": (
            agent_ok and env_ok and models.is_dir() and val_dir.is_dir() and any(val_dir.glob("episode_*.npz"))
        ),
    }


def main() -> None:
    report = {
        "libero": check_libero(),
        "calvin": check_calvin(),
        "gaps": [],
    }
    libero_train_ok = any(s["ready_for_bridge_train"] for s in report["libero"]["suites"].values())
    if not libero_train_ok:
        report["gaps"].append("LIBERO RLDS missing or incomplete under data/libero/modified_libero_rlds/")
    if not report["libero"]["ready_for_eval"]:
        report["gaps"].append("LIBERO eval: pip install -e third_party/LIBERO")
    if not report["calvin"]["ready_for_bridge_train"]:
        report["gaps"].append("CALVIN RLDS train shards missing under data/calvin/calvin_abc_rlds/")
    if not report["calvin"]["rlds_train_complete"]:
        report["gaps"].append(f"CALVIN RLDS partial: {report['calvin']['rlds_train_shards']}/512 train shards")
    if not report["calvin"]["validation_ready"]:
        report["gaps"].append(
            "CALVIN 仿真 eval 还需 dataset/task_ABC_D/validation（npz）；"
            "当前 data/calvin/calvin_abc_rlds 仅用于 bridge 训练"
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

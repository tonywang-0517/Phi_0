#!/usr/bin/env python3
"""Download minimal Xperience + EgoDex samples for Phi_0 smoke training."""

from __future__ import annotations

import argparse
import os
import subprocess
import zipfile
from pathlib import Path

import requests
from huggingface_hub import hf_hub_url

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("PHI0_WORKSPACE", "/mnt/data2/wpy/workspace"))
HF_BASE = "https://huggingface.co/datasets/ropedia-ai/xperience-10m-sample/resolve/main"
XPERIENCE_DIR = WORKSPACE / "Isaac-GR00T/demo_data/xperience-10m-sample"
EGODEX_DIR = WORKSPACE / "Isaac-GR00T/demo_data/egodex/test/add_remove_lid"
EGODEX_ZIP_ENTRIES = ("test/add_remove_lid/0.hdf5", "test/add_remove_lid/0.mp4")

# Minimum for ego-video smoke test: annotation + one rectified stereo stream (~22 MB)
DEFAULT_XPERIENCE_FILES = {
    "annotation.hdf5": "annotation.hdf5",
    "stereo_left.mp4": "stereo_left.mp4",
}


def _download(url: str, dest: Path, resume: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-fsSL", "-L", "-o", str(dest), url]
    if resume and dest.exists() and dest.stat().st_size > 0:
        cmd.insert(1, "-C")
        cmd.insert(2, "-")
        print(f"Resuming {dest} from {dest.stat().st_size} bytes")
    subprocess.run(cmd, check=True)
    print(f"Downloaded {dest} ({dest.stat().st_size} bytes)")


class _HTTPRangeReader:
    """Seekable reader for HuggingFace LFS files (Accept-Ranges: bytes)."""

    def __init__(self, url: str, size: int, headers: dict[str, str]):
        self.url = url
        self.size = size
        self.headers = headers
        self.pos = 0

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def seekable(self) -> bool:
        return True

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        if n < 0:
            n = self.size - self.pos
        end = min(self.pos + n - 1, self.size - 1)
        headers = dict(self.headers)
        headers["Range"] = f"bytes={self.pos}-{end}"
        resp = requests.get(self.url, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.content
        self.pos += len(data)
        return data


def download_egodex_episode0(out_dir: Path, token: str | None = None) -> None:
    """Extract episode 0 from test.zip via HTTP range (~2 MB), not the full 17 GB archive."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hdf5 = out_dir / "0.hdf5"
    mp4 = out_dir / "0.mp4"
    if hdf5.exists() and mp4.exists() and hdf5.stat().st_size > 0 and mp4.stat().st_size > 0:
        print(f"Skip EgoDex episode 0 ({hdf5.stat().st_size}+{mp4.stat().st_size} bytes)")
        return

    url = hf_hub_url("zhenyuxie-zhzh/egodex", "test.zip", repo_type="dataset")
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    head = requests.head(url, headers=headers, allow_redirects=True, timeout=30)
    head.raise_for_status()
    size = int(head.headers.get("content-length", 0))
    if head.headers.get("accept-ranges") != "bytes":
        raise RuntimeError("EgoDex test.zip does not support HTTP range requests")

    reader = _HTTPRangeReader(url, size, headers)
    with zipfile.ZipFile(reader) as zf:
        for entry in EGODEX_ZIP_ENTRIES:
            info = zf.getinfo(entry)
            dest = out_dir / Path(entry).name
            print(f"Extract {entry} -> {dest} ({info.file_size} bytes)")
            dest.write_bytes(zf.read(entry))


def download_xperience(out_dir: Path, files: dict[str, str] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = files or DEFAULT_XPERIENCE_FILES
    for local_name, remote_name in files.items():
        dest = out_dir / local_name
        if dest.exists() and dest.stat().st_size > 0:
            if local_name == "annotation.hdf5" and dest.stat().st_size < 1_900_000_000:
                url = f"{HF_BASE}/{remote_name}"
                _download(url, dest, resume=True)
                continue
            print(f"Skip {dest} ({dest.stat().st_size} bytes)")
            continue
        url = f"{HF_BASE}/{remote_name}"
        _download(url, dest)


def _load_hf_token() -> str | None:
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip() or None
    return os.environ.get("HF_TOKEN")


def parse_args():
    p = argparse.ArgumentParser(description="Download Phi_0 demo samples")
    p.add_argument("--xperience-dir", type=str, default=str(XPERIENCE_DIR))
    p.add_argument("--egodex-dir", type=str, default=str(EGODEX_DIR))
    p.add_argument(
        "--include-fisheye",
        action="store_true",
        help="Also download fisheye_cam0.mp4 (~86 MB) instead of stereo only",
    )
    p.add_argument("--skip-egodex", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    xperience_dir = Path(args.xperience_dir)

    files = dict(DEFAULT_XPERIENCE_FILES)
    if args.include_fisheye:
        files["fisheye_cam0.mp4"] = "fisheye_cam0.mp4"

    egodex_dir = Path(args.egodex_dir)
    if not args.skip_egodex:
        download_egodex_episode0(egodex_dir, token=_load_hf_token())
        hdf5 = egodex_dir / "0.hdf5"
        if hdf5.exists():
            out = hdf5.with_name(f"{hdf5.stem}_smplh.hdf5")
            if not out.exists():
                env = os.environ.copy()
                env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
                subprocess.run(
                    ["python", str(ROOT / "scripts/preprocess_egodex_smplh.py"), str(hdf5)],
                    check=True,
                    cwd=str(ROOT),
                    env=env,
                )

    download_xperience(xperience_dir, files)

    print("\nDownloaded assets for Phi_0:")
    for name in sorted(files):
        p = xperience_dir / name
        status = f"{p.stat().st_size} bytes" if p.exists() else "MISSING"
        print(f"  - {name}: {status}")
    for name in ("0.hdf5", "0.mp4"):
        p = egodex_dir / name
        status = f"{p.stat().st_size} bytes" if p.exists() else "MISSING"
        print(f"  - egodex/{name}: {status}")
    print("\nNot downloaded (optional/large):")
    print("  - Cosmos-Predict2.5-2B weights (~21 GB) — only needed for full GPU training")
    print("  - egodex test.zip full archive (~17 GB) — episode 0 extracted via HTTP range")
    print("  - fisheye_cam1-3, stereo_right (~5.5 GB total)")


if __name__ == "__main__":
    main()

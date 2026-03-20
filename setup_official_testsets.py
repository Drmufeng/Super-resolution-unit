#!/usr/bin/env python3
"""Install official Set14/B100/Urban100 test sets for both workspaces."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
WORKSPACES = [ROOT / "sr30_workspace", ROOT / "srcnn"]
CACHE_DIR = ROOT / "_benchmark_cache"
DEFAULT_URL = "https://cv.snu.ac.kr/research/EDSR/benchmark.tar"
DATASETS = ["Set14", "B100", "Urban100"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_archive(url: str, out_path: Path) -> Path:
    if out_path.is_file() and out_path.stat().st_size > 0:
        return out_path
    print(f"[Info] downloading official benchmark: {url}")
    urllib.request.urlretrieve(url, out_path)
    return out_path


def extract_archive(archive_path: Path, out_dir: Path) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r") as tf:
        tf.extractall(path=out_dir)
    return out_dir


def find_dataset_dir(extracted_root: Path, name: str) -> Path:
    lower = name.lower()
    for p in extracted_root.rglob("*"):
        if p.is_dir() and p.name.lower() == lower:
            return p
    raise FileNotFoundError(f"dataset folder not found: {name}")


def pick_subdir(base: Path, candidates: list[str]) -> Path:
    for rel in candidates:
        p = base / rel
        if p.is_dir():
            return p
    raise FileNotFoundError(f"none of candidate dirs found under {base}: {candidates}")


def list_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def verify_readable(folder: Path) -> tuple[int, int]:
    total = 0
    bad = 0
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            total += 1
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                bad += 1
    return total, bad


def copy_tree_images(src_dir: Path, dst_dir: Path) -> int:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for p in list_images(src_dir):
        shutil.copy2(p, dst_dir / p.name)
        copied += 1
    return copied


def install_dataset_for_workspace(ws: Path, extracted_root: Path) -> None:
    print(f"[Info] installing official testsets into {ws}")
    test_root = ws / "test"
    for ds in DATASETS:
        src_ds = find_dataset_dir(extracted_root, ds)
        hr_src = pick_subdir(src_ds, ["HR", "hr"])
        lr2_src = pick_subdir(src_ds, ["LR_bicubic/X2", "LR_bicubic/x2", "lr_bicubic/X2", "lr_bicubic/x2", "LR_bicubic"])
        lr4_src = pick_subdir(src_ds, ["LR_bicubic/X4", "LR_bicubic/x4", "lr_bicubic/X4", "lr_bicubic/x4", "LR_bicubic"])

        dst_hr = test_root / ds / "HR"
        dst_lr2 = test_root / ds / "LR_bicubic"
        dst_lr4 = test_root / ds / "LR_bicubic_x4"

        n_hr = copy_tree_images(hr_src, dst_hr)
        n_lr2 = copy_tree_images(lr2_src, dst_lr2)
        n_lr4 = copy_tree_images(lr4_src, dst_lr4)

        print(f"[Done] {ws.name}/{ds}: HR={n_hr}, LRx2={n_lr2}, LRx4={n_lr4}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="", help="Path to benchmark.tar")
    parser.add_argument("--url", default=DEFAULT_URL, help="Download URL for benchmark.tar")
    args = parser.parse_args()

    ensure_cache()
    archive = Path(args.archive) if args.archive else CACHE_DIR / "benchmark.tar"
    if not archive.is_absolute():
        archive = (ROOT / archive).resolve()
    if not archive.exists():
        download_archive(args.url, archive)

    extracted = extract_archive(archive, CACHE_DIR / "benchmark_extracted")

    for ws in WORKSPACES:
        install_dataset_for_workspace(ws, extracted)

    for ws in WORKSPACES:
        total, bad = verify_readable(ws / "test")
        print(f"[Verify] {ws.name}: total={total}, bad={bad}")
        if bad > 0:
            raise RuntimeError(f"{ws} still has unreadable test images")

    print("[Done] official testsets installed for both workspaces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

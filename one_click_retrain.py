#!/usr/bin/env python3
"""One-click recovery retraining pipeline for GAN workspace."""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SR30 = ROOT / "sr30_workspace"
SRCNN = ROOT / "srcnn"


def run(cmd: list[str], cwd: Path | None = None) -> int:
    print("[Cmd]", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None)


def clean_corrupt_checkpoints() -> int:
    roots = [
        SRCNN / "checkpoints",
        SRCNN / "checkpoints_backup",
        SR30 / "checkpoints",
        SR30 / "checkpoints_backup",
    ]
    deleted = 0
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.pth"):
            p.unlink(missing_ok=True)
            deleted += 1
    print(f"[Info] deleted checkpoint files: {deleted}")
    return deleted


def _has_images(dir_path: Path) -> bool:
    if not dir_path.exists():
        return False
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        if any(dir_path.rglob(ext)):
            return True
    return False


def extract_if_needed(base: Path) -> None:
    df2k = base / "data" / "df2k"
    if not df2k.exists():
        return

    train_zip = df2k / "DIV2K_train_HR.zip"
    valid_zip = df2k / "DIV2K_valid_HR.zip"

    train_dir = df2k / "DIV2K_train_HR"
    valid_dir = df2k / "DIV2K_valid_HR"

    if train_zip.exists() and not _has_images(train_dir):
        print(f"[Info] extracting {train_zip}")
        with zipfile.ZipFile(train_zip, "r") as zf:
            zf.extractall(df2k)

    if valid_zip.exists() and not _has_images(valid_dir):
        print(f"[Info] extracting {valid_zip}")
        with zipfile.ZipFile(valid_zip, "r") as zf:
            zf.extractall(df2k)


def main() -> int:
    parser = argparse.ArgumentParser(description="One-click retraining for both independent pipelines")
    parser.add_argument("--only", choices=["all", "x2", "x4"], default="all")
    parser.add_argument("--pipeline", choices=["all", "sr30", "srcnn"], default="all")
    parser.add_argument("--skip-clean-pth", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    if not args.skip_clean_pth:
        clean_corrupt_checkpoints()

    if not args.skip_extract:
        extract_if_needed(SR30)
        extract_if_needed(SRCNN)

    if not args.skip_train:
        if args.pipeline in {"all", "sr30"}:
            print("[Info] training pipeline=sr30_workspace (EDSR)")
            code = run([sys.executable, str(SR30 / "run_full_from_scratch.py"), "--only", args.only], cwd=SR30)
            if code != 0:
                return code
        if args.pipeline in {"all", "srcnn"}:
            print("[Info] training pipeline=srcnn (SRResNet/SRGAN)")
            code = run([sys.executable, str(SRCNN / "run_full_from_scratch.py"), "--only", args.only], cwd=SRCNN)
            if code != 0:
                return code

    print("[Done] one-click pipeline completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

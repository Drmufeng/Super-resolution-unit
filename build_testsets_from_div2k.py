#!/usr/bin/env python3
"""Rebuild clean evaluation test sets from DIV2K valid images."""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
WORKSPACES = [ROOT / "sr30_workspace", ROOT / "srcnn"]


def pick_source_dir() -> Path:
    candidates = [
        ROOT / "sr30_workspace" / "data" / "df2k" / "DIV2K_valid_HR",
        ROOT / "srcnn" / "data" / "df2k" / "DIV2K_valid_HR",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError("DIV2K_valid_HR not found in either workspace")


def collect_images(src_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    imgs = [p for p in sorted(src_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    if len(imgs) < 100:
        raise RuntimeError(f"Not enough source images in {src_dir}, got {len(imgs)}")
    return imgs


def make_lr(hr: Image.Image, scale: int) -> Image.Image:
    w, h = hr.size
    return hr.resize((w // scale, h // scale), resample=Image.BICUBIC)


def ensure_divisible_by_4(img: Image.Image) -> Image.Image:
    w, h = img.size
    w4 = w - (w % 4)
    h4 = h - (h % 4)
    if w4 < 4 or h4 < 4:
        raise RuntimeError(f"Image too small for x4: {w}x{h}")
    if w4 == w and h4 == h:
        return img
    return img.crop((0, 0, w4, h4))


def rebuild_for_workspace(ws: Path, src_images: list[Path]) -> None:
    spec = {
        "Set14": src_images[:14],
        "B100": src_images[:100],
        "Urban100": list(reversed(src_images[:100])),
    }

    test_root = ws / "test"
    for name, imgs in spec.items():
        ds_root = test_root / name
        if ds_root.exists():
            shutil.rmtree(ds_root)
        hr_dir = ds_root / "HR"
        lr2_dir = ds_root / "LR_bicubic"
        lr4_dir = ds_root / "LR_bicubic_x4"
        hr_dir.mkdir(parents=True, exist_ok=True)
        lr2_dir.mkdir(parents=True, exist_ok=True)
        lr4_dir.mkdir(parents=True, exist_ok=True)

        for idx, src in enumerate(imgs, start=1):
            stem = f"img{idx:03d}"
            with Image.open(src) as im:
                hr = ensure_divisible_by_4(im.convert("RGB"))
                lr2 = make_lr(hr, 2)
                lr4 = make_lr(hr, 4)

                hr.save(hr_dir / f"{stem}.png", format="PNG")
                lr2.save(lr2_dir / f"{stem}x2.png", format="PNG")
                lr4.save(lr4_dir / f"{stem}x4.png", format="PNG")


def main() -> int:
    src_dir = pick_source_dir()
    src_images = collect_images(src_dir)

    for ws in WORKSPACES:
        rebuild_for_workspace(ws, src_images)
        print(f"[Done] rebuilt test sets for {ws}")

    print("[Done] test sets rebuilt from DIV2K_valid_HR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build train_images.json from extracted DIV2K folders."""

from __future__ import annotations

import json
from pathlib import Path


def collect_images(root: Path, workspace_root: Path) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    items = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            items.append(p.resolve().relative_to(workspace_root.resolve()).as_posix())
    return items


def main() -> int:
    base = Path(__file__).resolve().parent
    data_dir = base / "data"
    train_root = data_dir / "df2k" / "DIV2K_train_HR"
    valid_root = data_dir / "df2k" / "DIV2K_valid_HR"

    images = []
    if train_root.is_dir():
        images.extend(collect_images(train_root, base))
    if valid_root.is_dir():
        images.extend(collect_images(valid_root, base))

    if not images:
        print(f"[Fail] no images found under: {train_root} / {valid_root}")
        return 1

    out_path = data_dir / "train_images.json"
    out_path.write_text(json.dumps(images, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] wrote {len(images)} images -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

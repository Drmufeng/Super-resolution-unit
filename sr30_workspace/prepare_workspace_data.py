"""准备工作区数据目录（复制 train index + 测试集，可选复制DF2K）。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_if_missing(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy-df2k", action="store_true")
    args = parser.parse_args()

    ws = Path(__file__).resolve().parent
    srcnn = Path(__file__).resolve().parents[1] / "srcnn"

    copy_if_missing(srcnn / "data" / "train_images.json", ws / "data" / "train_images.json")
    for ds in ["Set14", "B100", "Urban100"]:
        copy_if_missing(srcnn / "test" / ds, ws / "test" / ds)
    if args.copy_df2k:
        copy_if_missing(srcnn / "data" / "df2k", ws / "data" / "df2k")

    print("[Done] data preparation completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run full SRCNN pipeline from scratch: x2/x4 pretrain + finetune."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> int:
    print("[Cmd]", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(cwd))


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["all", "x2", "x4"], default="all")
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    py = sys.executable

    if run([py, str(root / "prepare_data_index.py")], root) != 0:
        return 1

    if not args.no_clean:
        for p in [root / "checkpoints", root / "checkpoints_backup", root / "runs", root / "outputs"]:
            clean_dir(p)
        for p in [
            root / "checkpoints" / "srresnet_x2",
            root / "checkpoints" / "x2",
            root / "checkpoints" / "srresnet_x4",
            root / "checkpoints" / "x4",
            root / "checkpoints_backup" / "srresnet_x2",
            root / "checkpoints_backup" / "x2",
            root / "checkpoints_backup" / "srresnet_x4",
            root / "checkpoints_backup" / "x4",
            root / "runs" / "srresnet_x2",
            root / "runs" / "x2",
            root / "runs" / "srresnet_x4",
            root / "runs" / "x4",
            root / "outputs" / "x2",
            root / "outputs" / "x4",
        ]:
            p.mkdir(parents=True, exist_ok=True)

    stages: list[list[str]] = []
    if args.only in {"all", "x2"}:
        stages += [
            [py, str(root / "train_srcnn.py"), "--config", str(root / "configs" / "pretrain_x2_srresnet.json")],
            [py, str(root / "train_srcnn.py"), "--config", str(root / "configs" / "finetune_x2_srgan.json")],
        ]
    if args.only in {"all", "x4"}:
        stages += [
            [py, str(root / "train_srcnn.py"), "--config", str(root / "configs" / "pretrain_x4_srresnet.json")],
            [py, str(root / "train_srcnn.py"), "--config", str(root / "configs" / "finetune_x4_srgan.json")],
        ]

    for cmd in stages:
        code = run(cmd, root)
        if code != 0:
            return code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

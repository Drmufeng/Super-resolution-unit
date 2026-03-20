"""从零开始全流程：清理后跑 x2/x4 预训练+精调。"""

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

    ws = Path(__file__).resolve().parent
    py = sys.executable

    if run([py, str(ws / "prepare_data_index.py")], ws) != 0:
        return 1

    if not args.no_clean:
        for p in [ws / "checkpoints", ws / "checkpoints_backup", ws / "runs", ws / "outputs"]:
            clean_dir(p)
        for p in [
            ws / "checkpoints" / "srresnet_x2", ws / "checkpoints" / "x2", ws / "checkpoints" / "srresnet_x4", ws / "checkpoints" / "x4",
            ws / "checkpoints_backup" / "srresnet_x2", ws / "checkpoints_backup" / "x2", ws / "checkpoints_backup" / "srresnet_x4", ws / "checkpoints_backup" / "x4",
            ws / "runs" / "srresnet_x2", ws / "runs" / "x2", ws / "runs" / "srresnet_x4", ws / "runs" / "x4",
            ws / "outputs" / "x2", ws / "outputs" / "x4",
        ]:
            p.mkdir(parents=True, exist_ok=True)

    stages: list[list[str]] = []
    if args.only in {"all", "x2"}:
        stages += [[py, str(ws / "train_psnr_edsr.py"), "--config", str(ws / "configs" / "pretrain_x2_edsr.json")], [py, str(ws / "train_psnr_edsr.py"), "--config", str(ws / "configs" / "finetune_x2_edsr.json")]]
    if args.only in {"all", "x4"}:
        stages += [[py, str(ws / "train_psnr_edsr.py"), "--config", str(ws / "configs" / "pretrain_x4_edsr.json")], [py, str(ws / "train_psnr_edsr.py"), "--config", str(ws / "configs" / "finetune_x4_edsr.json")]]

    for cmd in stages:
        code = run(cmd, ws)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

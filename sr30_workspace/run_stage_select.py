"""按指定 checkpoint 版本运行阶段。"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


STAGE_TO_CONFIG = {
    "pretrain_x2": "configs/pretrain_x2_edsr.json",
    "finetune_x2": "configs/finetune_x2_edsr.json",
    "pretrain_x4": "configs/pretrain_x4_edsr.json",
    "finetune_x4": "configs/finetune_x4_edsr.json",
}
STAGE_TO_CKPT_DIR = {
    "pretrain_x2": "checkpoints/srresnet_x2",
    "finetune_x2": "checkpoints/x2",
    "pretrain_x4": "checkpoints/srresnet_x4",
    "finetune_x4": "checkpoints/x4",
}
STAGE_TO_PREFIX = {
    "pretrain_x2": "edsr_x2_",
    "finetune_x2": "edsr_x2_",
    "pretrain_x4": "edsr_x4_",
    "finetune_x4": "edsr_x4_",
}


def epoch_from_name(path: Path) -> int:
    m = re.search(r"_e(\d+)\.pth$", path.name)
    return int(m.group(1)) if m else -1


def pick_checkpoint(ws: Path, stage: str, select: str, epoch: int, ckpt_path: str) -> Path | None:
    if select == "path":
        p = Path(ckpt_path)
        return p if p.is_absolute() else (ws / p).resolve()

    ckpt_dir = (ws / STAGE_TO_CKPT_DIR[stage]).resolve()
    prefix = STAGE_TO_PREFIX[stage]
    if not ckpt_dir.exists():
        return None

    if select == "best":
        p = ckpt_dir / f"{prefix}best.pth"
        return p if p.is_file() else None

    cands = sorted(ckpt_dir.glob(f"{prefix}e*.pth"), key=epoch_from_name)
    if not cands:
        return None
    if select == "latest":
        return cands[-1]
    if select == "epoch":
        p = ckpt_dir / f"{prefix}e{epoch}.pth"
        return p if p.is_file() else None
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=list(STAGE_TO_CONFIG.keys()), required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--select", choices=["fresh", "latest", "best", "epoch", "path"], default="latest")
    parser.add_argument("--source", choices=["resume", "init"], default="resume")
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    ws = Path(__file__).resolve().parent
    cfg_path = (ws / STAGE_TO_CONFIG[args.stage]).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    target = None
    if args.select != "fresh":
        target = pick_checkpoint(ws, args.stage, args.select, args.epoch, args.checkpoint)
        if target is None or not target.is_file():
            print("[Fail] checkpoint not found")
            return 1

    train = cfg.setdefault("train", {})
    if args.source == "resume":
        train["resume_checkpoint"] = "" if target is None else str(target).replace("\\", "/")
    else:
        train["resume_checkpoint"] = ""
        train["init_checkpoint"] = "" if target is None else str(target).replace("\\", "/")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(cfg, tmp, ensure_ascii=False, indent=2)
        tmp_cfg = tmp.name

    try:
        cmd = [args.python, str(ws / "train_psnr_edsr.py"), "--config", tmp_cfg]
        print("[Cmd]", " ".join(cmd), flush=True)
        return subprocess.call(cmd, cwd=str(ws))
    finally:
        Path(tmp_cfg).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

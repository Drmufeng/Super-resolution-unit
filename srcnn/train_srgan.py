"""SRGAN finetune entry (compatibility wrapper)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune_x2_srgan.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg = Path(args.config)
    if not cfg.is_absolute():
        cfg = (root / cfg).resolve()

    cmd = [sys.executable, str(root / "train_srcnn.py"), "--config", str(cfg)]
    print("[Cmd]", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(root))


if __name__ == "__main__":
    raise SystemExit(main())

"""Run SRCNN line x4 finetune."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    cmd = [sys.executable, str(root / "train_srcnn.py"), "--config", str(root / "configs" / "finetune_x4_srgan.json")]
    print("[Cmd]", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(root))


if __name__ == "__main__":
    raise SystemExit(main())

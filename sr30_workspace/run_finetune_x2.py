"""只运行 x2 精调。"""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    ws = Path(__file__).resolve().parent
    cmd = [sys.executable, str(ws / "train_psnr_edsr.py"), "--config", str(ws / "configs" / "finetune_x2_edsr.json")]
    print("[Cmd]", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ws))


if __name__ == "__main__":
    raise SystemExit(main())

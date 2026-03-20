"""x4 预训练 + 精调串行入口。"""

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> int:
    print("[Cmd]", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(cwd))


def main() -> int:
    ws = Path(__file__).resolve().parent
    c1 = run([sys.executable, str(ws / "train_psnr_edsr.py"), "--config", str(ws / "configs" / "pretrain_x4_edsr.json")], ws)
    if c1 != 0:
        return c1
    return run([sys.executable, str(ws / "train_psnr_edsr.py"), "--config", str(ws / "configs" / "finetune_x4_edsr.json")], ws)


if __name__ == "__main__":
    raise SystemExit(main())

"""运行 EDSR 全量评测。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_eval(ws: Path, cfg: Path) -> int:
    cmd = [sys.executable, str(ws / "eval_psnr_edsr.py"), "--config", str(cfg)]
    print("[Cmd]", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ws))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "x2", "x4"], default="all")
    args = parser.parse_args()

    ws = Path(__file__).resolve().parent
    cfgs = []
    if args.mode in {"all", "x2"}:
        cfgs += [ws / "configs" / "eval_x2_set14_edsr.json", ws / "configs" / "eval_x2_b100_edsr.json", ws / "configs" / "eval_x2_urban100_edsr.json"]
    if args.mode in {"all", "x4"}:
        cfgs += [ws / "configs" / "eval_x4_set14_edsr.json", ws / "configs" / "eval_x4_b100_edsr.json", ws / "configs" / "eval_x4_urban100_edsr.json"]

    for c in cfgs:
        if not c.is_file():
            print(f"[Skip] missing config: {c}")
            continue
        code = run_eval(ws, c)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

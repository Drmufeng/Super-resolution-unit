"""详细日志流水：分阶段输出日志与summary。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_stage(name: str, cmd: list[str], cwd: Path, log_path: Path) -> dict:
    ensure_dir(log_path.parent)
    started = time.time()
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] stage={name}\n")
        f.write("cmd=" + " ".join(cmd) + "\n")
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=False)
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace")
            print(line, end="")
            f.write(line)
        code = proc.wait()
    return {"stage": name, "returncode": code, "seconds": round(time.time() - started, 2), "log": str(log_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["all", "x2", "x4"], default="all")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    ws = Path(__file__).resolve().parent
    py = args.python
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ws / "runs" / "pipeline" / stamp
    ensure_dir(log_dir)

    stages: list[tuple[str, list[str]]] = []
    if args.only in {"all", "x2"}:
        if not args.skip_pretrain:
            stages.append(("pretrain_x2", [py, str(ws / "train_psnr_edsr.py"), "--config", str(ws / "configs" / "pretrain_x2_edsr.json")]))
        stages.append(("finetune_x2", [py, str(ws / "train_psnr_edsr.py"), "--config", str(ws / "configs" / "finetune_x2_edsr.json")]))
    if args.only in {"all", "x4"}:
        if not args.skip_pretrain:
            stages.append(("pretrain_x4", [py, str(ws / "train_psnr_edsr.py"), "--config", str(ws / "configs" / "pretrain_x4_edsr.json")]))
        stages.append(("finetune_x4", [py, str(ws / "train_psnr_edsr.py"), "--config", str(ws / "configs" / "finetune_x4_edsr.json")]))

    summary = {"started_at": datetime.now().isoformat(timespec="seconds"), "workspace": str(ws), "python": py, "stages": []}
    for name, cmd in stages:
        print(f"\n[Stage] {name}")
        r = run_stage(name, cmd, ws, log_dir / f"{name}.log")
        summary["stages"].append(r)
        if r["returncode"] != 0:
            summary["status"] = "failed"
            summary["failed_stage"] = name
            break
    else:
        summary["status"] = "ok"

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    out = log_dir / "pipeline_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] summary: {out}")
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

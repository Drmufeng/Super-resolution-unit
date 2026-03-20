"""批量验证 x2/x4 模型并输出汇总。"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path


AVG_PSNR_RE = re.compile(r"\[Done\]\s+avg_psnr=([0-9eE+\-.]+)")
AVG_SSIM_RE = re.compile(r"\[Done\]\s+avg_ssim=([0-9eE+\-.]+)")
COUNT_RE = re.compile(r"\[Done\]\s+count=(\d+)")


def parse_metrics(text: str):
    c = COUNT_RE.search(text)
    p = AVG_PSNR_RE.search(text)
    s = AVG_SSIM_RE.search(text)
    return (int(c.group(1)) if c else None, float(p.group(1)) if p else None, float(s.group(1)) if s else None)


def run_eval(root: Path, config_path: Path) -> dict:
    cmd = [sys.executable, str(root / "eval.py"), "--config", str(config_path)]
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    c, p, s = parse_metrics(out)
    return {"config": str(config_path.relative_to(root)), "status": "ok" if proc.returncode == 0 else "failed", "returncode": proc.returncode, "count": c, "avg_psnr": p, "avg_ssim": s, "output": out}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["x2", "x4", "all"], default="all")
    parser.add_argument("--config", default="")
    parser.add_argument("--report", default="outputs/test_report.csv")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg_dir = root / "configs"
    if args.config:
        cfgs = [cfg_dir / args.config]
    elif args.mode == "x2":
        cfgs = sorted(cfg_dir.glob("eval_x2_*.json"))
    elif args.mode == "x4":
        cfgs = sorted(cfg_dir.glob("eval_x4_*.json"))
    else:
        cfgs = sorted(cfg_dir.glob("eval_*.json"))

    rows = []
    for c in cfgs:
        print(f"[Run] {c.relative_to(root)}", flush=True)
        st = time.time()
        row = run_eval(root, c)
        row["seconds"] = round(time.time() - st, 2)
        rows.append(row)
        if row["status"] == "ok":
            print(f"[Done] {row['config']} count={row['count']} psnr={row['avg_psnr']:.4f} ssim={row['avg_ssim']:.4f} time={row['seconds']:.2f}s", flush=True)
        else:
            print(f"[Fail] {row['config']}", flush=True)

    report = (root / args.report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "status", "returncode", "count", "avg_psnr", "avg_ssim", "seconds"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})

    failed = sum(1 for r in rows if r["status"] != "ok")
    print(f"[Done] Report saved to: {report}")
    print(f"[Done] total={len(rows)}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

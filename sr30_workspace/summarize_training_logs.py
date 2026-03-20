"""训练日志汇总工具（CSV -> Markdown）。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_psnr_row(rows: list[dict]) -> dict | None:
    best = None
    for r in rows:
        if r.get("status") != "ok" or not r.get("avg_psnr"):
            continue
        ps = float(r["avg_psnr"])
        ss = float(r.get("avg_ssim") or -1.0)
        ep = int(r.get("epoch") or -1)
        score = (ps, ss, ep)
        if best is None or score > best[0]:
            best = (score, r)
    return best[1] if best else None


def build_report(ws: Path) -> str:
    lines = ["# 训练日志汇总", "", f"- workspace: `{ws}`", ""]
    targets = [
        ("x2 预训练", ws / "runs" / "srresnet_x2" / "val_metrics.csv"),
        ("x2 精调", ws / "runs" / "x2" / "val_metrics.csv"),
        ("x4 预训练", ws / "runs" / "srresnet_x4" / "val_metrics.csv"),
        ("x4 精调", ws / "runs" / "x4" / "val_metrics.csv"),
    ]
    for title, p in targets:
        rows = read_rows(p)
        lines += [f"## {title}", f"- log: `{p}`", f"- rows: {len(rows)}"]
        b = best_psnr_row(rows)
        if b:
            lines.append(f"- best: epoch={b.get('epoch')} psnr={b.get('avg_psnr')} ssim={b.get('avg_ssim')} checkpoint={b.get('checkpoint')}")
        else:
            lines.append("- best: 无有效记录")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--out", default="docs/训练日志汇总.md")
    args = parser.parse_args()
    ws = Path(args.workspace).resolve()
    out = (ws / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(ws), encoding="utf-8")
    print(f"[Done] report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

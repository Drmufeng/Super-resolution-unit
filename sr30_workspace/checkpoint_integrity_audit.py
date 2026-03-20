#!/usr/bin/env python3
"""Audit .pth checkpoint integrity for recovery triage.

This script scans one or more roots, attempts to load each checkpoint via
torch.load(..., weights_only=False), and writes machine-readable reports.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
from pathlib import Path
from typing import Any

import torch


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parent


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def safe_text(value: Any, max_len: int = 800) -> str:
    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ").strip()
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def classify_loaded_obj(obj: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "object_type": type(obj).__name__,
        "keys": [],
        "resume_ready": False,
        "init_ready": False,
    }

    if isinstance(obj, dict):
        keys = list(obj.keys())
        str_keys = [str(k) for k in keys]
        info["keys"] = str_keys[:80]

        tensor_values = sum(1 for v in obj.values() if isinstance(v, torch.Tensor))
        dict_values = sum(1 for v in obj.values() if isinstance(v, dict))
        info["tensor_values"] = tensor_values
        info["dict_values"] = dict_values

        has_model_sd = "model_state_dict" in obj
        has_state_dict = "state_dict" in obj
        has_optimizer = "optimizer_state_dict" in obj or "optimizer" in obj

        # Resume means likely enough metadata for full training continuation.
        info["resume_ready"] = bool((has_model_sd or has_state_dict) and has_optimizer)

        # Init means we can probably initialize model weights from this file.
        if has_model_sd or has_state_dict:
            info["init_ready"] = True
        elif tensor_values > 0:
            info["init_ready"] = True

    return info


def audit_file(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "modified_utc": dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc).isoformat(),
        "status": "unknown",
        "error_type": "",
        "error_message": "",
        "object_type": "",
        "key_count": 0,
        "keys_preview": "",
        "resume_ready": False,
        "init_ready": False,
    }

    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        details = classify_loaded_obj(obj)
        row["status"] = "ok"
        row["object_type"] = details.get("object_type", "")
        keys = details.get("keys", [])
        row["key_count"] = len(keys)
        row["keys_preview"] = ",".join(keys[:25])
        row["resume_ready"] = bool(details.get("resume_ready", False))
        row["init_ready"] = bool(details.get("init_ready", False))
        del obj
        gc.collect()
    except Exception as exc:  # noqa: BLE001
        row["status"] = "corrupt_or_unloadable"
        row["error_type"] = type(exc).__name__
        row["error_message"] = safe_text(exc)

    return row


def find_checkpoints(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(sorted(root.rglob("*.pth")))
    return files


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "path",
        "size_bytes",
        "modified_utc",
        "status",
        "error_type",
        "error_message",
        "object_type",
        "key_count",
        "keys_preview",
        "resume_ready",
        "init_ready",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: list[dict[str, Any]], roots: list[Path]) -> dict[str, Any]:
    ok_rows = [r for r in rows if r["status"] == "ok"]
    bad_rows = [r for r in rows if r["status"] != "ok"]

    return {
        "generated_utc": utc_now(),
        "roots": [str(p) for p in roots],
        "total": len(rows),
        "ok": len(ok_rows),
        "corrupt_or_unloadable": len(bad_rows),
        "resume_ready": sum(1 for r in ok_rows if r.get("resume_ready")),
        "init_ready": sum(1 for r in ok_rows if r.get("init_ready")),
        "errors_by_type": {
            err: sum(1 for r in bad_rows if r.get("error_type") == err)
            for err in sorted({r.get("error_type", "") for r in bad_rows if r.get("error_type")})
        },
        "largest_files": sorted(rows, key=lambda r: r["size_bytes"], reverse=True)[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit .pth checkpoints integrity")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[str(REPO_ROOT / "srcnn" / "checkpoints"), str(SCRIPT_ROOT / "checkpoints")],
        help="Checkpoint roots to scan",
    )
    parser.add_argument(
        "--out-dir",
        default=str(SCRIPT_ROOT / "reports"),
        help="Output directory for csv/json reports",
    )
    args = parser.parse_args()

    roots = [Path(p) for p in args.roots]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_checkpoints(roots)
    rows = [audit_file(p) for p in files]
    summary = summarize(rows, roots)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"checkpoint_audit_{timestamp}.csv"
    json_path = out_dir / f"checkpoint_audit_{timestamp}.json"
    latest_csv = out_dir / "checkpoint_audit_latest.csv"
    latest_json = out_dir / "checkpoint_audit_latest.json"

    write_csv(rows, csv_path)
    write_csv(rows, latest_csv)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with latest_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

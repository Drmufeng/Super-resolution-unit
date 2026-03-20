"""EDSR 训练脚本：支持 warmup 调度、早停、版本参数记录。"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
SRCNN_ROOT = REPO_ROOT / "srcnn"
if str(SRCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(SRCNN_ROOT))

from src.datasets import SRDataset  # type: ignore  # noqa: E402
from src.utils import AverageMeter  # type: ignore  # noqa: E402

from psnr_models import EDSRGenerator
from schedulers import build_lr_lambda

AVG_PSNR_RE = re.compile(r"\[Done\]\s+avg_psnr=([0-9eE+\-.]+)")
AVG_SSIM_RE = re.compile(r"\[Done\]\s+avg_ssim=([0-9eE+\-.]+)")
COUNT_RE = re.compile(r"\[Done\]\s+count=(\d+)")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_eval_metrics(text: str):
    c = COUNT_RE.search(text)
    p = AVG_PSNR_RE.search(text)
    s = AVG_SSIM_RE.search(text)
    return (int(c.group(1)) if c else None, float(p.group(1)) if p else None, float(s.group(1)) if s else None)


def run_eval_monitor(workspace: Path, eval_cfg_path: Path, checkpoint_path: Path) -> dict:
    if not eval_cfg_path.is_file():
        return {"status": "failed", "count": None, "avg_psnr": None, "avg_ssim": None, "seconds": 0.0, "message": "eval config missing"}

    cfg = load_json(eval_cfg_path)
    cfg.setdefault("eval", {})["checkpoint"] = str(checkpoint_path).replace("\\", "/")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(cfg, tmp, indent=2)
        tmp_path = Path(tmp.name)

    try:
        cmd = [sys.executable, str(workspace / "eval_psnr_edsr.py"), "--config", str(tmp_path)]
        started = time.time()
        proc = subprocess.run(cmd, cwd=str(workspace), capture_output=True, text=True)
        elapsed = round(time.time() - started, 2)
        output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        cnt, ps, ss = parse_eval_metrics(output)
        msg = ""
        if proc.returncode != 0:
            lines = [x.strip() for x in output.splitlines() if x.strip()]
            msg = lines[-1][:300] if lines else f"returncode={proc.returncode}"
        return {"status": "ok" if proc.returncode == 0 else "failed", "count": cnt, "avg_psnr": ps, "avg_ssim": ss, "seconds": elapsed, "message": msg}
    finally:
        tmp_path.unlink(missing_ok=True)


def summarize_model_parameters(model: nn.Module) -> dict:
    total_params = 0
    trainable_params = 0
    tensor_count = 0
    elem_count = 0
    sum_val = 0.0
    sum_sq = 0.0
    sum_abs = 0.0
    min_val = None
    max_val = None

    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            total_params += n
            if p.requires_grad:
                trainable_params += n
            tensor_count += 1
            t = p.detach().float()
            elem_count += n
            sum_val += float(t.sum().item())
            sum_sq += float((t * t).sum().item())
            sum_abs += float(t.abs().sum().item())
            p_min = float(t.min().item())
            p_max = float(t.max().item())
            min_val = p_min if min_val is None else min(min_val, p_min)
            max_val = p_max if max_val is None else max(max_val, p_max)

    mean = sum_val / elem_count if elem_count else 0.0
    var = max(0.0, (sum_sq / elem_count) - (mean * mean)) if elem_count else 0.0
    std = var**0.5
    abs_mean = sum_abs / elem_count if elem_count else 0.0

    return {
        "tensor_count": tensor_count,
        "param_total": total_params,
        "param_trainable": trainable_params,
        "weight_mean": mean,
        "weight_std": std,
        "weight_abs_mean": abs_mean,
        "weight_min": 0.0 if min_val is None else min_val,
        "weight_max": 0.0 if max_val is None else max_val,
    }


def prune_old_checkpoints(save_dir: Path, scale: int, keep_last: int) -> int:
    if keep_last <= 0:
        return 0
    pat = re.compile(rf"edsr_x{scale}_e(\d+)\.pth$")
    cands = []
    for p in save_dir.glob(f"edsr_x{scale}_e*.pth"):
        m = pat.search(p.name)
        if m:
            cands.append((int(m.group(1)), p))
    cands.sort(key=lambda x: x[0])
    removed = 0
    for _, p in cands[:-keep_last]:
        try:
            p.unlink(missing_ok=True)
            removed += 1
        except OSError:
            pass
    return removed


def atomic_save_checkpoint(payload: dict, out_path: Path, mirror_dir: Path | None = None) -> None:
    ensure_dir(out_path.parent)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(payload, str(tmp_path))
    tmp_path.replace(out_path)

    if mirror_dir is not None:
        ensure_dir(mirror_dir)
        mirror_path = mirror_dir / out_path.name
        mirror_tmp = mirror_path.with_suffix(mirror_path.suffix + ".tmp")
        torch.save(payload, str(mirror_tmp))
        mirror_tmp.replace(mirror_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_json(cfg_path)
    ws = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    print(f"[Info] device={device}")

    save_dir = (ws / cfg["save_dir"]).resolve() if not Path(cfg["save_dir"]).is_absolute() else Path(cfg["save_dir"])
    backup_dir_cfg = str(cfg.get("backup_dir", "")).strip()
    backup_dir = None
    if backup_dir_cfg:
        b = Path(backup_dir_cfg)
        backup_dir = (ws / b).resolve() if not b.is_absolute() else b
    log_dir = (ws / cfg["log_dir"]).resolve() if not Path(cfg["log_dir"]).is_absolute() else Path(cfg["log_dir"])
    csv_path = (ws / cfg["csv_log"]).resolve() if not Path(cfg["csv_log"]).is_absolute() else Path(cfg["csv_log"])
    ensure_dir(save_dir)
    ensure_dir(log_dir)
    ensure_dir(csv_path.parent)

    model = EDSRGenerator(
        scale=int(cfg["data"]["scaling_factor"]),
        n_resblocks=int(cfg["model"].get("n_blocks_g", 32)),
        n_feats=int(cfg["model"].get("n_channels_g", 128)),
        res_scale=float(cfg["model"].get("res_scale", 0.1)),
    ).to(device)

    train_cfg = cfg["train"]
    epochs = int(train_cfg["epochs"])
    save_every = int(train_cfg.get("save_every", 20))
    keep_last = int(train_cfg.get("keep_last", 0))
    print_every = int(train_cfg.get("print_every", 20))

    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=build_lr_lambda(epochs, cfg.get("scheduler", {})))

    l1_w = float(cfg.get("loss", {}).get("l1_weight", 0.2))
    mse_w = float(cfg.get("loss", {}).get("mse_weight", 0.8))
    l1_loss = nn.L1Loss().to(device)
    mse_loss = nn.MSELoss().to(device)

    start_epoch = int(train_cfg.get("start_epoch", 1))
    init_ckpt = str(train_cfg.get("init_checkpoint", "")).strip()
    resume_ckpt = str(train_cfg.get("resume_checkpoint", "")).strip()

    if init_ckpt:
        p = Path(init_ckpt)
        if not p.is_absolute():
            p = (ws / p).resolve()
        if p.is_file():
            c = torch.load(str(p), map_location=device, weights_only=False)
            model.load_state_dict(c.get("model", c), strict=False)
            print(f"[Info] init_checkpoint={p}")

    if resume_ckpt:
        p = Path(resume_ckpt)
        if not p.is_absolute():
            p = (ws / p).resolve()
        if p.is_file():
            c = torch.load(str(p), map_location=device, weights_only=False)
            model.load_state_dict(c["model"], strict=False)
            optimizer.load_state_dict(c["optimizer"])
            start_epoch = int(c.get("epoch", 0)) + 1
            print(f"[Info] resumed={p}")

    data_folder = str(cfg["data"]["data_folder"])
    if not Path(data_folder).is_absolute():
        data_folder = str((ws / data_folder).resolve())
    ds = SRDataset(data_folder=data_folder, split="train", crop_size=int(cfg["data"]["crop_size"]), scaling_factor=int(cfg["data"]["scaling_factor"]), lr_img_type="imagenet-norm", hr_img_type="[-1, 1]")
    loader = torch.utils.data.DataLoader(ds, batch_size=int(train_cfg["batch_size"]), shuffle=True, num_workers=int(train_cfg.get("workers", 0)), pin_memory=bool(train_cfg.get("pin_memory", True)) and torch.cuda.is_available())

    writer = SummaryWriter(log_dir=str(log_dir))
    train_csv = csv_path.open("a", newline="", encoding="utf-8")
    train_writer = csv.DictWriter(train_csv, fieldnames=["epoch", "loss", "loss_l1", "loss_mse", "lr", "seconds"])
    if train_csv.tell() == 0:
        train_writer.writeheader()

    mon_cfg = cfg.get("monitor", {})
    mon_path = Path(mon_cfg.get("csv_log", str(log_dir / "val_metrics.csv")))
    if not mon_path.is_absolute():
        mon_path = (ws / mon_path).resolve()
    ensure_dir(mon_path.parent)
    mon_csv = mon_path.open("a", newline="", encoding="utf-8")
    mon_writer = csv.DictWriter(mon_csv, fieldnames=["epoch", "checkpoint", "status", "count", "avg_psnr", "avg_ssim", "seconds", "message"])
    if mon_csv.tell() == 0:
        mon_writer.writeheader()

    meta_path = log_dir / "checkpoint_params.csv"
    meta_csv = meta_path.open("a", newline="", encoding="utf-8")
    meta_writer = csv.DictWriter(meta_csv, fieldnames=["epoch", "checkpoint_type", "checkpoint", "file_mb", "lr", "tensor_count", "param_total", "param_trainable", "weight_mean", "weight_std", "weight_abs_mean", "weight_min", "weight_max", "monitor_psnr", "monitor_ssim", "note"])
    if meta_csv.tell() == 0:
        meta_writer.writeheader()

    eval_cfg = Path(mon_cfg.get("eval_config", ""))
    if not eval_cfg.is_absolute():
        eval_cfg = (ws / eval_cfg).resolve()
    mon_every = int(mon_cfg.get("every", 1))
    target_psnr = float(mon_cfg.get("target_psnr", 999.0))
    patience = int(mon_cfg.get("patience", 60))
    min_delta = float(mon_cfg.get("min_delta", 0.0002))
    best_psnr = -1.0
    no_improve = 0

    for epoch in range(start_epoch, epochs + 1):
        st = time.time()
        model.train()
        meter = AverageMeter()
        meter_l1 = AverageMeter()
        meter_mse = AverageMeter()

        for i, (lr_img, hr_img) in enumerate(loader):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            sr = model(lr_img)
            l1 = l1_loss(sr, hr_img)
            mse = mse_loss(sr, hr_img)
            loss = l1_w * l1 + mse_w * mse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            meter.update(loss.item(), lr_img.size(0))
            meter_l1.update(l1.item(), lr_img.size(0))
            meter_mse.update(mse.item(), lr_img.size(0))
            if i % print_every == 0:
                print(f"epoch={epoch} iter={i} loss={meter.val:.6f} l1={meter_l1.val:.6f} mse={meter_mse.val:.6f}")

        scheduler.step()
        lr_now = float(optimizer.param_groups[0]["lr"])
        sec = time.time() - st
        writer.add_scalar("EDSR/loss", meter.avg, epoch)
        writer.add_scalar("EDSR/lr", lr_now, epoch)
        train_writer.writerow({"epoch": epoch, "loss": f"{meter.avg:.8f}", "loss_l1": f"{meter_l1.avg:.8f}", "loss_mse": f"{meter_mse.avg:.8f}", "lr": f"{lr_now:.10f}", "seconds": f"{sec:.2f}"})
        train_csv.flush()

        should_save = (epoch % save_every == 0 or epoch == epochs)
        should_mon = (epoch % mon_every == 0 or epoch == epochs)
        if not (should_save or should_mon):
            continue

        scale = int(cfg["data"]["scaling_factor"])
        ckpt_path = save_dir / (f"edsr_x{scale}_e{epoch}.pth" if should_save else f"_monitor_tmp_x{scale}.pth")
        ckpt_payload = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
        atomic_save_checkpoint(ckpt_payload, ckpt_path, mirror_dir=backup_dir if should_save else None)
        if should_save:
            removed = prune_old_checkpoints(save_dir, scale, keep_last)
            print(f"[Info] saved={ckpt_path}")
            if backup_dir is not None:
                print(f"[Info] mirrored={backup_dir / ckpt_path.name}")
            if removed:
                print(f"[Info] pruned old checkpoints: {removed} files")

        stats = summarize_model_parameters(model)
        file_mb = ckpt_path.stat().st_size / 1024 / 1024 if ckpt_path.exists() else 0.0
        meta_writer.writerow({"epoch": epoch, "checkpoint_type": "epoch" if should_save else "temp_monitor", "checkpoint": str(ckpt_path), "file_mb": f"{file_mb:.3f}", "lr": f"{lr_now:.10f}", "tensor_count": stats["tensor_count"], "param_total": stats["param_total"], "param_trainable": stats["param_trainable"], "weight_mean": f"{stats['weight_mean']:.8f}", "weight_std": f"{stats['weight_std']:.8f}", "weight_abs_mean": f"{stats['weight_abs_mean']:.8f}", "weight_min": f"{stats['weight_min']:.8f}", "weight_max": f"{stats['weight_max']:.8f}", "monitor_psnr": "", "monitor_ssim": "", "note": "saved checkpoint"})
        meta_csv.flush()

        if should_mon:
            print(f"[Monitor] running eval at epoch={epoch} with checkpoint={ckpt_path}")
        mon = run_eval_monitor(ws, eval_cfg, ckpt_path)
        mon_writer.writerow({"epoch": epoch, "checkpoint": str(ckpt_path), "status": mon["status"], "count": mon["count"], "avg_psnr": "" if mon["avg_psnr"] is None else f"{mon['avg_psnr']:.4f}", "avg_ssim": "" if mon["avg_ssim"] is None else f"{mon['avg_ssim']:.4f}", "seconds": f"{mon['seconds']:.2f}", "message": mon["message"]})
        mon_csv.flush()

        if mon["status"] == "ok" and mon["avg_psnr"] is not None:
            ps = float(mon["avg_psnr"])
            ss = float(mon["avg_ssim"] or 0.0)
            print(f"[Monitor] epoch={epoch} psnr={ps:.4f} ssim={ss:.4f}")

            if ps > best_psnr + min_delta:
                best_psnr = ps
                no_improve = 0
                best_path = save_dir / f"edsr_x{scale}_best.pth"
                best_payload = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_psnr": best_psnr}
                atomic_save_checkpoint(best_payload, best_path, mirror_dir=backup_dir)
                print(f"[Best] epoch={epoch} psnr={ps:.4f} -> {best_path}")
                best_mb = best_path.stat().st_size / 1024 / 1024 if best_path.exists() else 0.0
                meta_writer.writerow({"epoch": epoch, "checkpoint_type": "best", "checkpoint": str(best_path), "file_mb": f"{best_mb:.3f}", "lr": f"{lr_now:.10f}", "tensor_count": stats["tensor_count"], "param_total": stats["param_total"], "param_trainable": stats["param_trainable"], "weight_mean": f"{stats['weight_mean']:.8f}", "weight_std": f"{stats['weight_std']:.8f}", "weight_abs_mean": f"{stats['weight_abs_mean']:.8f}", "weight_min": f"{stats['weight_min']:.8f}", "weight_max": f"{stats['weight_max']:.8f}", "monitor_psnr": f"{ps:.4f}", "monitor_ssim": f"{ss:.4f}", "note": "best checkpoint updated"})
                meta_csv.flush()
            else:
                no_improve += 1

            if ps >= target_psnr:
                print(f"[EarlyStop] target reached: {ps:.4f} >= {target_psnr:.4f}")
                break
            if no_improve >= patience:
                print(f"[EarlyStop] plateau reached: no_improve={no_improve}, best_psnr={best_psnr:.4f}")
                break
        elif should_mon:
            msg = mon["message"] or "unknown monitor failure"
            print(f"[Monitor] epoch={epoch} status=failed message={msg}")

        if not should_save:
            ckpt_path.unlink(missing_ok=True)

    train_csv.close()
    mon_csv.close()
    meta_csv.close()
    writer.close()
    print("[Done] training finished")


if __name__ == "__main__":
    main()

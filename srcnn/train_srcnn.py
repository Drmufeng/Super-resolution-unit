"""SRCNN workspace training entry for SRResNet pretrain and SRGAN finetune."""

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

from src.datasets import SRDataset
from src.models import Discriminator, Generator
from src.utils import AverageMeter

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


def run_eval_monitor(root: Path, eval_cfg_path: Path, checkpoint_path: Path) -> dict:
    if not eval_cfg_path.is_file():
        return {"status": "failed", "count": None, "avg_psnr": None, "avg_ssim": None, "seconds": 0.0, "message": "eval config missing"}

    cfg = load_json(eval_cfg_path)
    cfg.setdefault("eval", {})["checkpoint"] = str(checkpoint_path).replace("\\", "/")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(cfg, tmp, indent=2)
        tmp_path = Path(tmp.name)

    try:
        cmd = [sys.executable, str(root / "eval.py"), "--config", str(tmp_path)]
        started = time.time()
        proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
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


def prune_old_checkpoints(save_dir: Path, prefix: str, keep_last: int) -> int:
    if keep_last <= 0:
        return 0
    pat = re.compile(rf"{re.escape(prefix)}_e(\d+)\.pth$")
    cands = []
    for p in save_dir.glob(f"{prefix}_e*.pth"):
        m = pat.search(p.name)
        if m:
            cands.append((int(m.group(1)), p))
    cands.sort(key=lambda x: x[0])
    removed = 0
    for _, p in cands[:-keep_last]:
        p.unlink(missing_ok=True)
        removed += 1
    return removed


def atomic_save(payload: dict, out_path: Path, backup_dir: Path | None) -> None:
    ensure_dir(out_path.parent)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(payload, str(tmp))
    tmp.replace(out_path)
    if backup_dir is not None:
        ensure_dir(backup_dir)
        b = backup_dir / out_path.name
        btmp = b.with_suffix(b.suffix + ".tmp")
        torch.save(payload, str(btmp))
        btmp.replace(b)


def try_load_checkpoint(path_text: str, root: Path, device: torch.device) -> dict | None:
    if not path_text:
        return None
    p = Path(path_text)
    if not p.is_absolute():
        p = (root / p).resolve()
    if not p.is_file():
        return None
    return torch.load(str(p), map_location=device, weights_only=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg = load_json(Path(args.config).resolve())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    print(f"[Info] device={device}")
    if torch.cuda.is_available():
        print(f"[Info] gpu={torch.cuda.get_device_name(0)}")

    train_cfg = cfg["train"]
    stage = str(train_cfg.get("stage", "pretrain")).strip().lower()
    scale = int(cfg["data"]["scaling_factor"])
    model_cfg = cfg["model"]

    generator = Generator(
        large_kernel_size=int(model_cfg.get("large_kernel_size_g", 9)),
        small_kernel_size=int(model_cfg.get("small_kernel_size_g", 3)),
        n_channels=int(model_cfg.get("n_channels_g", 64)),
        n_blocks=int(model_cfg.get("n_blocks_g", 16)),
        scaling_factor=scale,
    ).to(device)

    discriminator = None
    if stage == "finetune":
        discriminator = Discriminator(
            kernel_size=int(model_cfg.get("small_kernel_size_d", 3)),
            n_channels=int(model_cfg.get("n_channels_d", 64)),
            n_blocks=int(model_cfg.get("n_blocks_d", 8)),
            fc_size=int(model_cfg.get("fc_size_d", 1024)),
        ).to(device)

    save_dir = Path(cfg["save_dir"])
    if not save_dir.is_absolute():
        save_dir = (root / save_dir).resolve()
    log_dir = Path(cfg["log_dir"])
    if not log_dir.is_absolute():
        log_dir = (root / log_dir).resolve()
    csv_log = Path(cfg["csv_log"])
    if not csv_log.is_absolute():
        csv_log = (root / csv_log).resolve()

    backup_dir = None
    backup_text = str(cfg.get("backup_dir", "")).strip()
    if backup_text:
        backup_dir = Path(backup_text)
        if not backup_dir.is_absolute():
            backup_dir = (root / backup_dir).resolve()

    ensure_dir(save_dir)
    ensure_dir(log_dir)
    ensure_dir(csv_log.parent)

    data_folder = Path(cfg["data"]["data_folder"])
    if not data_folder.is_absolute():
        data_folder = (root / data_folder).resolve()

    ds = SRDataset(
        data_folder=str(data_folder),
        split="train",
        crop_size=int(cfg["data"]["crop_size"]),
        scaling_factor=scale,
        lr_img_type="imagenet-norm",
        hr_img_type="[-1, 1]",
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("workers", 0)),
        pin_memory=bool(train_cfg.get("pin_memory", True)) and torch.cuda.is_available(),
    )

    opt_g = torch.optim.Adam(generator.parameters(), lr=float(train_cfg["lr_g"]))
    sch_g = torch.optim.lr_scheduler.StepLR(
        opt_g,
        step_size=max(1, int(train_cfg.get("lr_step", 30))),
        gamma=float(train_cfg.get("lr_gamma", 0.5)),
    )

    opt_d = None
    sch_d = None
    if discriminator is not None:
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=float(train_cfg.get("lr_d", train_cfg["lr_g"])))
        sch_d = torch.optim.lr_scheduler.StepLR(
            opt_d,
            step_size=max(1, int(train_cfg.get("lr_step", 30))),
            gamma=float(train_cfg.get("lr_gamma", 0.5)),
        )

    start_epoch = int(train_cfg.get("start_epoch", 1))
    init_ckpt = try_load_checkpoint(str(train_cfg.get("init_checkpoint", "")).strip(), root, device)
    if init_ckpt is not None:
        g_state = init_ckpt.get("generator") or init_ckpt.get("model") or init_ckpt
        generator.load_state_dict(g_state, strict=False)
        print("[Info] init checkpoint loaded")

    resume_ckpt = try_load_checkpoint(str(train_cfg.get("resume_checkpoint", "")).strip(), root, device)
    if resume_ckpt is not None:
        g_state = resume_ckpt.get("generator") or resume_ckpt.get("model") or resume_ckpt
        generator.load_state_dict(g_state, strict=False)
        if isinstance(resume_ckpt.get("optimizer_g"), dict):
            opt_g.load_state_dict(resume_ckpt["optimizer_g"])
        if discriminator is not None and isinstance(resume_ckpt.get("discriminator"), dict):
            discriminator.load_state_dict(resume_ckpt["discriminator"], strict=False)
        if opt_d is not None and isinstance(resume_ckpt.get("optimizer_d"), dict):
            opt_d.load_state_dict(resume_ckpt["optimizer_d"])
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        print("[Info] resume checkpoint loaded")

    writer = SummaryWriter(log_dir=str(log_dir))
    train_csv = csv_log.open("a", newline="", encoding="utf-8")
    fields = ["epoch", "loss_g", "loss_pix", "loss_adv", "loss_d", "lr_g", "lr_d", "seconds"]
    writer_csv = csv.DictWriter(train_csv, fieldnames=fields)
    if train_csv.tell() == 0:
        writer_csv.writeheader()

    mon_cfg = cfg.get("monitor", {})
    mon_path = Path(mon_cfg.get("csv_log", str(log_dir / "val_metrics.csv")))
    if not mon_path.is_absolute():
        mon_path = (root / mon_path).resolve()
    ensure_dir(mon_path.parent)
    mon_csv = mon_path.open("a", newline="", encoding="utf-8")
    mon_writer = csv.DictWriter(mon_csv, fieldnames=["epoch", "checkpoint", "status", "count", "avg_psnr", "avg_ssim", "seconds", "message"])
    if mon_csv.tell() == 0:
        mon_writer.writeheader()

    eval_cfg = Path(mon_cfg.get("eval_config", ""))
    if not eval_cfg.is_absolute():
        eval_cfg = (root / eval_cfg).resolve()
    mon_every = int(mon_cfg.get("every", 1))
    target_psnr = float(mon_cfg.get("target_psnr", 999.0))
    patience = int(mon_cfg.get("patience", 60))
    min_delta = float(mon_cfg.get("min_delta", 0.0002))

    epochs = int(train_cfg["epochs"])
    save_every = int(train_cfg.get("save_every", 20))
    keep_last = int(train_cfg.get("keep_last", 8))
    print_every = int(train_cfg.get("print_every", 20))

    l1_w = float(cfg.get("loss", {}).get("l1_weight", 0.2))
    mse_w = float(cfg.get("loss", {}).get("mse_weight", 0.8))
    adv_w = float(cfg.get("loss", {}).get("adv_weight", 0.001))
    l1_loss = nn.L1Loss().to(device)
    mse_loss = nn.MSELoss().to(device)
    bce_logits = nn.BCEWithLogitsLoss().to(device)

    best_psnr = -1.0
    no_improve = 0
    prefix = str(cfg.get("checkpoint_prefix", f"srresnet_x{scale}" if stage == "pretrain" else f"srgan_x{scale}"))

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        generator.train()
        if discriminator is not None:
            discriminator.train()

        meter_g = AverageMeter()
        meter_pix = AverageMeter()
        meter_adv = AverageMeter()
        meter_d = AverageMeter()

        for i, (lr_img, hr_img) in enumerate(loader):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            bsz = lr_img.size(0)

            sr = generator(lr_img)
            loss_pix = l1_w * l1_loss(sr, hr_img) + mse_w * mse_loss(sr, hr_img)

            loss_adv = torch.tensor(0.0, device=device)
            loss_d = torch.tensor(0.0, device=device)

            if discriminator is not None and opt_d is not None:
                with torch.no_grad():
                    sr_detach = sr.detach()
                pred_real = discriminator(hr_img)
                pred_fake = discriminator(sr_detach)
                real_label = torch.ones_like(pred_real)
                fake_label = torch.zeros_like(pred_fake)
                loss_d = 0.5 * (bce_logits(pred_real, real_label) + bce_logits(pred_fake, fake_label))
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

                pred_fake_for_g = discriminator(sr)
                loss_adv = bce_logits(pred_fake_for_g, torch.ones_like(pred_fake_for_g))

            loss_g = loss_pix + adv_w * loss_adv
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            meter_g.update(float(loss_g.item()), bsz)
            meter_pix.update(float(loss_pix.item()), bsz)
            meter_adv.update(float(loss_adv.item()), bsz)
            meter_d.update(float(loss_d.item()), bsz)

            if i % print_every == 0:
                print(
                    f"epoch={epoch} iter={i} loss_g={meter_g.val:.6f} "
                    f"loss_pix={meter_pix.val:.6f} loss_adv={meter_adv.val:.6f} loss_d={meter_d.val:.6f}"
                )

        sch_g.step()
        if sch_d is not None:
            sch_d.step()

        lr_g_now = float(opt_g.param_groups[0]["lr"])
        lr_d_now = float(opt_d.param_groups[0]["lr"]) if opt_d is not None else 0.0
        sec = time.time() - t0

        writer.add_scalar("train/loss_g", meter_g.avg, epoch)
        writer.add_scalar("train/loss_pix", meter_pix.avg, epoch)
        writer.add_scalar("train/loss_adv", meter_adv.avg, epoch)
        if discriminator is not None:
            writer.add_scalar("train/loss_d", meter_d.avg, epoch)
        writer.add_scalar("train/lr_g", lr_g_now, epoch)

        writer_csv.writerow(
            {
                "epoch": epoch,
                "loss_g": f"{meter_g.avg:.8f}",
                "loss_pix": f"{meter_pix.avg:.8f}",
                "loss_adv": f"{meter_adv.avg:.8f}",
                "loss_d": f"{meter_d.avg:.8f}",
                "lr_g": f"{lr_g_now:.10f}",
                "lr_d": f"{lr_d_now:.10f}",
                "seconds": f"{sec:.2f}",
            }
        )
        train_csv.flush()

        should_save = epoch % save_every == 0 or epoch == epochs
        should_mon = epoch % mon_every == 0 or epoch == epochs
        if not (should_save or should_mon):
            continue

        ckpt_name = f"{prefix}_e{epoch}.pth" if should_save else f"_monitor_tmp_{prefix}.pth"
        ckpt_path = save_dir / ckpt_name
        payload = {
            "epoch": epoch,
            "stage": stage,
            "generator": generator.state_dict(),
            "optimizer_g": opt_g.state_dict(),
        }
        if discriminator is not None and opt_d is not None:
            payload["discriminator"] = discriminator.state_dict()
            payload["optimizer_d"] = opt_d.state_dict()

        atomic_save(payload, ckpt_path, backup_dir if should_save else None)
        if should_save:
            removed = prune_old_checkpoints(save_dir, prefix, keep_last)
            print(f"[Info] saved={ckpt_path}")
            if backup_dir is not None:
                print(f"[Info] mirrored={backup_dir / ckpt_path.name}")
            if removed:
                print(f"[Info] pruned old checkpoints: {removed}")

        if should_mon:
            print(f"[Monitor] running eval at epoch={epoch} with checkpoint={ckpt_path}")
        mon = run_eval_monitor(root, eval_cfg, ckpt_path)
        mon_writer.writerow(
            {
                "epoch": epoch,
                "checkpoint": str(ckpt_path),
                "status": mon["status"],
                "count": mon["count"],
                "avg_psnr": "" if mon["avg_psnr"] is None else f"{mon['avg_psnr']:.4f}",
                "avg_ssim": "" if mon["avg_ssim"] is None else f"{mon['avg_ssim']:.4f}",
                "seconds": f"{mon['seconds']:.2f}",
                "message": mon["message"],
            }
        )
        mon_csv.flush()

        if mon["status"] == "ok" and mon["avg_psnr"] is not None:
            ps = float(mon["avg_psnr"])
            ss = float(mon["avg_ssim"] or 0.0)
            print(f"[Monitor] epoch={epoch} psnr={ps:.4f} ssim={ss:.4f}")
            if ps > best_psnr + min_delta:
                best_psnr = ps
                no_improve = 0
                best_path = save_dir / f"{prefix}_best.pth"
                best_payload = dict(payload)
                best_payload["best_psnr"] = best_psnr
                atomic_save(best_payload, best_path, backup_dir)
                print(f"[Best] epoch={epoch} psnr={ps:.4f} -> {best_path}")
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
    writer.close()
    print("[Done] training finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""EDSR 评测：出图 + PSNR/SSIM。"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.transforms import InterpolationMode, ToPILImage

from psnr_models import EDSRGenerator


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0].lower()
    stem = re.sub(r"([_\-]?)x\d+$", "", stem)
    stem = stem.replace("_bicubic", "").replace("_lr", "").rstrip("_-")
    return stem


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def build_stem_map(folder: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in folder.iterdir():
        if is_image_file(p):
            out.setdefault(normalize_stem(p.name), p)
    return out


def pil_to_imagenet_norm(img: Image.Image) -> torch.Tensor:
    t = TF.to_tensor(img)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t - mean) / std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    ws = Path(__file__).resolve().parent
    cfg = load_json(Path(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EDSRGenerator(
        scale=int(cfg["data"]["scaling_factor"]),
        n_resblocks=int(cfg["model"].get("n_blocks_g", 32)),
        n_feats=int(cfg["model"].get("n_channels_g", 128)),
        res_scale=float(cfg["model"].get("res_scale", 0.1)),
    ).to(device)

    ckpt = Path(cfg["eval"]["checkpoint"])
    if not ckpt.is_absolute():
        ckpt = (ws / ckpt).resolve()
    state = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(state.get("model", state), strict=False)
    model.eval()

    lr_dir = Path(cfg["eval"]["lr_folder"])
    hr_dir = Path(cfg["eval"]["hr_folder"])
    out_dir = Path(cfg["eval"]["output_folder"])
    if not lr_dir.is_absolute():
        lr_dir = (ws / lr_dir).resolve()
    if not hr_dir.is_absolute():
        hr_dir = (ws / hr_dir).resolve()
    if not out_dir.is_absolute():
        out_dir = (ws / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hr_map = build_stem_map(hr_dir)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    to_pil = ToPILImage()

    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0

    with torch.no_grad():
        for p in sorted(lr_dir.iterdir()):
            if not is_image_file(p):
                continue
            key = normalize_stem(p.name)
            if key not in hr_map:
                continue

            lr_img = Image.open(p).convert("RGB")
            hr_img = Image.open(hr_map[key]).convert("RGB")
            inp = pil_to_imagenet_norm(lr_img).unsqueeze(0).to(device)
            out = model(inp).squeeze(0).cpu()
            sr_01 = out.add(1.0).div(2.0).clamp(0.0, 1.0)

            to_pil(sr_01).save(out_dir / p.name)

            hr_01 = TF.to_tensor(hr_img)
            if sr_01.size() != hr_01.size():
                hr_01 = TF.resize(hr_01, sr_01.shape[1:], interpolation=InterpolationMode.BICUBIC)

            ps = psnr_metric(sr_01.unsqueeze(0).to(device), hr_01.unsqueeze(0).to(device)).item()
            ss = ssim_metric(sr_01.unsqueeze(0).to(device), hr_01.unsqueeze(0).to(device)).item()
            psnr_sum += ps
            ssim_sum += ss
            count += 1
            print(f"{p.name} -> PSNR: {ps:.4f}, SSIM: {ss:.4f}")

    if count == 0:
        print("[Done] No valid image pairs evaluated")
        return

    print(f"[Done] count={count}")
    print(f"[Done] avg_psnr={psnr_sum / count:.4f}")
    print(f"[Done] avg_ssim={ssim_sum / count:.4f}")


if __name__ == "__main__":
    main()

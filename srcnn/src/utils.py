"""通用工具函数。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torchvision.transforms.functional as TF
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def convert_image(img, source: str, target: str):
    """图像格式与数值域转换。

    常用：
    - source='pil' -> target='imagenet-norm' 或 '[-1, 1]'
    - source='[-1, 1]' -> target='imagenet-norm'
    """
    if source == target:
        return img

    if source == "pil":
        x = TF.to_tensor(img)
    elif source == "[0, 1]":
        x = img
    elif source == "[-1, 1]":
        x = img.add(1.0).div(2.0)
    elif source == "imagenet-norm":
        x = img * IMAGENET_STD.to(img.device) + IMAGENET_MEAN.to(img.device)
    else:
        raise ValueError(f"Unsupported source: {source}")

    if target == "[0, 1]":
        return x
    if target == "[-1, 1]":
        return x.mul(2.0).sub(1.0)
    if target == "imagenet-norm":
        mean = IMAGENET_MEAN.to(x.device)
        std = IMAGENET_STD.to(x.device)
        return (x - mean) / std
    if target == "pil":
        return TF.to_pil_image(x.clamp(0.0, 1.0))
    raise ValueError(f"Unsupported target: {target}")


@dataclass
class AverageMeter:
    """平均值统计器。"""

    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


def adjust_learning_rate(optimizer: torch.optim.Optimizer, shrink_factor: float) -> None:
    """按比例衰减学习率。"""
    for param_group in optimizer.param_groups:
        param_group["lr"] *= shrink_factor

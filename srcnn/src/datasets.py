"""训练数据集读取。"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

from .utils import convert_image


class SRDataset(Dataset):
    """从 train_images.json 读取 HR 图，在线生成 LR 图。"""

    def __init__(
        self,
        data_folder: str,
        split: str,
        crop_size: int,
        scaling_factor: int,
        lr_img_type: str,
        hr_img_type: str,
    ) -> None:
        super().__init__()
        if split != "train":
            raise ValueError("Only 'train' split is supported in this dataset.")

        self.data_folder = Path(data_folder)
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type

        list_path = self.data_folder / "train_images.json"
        if not list_path.is_file():
            raise FileNotFoundError(f"Missing data index: {list_path}")

        raw = json.loads(list_path.read_text(encoding="utf-8"))
        self.images = [self._resolve_path(p) for p in raw]

    def _resolve_path(self, p: str) -> Path:
        p = p.replace("\\", "/")
        candidate = Path(p)
        if candidate.is_absolute() and candidate.is_file():
            return candidate

        if p.startswith("./"):
            p = p[2:]
        # 常见写法：data/df2k/...，拼接到 data_folder 的上级。
        cand1 = (self.data_folder.parent / p).resolve()
        if cand1.is_file():
            return cand1
        cand2 = (self.data_folder / p).resolve()
        if cand2.is_file():
            return cand2
        return cand1

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        img = Image.open(self.images[i]).convert("RGB")

        # 随机裁剪 HR patch（可被 scaling_factor 整除）。
        crop = self.crop_size
        crop -= crop % self.scaling_factor
        w, h = img.size
        if w < crop or h < crop:
            new_w = max(w, crop)
            new_h = max(h, crop)
            img = img.resize((new_w, new_h), resample=Image.BICUBIC)
            w, h = img.size

        left = random.randint(0, w - crop)
        top = random.randint(0, h - crop)
        hr = TF.crop(img, top=top, left=left, height=crop, width=crop)

        lr = TF.resize(
            hr,
            [crop // self.scaling_factor, crop // self.scaling_factor],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

        lr = convert_image(lr, source="pil", target=self.lr_img_type)
        hr = convert_image(hr, source="pil", target=self.hr_img_type)
        return lr, hr

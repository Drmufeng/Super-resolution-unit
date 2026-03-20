"""EDSR 风格模型（PSNR 主线）。"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """无 BN 残差块，含残差缩放。"""

    def __init__(self, n_feats: int, res_scale: float = 0.1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
        )
        self.res_scale = float(res_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class UpsampleBlock(nn.Module):
    """PixelShuffle 上采样。"""

    def __init__(self, n_feats: int, scale: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if scale in (2, 4, 8):
            for _ in range({2: 1, 4: 2, 8: 3}[scale]):
                layers += [nn.Conv2d(n_feats, n_feats * 4, 3, 1, 1), nn.PixelShuffle(2)]
        elif scale == 3:
            layers += [nn.Conv2d(n_feats, n_feats * 9, 3, 1, 1), nn.PixelShuffle(3)]
        else:
            raise ValueError(f"Unsupported scale: {scale}")
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class EDSRGenerator(nn.Module):
    """EDSR 工程实现：无 BN + residual scaling + PixelShuffle。"""

    def __init__(self, scale: int, n_resblocks: int = 32, n_feats: int = 128, res_scale: float = 0.1) -> None:
        super().__init__()
        self.head = nn.Conv2d(3, n_feats, 3, 1, 1)
        self.body = nn.Sequential(*[ResidualBlock(n_feats, res_scale) for _ in range(n_resblocks)])
        self.body_tail = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.upsample = UpsampleBlock(n_feats, scale)
        self.tail = nn.Sequential(nn.Conv2d(n_feats, 3, 3, 1, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        res = self.body_tail(self.body(x))
        x = x + res
        x = self.upsample(x)
        return self.tail(x)

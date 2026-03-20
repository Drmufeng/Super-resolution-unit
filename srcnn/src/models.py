"""SRResNet/SRGAN 兼容模型定义。"""

from __future__ import annotations

import math

import torch
from torch import nn


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation == "prelu":
            layers.append(nn.PReLU())
        elif activation == "leakyrelu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "tanh":
            layers.append(nn.Tanh())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SubPixelConvolutionalBlock(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels * (scaling_factor**2), kernel_size, padding=kernel_size // 2)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return self.prelu(x)


class ResidualBlock(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64):
        super().__init__()
        self.conv_block1 = ConvolutionalBlock(n_channels, n_channels, kernel_size, batch_norm=True, activation="prelu")
        self.conv_block2 = ConvolutionalBlock(n_channels, n_channels, kernel_size, batch_norm=True, activation=None)

    def forward(self, x):
        return self.conv_block2(self.conv_block1(x)) + x


class SRResNet(nn.Module):
    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.conv_block1 = ConvolutionalBlock(3, n_channels, large_kernel_size, activation="prelu")
        self.residual_blocks = nn.Sequential(*[ResidualBlock(small_kernel_size, n_channels) for _ in range(n_blocks)])
        self.conv_block2 = ConvolutionalBlock(n_channels, n_channels, small_kernel_size, batch_norm=True, activation=None)
        n_upsample = int(math.log2(scaling_factor))
        self.subpixel_blocks = nn.Sequential(
            *[SubPixelConvolutionalBlock(small_kernel_size, n_channels, scaling_factor=2) for _ in range(n_upsample)]
        )
        self.conv_block3 = ConvolutionalBlock(n_channels, 3, large_kernel_size, activation="tanh")

    def forward(self, x):
        out1 = self.conv_block1(x)
        out = self.residual_blocks(out1)
        out = self.conv_block2(out)
        out = out + out1
        out = self.subpixel_blocks(out)
        return self.conv_block3(out)


class Generator(nn.Module):
    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        super().__init__()
        self.net = SRResNet(large_kernel_size, small_kernel_size, n_channels, n_blocks, scaling_factor)

    def forward(self, lr_imgs):
        return self.net(lr_imgs)


class Discriminator(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64, n_blocks=8, fc_size=1024):
        super().__init__()
        blocks = [ConvolutionalBlock(3, n_channels, kernel_size, activation="leakyrelu")]
        in_c = n_channels
        for i in range(1, n_blocks):
            out_c = in_c if i % 2 == 1 else min(in_c * 2, 512)
            stride = 2 if i % 2 == 1 else 1
            blocks.append(ConvolutionalBlock(in_c, out_c, kernel_size, stride=stride, batch_norm=True, activation="leakyrelu"))
            in_c = out_c
        self.conv_blocks = nn.Sequential(*blocks)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(in_c * 6 * 6, fc_size)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(fc_size, 1)

    def forward(self, imgs):
        out = self.conv_blocks(imgs)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        out = self.leaky(self.fc1(out))
        return self.fc2(out)

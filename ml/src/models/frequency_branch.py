"""Small CNN over the FFT-magnitude input.

Input: log-scaled magnitude + per-image standardized.
Picks up spectral-*shape* anomalies diffusion models leave behind compared to GAN upsampling artifacts - not to do expensive
general-purpose feature extraction, which is what the spatial branch is for.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU, stride-2 downsample."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class FrequencyBranch(nn.Module):
    """FFT-magnitude map (B, 1, H, W) -> embedding (B, embed_dim).

    Channel widths and embed_dim default to configs/model.yaml's frequency_branch section.
    """

    def __init__(
        self,
        conv_channels: tuple[int, ...] = (32, 64, 128, 256),
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        channels = [1, *conv_channels]
        self.blocks = nn.ModuleList(
            ConvBlock(channels[i], channels[i + 1]) for i in range(len(conv_channels))
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(conv_channels[-1], embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).flatten(1)
        return self.project(x)

"""Full dual-branch model: wires SpatialBranch + FrequencyBranch + FusionHead together.

Takes the two precomputed tensors produced by datasets/transforms.py - the RGB tensor
for the spatial branch and the FFT-magnitude tensor for the frequency branch - since
they're two different views of the same source image, not something this model derives.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.frequency_branch import FrequencyBranch
from src.models.fusion import FusionHead
from src.models.spatial_branch import SpatialBranch


class DualBranchDetector(nn.Module):
    def __init__(
        self,
        spatial_backbone: str = "vit_small_patch16_224",
        spatial_pretrained: bool = True,
        spatial_embed_dim: int = 384,
        freeze_spatial_backbone: bool = False,
        freq_conv_channels: tuple[int, ...] = (32, 64, 128, 256),
        freq_embed_dim: int = 256,
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.spatial_branch = SpatialBranch(
            backbone=spatial_backbone,
            pretrained=spatial_pretrained,
            embed_dim=spatial_embed_dim,
            freeze_backbone=freeze_spatial_backbone,
        )
        self.frequency_branch = FrequencyBranch(
            conv_channels=freq_conv_channels,
            embed_dim=freq_embed_dim,
        )
        self.fusion = FusionHead(
            spatial_dim=spatial_embed_dim,
            freq_dim=freq_embed_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=fusion_dropout,
            num_classes=num_classes,
        )

    def forward(self, rgb: torch.Tensor, freq: torch.Tensor) -> dict[str, torch.Tensor]:
        spatial_embed = self.spatial_branch(rgb)
        freq_embed = self.frequency_branch(freq)
        return self.fusion(spatial_embed, freq_embed)

    @classmethod
    def from_config(cls, model_cfg: dict) -> "DualBranchDetector":
        """Builds the model from a parsed configs/model.yaml dict."""
        sp = model_cfg["spatial_branch"]
        fr = model_cfg["frequency_branch"]
        fu = model_cfg["fusion"]
        return cls(
            spatial_backbone=sp["backbone"],
            spatial_pretrained=sp["pretrained"],
            spatial_embed_dim=sp["embed_dim"],
            freeze_spatial_backbone=sp["freeze_backbone"],
            freq_conv_channels=tuple(fr["conv_channels"]),
            freq_embed_dim=fr["embed_dim"],
            fusion_hidden_dim=fu["hidden_dim"],
            fusion_dropout=fu["dropout"],
            num_classes=fu["num_classes"],
        )

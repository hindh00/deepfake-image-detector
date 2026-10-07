"""ViT-Small wrapper for the spatial branch (see configs/model.yaml).

Picks up GAN-era spatial/upsampling artifacts via a standard pretrained ViT backbone.
Classification head is discarded; forward() returns the pooled CLS embedding.
"""

from __future__ import annotations

import timm
import torch
from torch import nn


class SpatialBranch(nn.Module):
    def __init__(
        self,
        backbone: str = "vit_small_patch16_224",
        pretrained: bool = True,
        embed_dim: int = 384,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)

        native_dim = self.backbone.num_features
        self.project = nn.Identity() if embed_dim == native_dim else nn.Linear(native_dim, embed_dim)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.backbone(x))

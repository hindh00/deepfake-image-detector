"""Fusion head: combines spatial + frequency embeddings into fused logits,
while also exposing auxiliary single-branch logits and a learned gate.

The gate and auxiliary logits aren't decorative - they're what the multi-task loss
(configs/model.yaml -> loss) trains against, and what lets the UI show a real
"how much did each branch contribute" breakdown instead of a post-hoc guess.
"""

from __future__ import annotations

import torch
from torch import nn


class FusionHead(nn.Module):
    def __init__(
        self,
        spatial_dim: int,
        freq_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.spatial_proj = nn.Linear(spatial_dim, hidden_dim)
        self.freq_proj = nn.Linear(freq_dim, hidden_dim)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

        self.fused_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.spatial_aux_head = nn.Linear(spatial_dim, num_classes)
        self.freq_aux_head = nn.Linear(freq_dim, num_classes)

    def forward(self, spatial_embed: torch.Tensor, freq_embed: torch.Tensor) -> dict[str, torch.Tensor]:
        spatial_proj = self.spatial_proj(spatial_embed)
        freq_proj = self.freq_proj(freq_embed)

        gate_logits = self.gate(torch.cat([spatial_proj, freq_proj], dim=-1))
        gate_weights = torch.softmax(gate_logits, dim=-1)  # (B, 2): [spatial_weight, freq_weight]

        fused_repr = gate_weights[:, 0:1] * spatial_proj + gate_weights[:, 1:2] * freq_proj
        fused_logits = self.fused_head(fused_repr)

        return {
            "fused_logits": fused_logits,
            "spatial_logits": self.spatial_aux_head(spatial_embed),
            "freq_logits": self.freq_aux_head(freq_embed),
            "gate_weights": gate_weights,
        }

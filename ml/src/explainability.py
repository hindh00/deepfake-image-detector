"""Dual explainability (configs/model.yaml -> explainability): attention rollout for
the spatial branch ("where" in the image looks suspicious) and gradient x input saliency
on the frequency branch's own auxiliary logit ("which frequencies" look suspicious).

Both heatmaps are only meaningful once the model is actually trained - on random weights
they're structurally valid (right shape, no NaNs) but not informative. Written now so they
can be wired into train.py's per-epoch canary-batch logging once a real checkpoint exists.
"""

from __future__ import annotations

import contextlib
import math
import types
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import nn

from src.models.frequency_branch import FrequencyBranch
from src.models.full_model import DualBranchDetector
from src.models.spatial_branch import SpatialBranch


def _capturing_attention_forward(self, x, attn_mask=None, is_causal=False, _storage=None):
    assert self.gate is None, "gated attention is not supported by attention_rollout"
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    attn = (q * self.scale) @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    _storage.append(attn.detach())
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, self.attn_dim)
    x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


@contextlib.contextmanager
def _capture_attention(blocks) -> Iterator[list[torch.Tensor]]:
    """Temporarily forces each block's attention to compute explicit softmax weights
    (bypassing timm's fused/SDPA fast path, which never exposes them) and records one
    (B, heads, N, N) tensor per layer. Restores the original forward on exit."""
    storage: list[torch.Tensor] = []
    had_override = ["forward" in block.attn.__dict__ for block in blocks]
    originals = [block.attn.__dict__.get("forward") for block in blocks]
    for block in blocks:
        bound = types.MethodType(
            lambda self, x, attn_mask=None, is_causal=False: _capturing_attention_forward(
                self, x, attn_mask, is_causal, _storage=storage
            ),
            block.attn,
        )
        block.attn.forward = bound
    try:
        yield storage
    finally:
        for block, had, original in zip(blocks, had_override, originals):
            if had:
                block.attn.forward = original
            else:
                del block.attn.forward


def _min_max_normalize(x: torch.Tensor) -> torch.Tensor:
    flat = x.flatten(1)
    mins = flat.min(dim=1, keepdim=True).values
    maxs = flat.max(dim=1, keepdim=True).values
    normalized = (flat - mins) / (maxs - mins + 1e-8)
    return normalized.reshape(x.shape)


@torch.no_grad()
def attention_rollout(spatial_branch: SpatialBranch, rgb: torch.Tensor) -> torch.Tensor:
    """Returns a (B, H, W) heatmap in [0, 1]: how much each region of the input image
    influenced the CLS token's final representation, aggregated across all ViT layers
    (Abnar & Zuidema, 2020)."""
    backbone = spatial_branch.backbone
    blocks = backbone.blocks

    with _capture_attention(blocks) as storage:
        backbone.forward_features(rgb)

    num_tokens = storage[0].shape[-1]
    eye = torch.eye(num_tokens, device=rgb.device)

    rollout = None
    for attn in storage:
        attn_avg = attn.mean(dim=1)  # average over heads -> (B, N, N)
        attn_aug = attn_avg + eye
        attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True)
        rollout = attn_aug if rollout is None else attn_aug @ rollout

    cls_attention = rollout[:, 0, 1:]  # CLS token's rolled-out attention to each patch
    grid_size = math.isqrt(cls_attention.shape[-1])
    heatmap = cls_attention.reshape(-1, 1, grid_size, grid_size)
    heatmap = F.interpolate(heatmap, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
    heatmap = heatmap.squeeze(1)

    return _min_max_normalize(heatmap)


def frequency_saliency(
    frequency_branch: FrequencyBranch,
    freq_aux_head: nn.Linear,
    freq_input: torch.Tensor,
    target_class: int = 1,
) -> torch.Tensor:
    """Returns a (B, H, W) heatmap in [0, 1]: |gradient x input| of the frequency
    branch's own auxiliary logit for `target_class` (default 1 = fake) with respect to
    the FFT-magnitude input - which frequency bands pushed the branch toward that class."""
    freq_input = freq_input.clone().requires_grad_(True)

    freq_embed = frequency_branch(freq_input)
    freq_logits = freq_aux_head(freq_embed)
    score = freq_logits[:, target_class].sum()
    score.backward()

    saliency = (freq_input.grad * freq_input).detach().abs().squeeze(1)
    return _min_max_normalize(saliency)


def explain(model: DualBranchDetector, rgb: torch.Tensor, freq: torch.Tensor, target_class: int = 1) -> dict:
    """Bundles both heatmaps with the model's own prediction breakdown - the single
    entry point train.py's canary logging and space/'s inference code are meant to call."""
    model.eval()

    spatial_heatmap = attention_rollout(model.spatial_branch, rgb)
    frequency_heatmap = frequency_saliency(model.frequency_branch, model.fusion.freq_aux_head, freq, target_class)

    with torch.no_grad():
        outputs = model(rgb, freq)

    return {
        "spatial_heatmap": spatial_heatmap,
        "frequency_heatmap": frequency_heatmap,
        "fused_probs": outputs["fused_logits"].softmax(dim=-1),
        "spatial_probs": outputs["spatial_logits"].softmax(dim=-1),
        "freq_probs": outputs["freq_logits"].softmax(dim=-1),
        "gate_weights": outputs["gate_weights"],
    }

import torch

from src.models.fusion import FusionHead


def _make_head(spatial_dim=384, freq_dim=256, num_classes=2):
    return FusionHead(spatial_dim=spatial_dim, freq_dim=freq_dim, num_classes=num_classes)


def test_output_keys_and_shapes():
    head = _make_head()
    spatial_embed = torch.randn(4, 384)
    freq_embed = torch.randn(4, 256)
    out = head(spatial_embed, freq_embed)

    assert set(out.keys()) == {"fused_logits", "spatial_logits", "freq_logits", "gate_weights"}
    assert out["fused_logits"].shape == (4, 2)
    assert out["spatial_logits"].shape == (4, 2)
    assert out["freq_logits"].shape == (4, 2)
    assert out["gate_weights"].shape == (4, 2)


def test_handles_batch_size_one():
    head = _make_head()
    out = head(torch.randn(1, 384), torch.randn(1, 256))
    assert out["fused_logits"].shape == (1, 2)


def test_gate_weights_are_a_valid_distribution():
    head = _make_head()
    out = head(torch.randn(8, 384), torch.randn(8, 256))
    gate = out["gate_weights"]
    assert torch.all(gate >= 0) and torch.all(gate <= 1)
    assert torch.allclose(gate.sum(dim=-1), torch.ones(8), atol=1e-5)


def test_gradients_flow_to_both_branches_and_all_heads():
    head = _make_head()
    spatial_embed = torch.randn(4, 384, requires_grad=True)
    freq_embed = torch.randn(4, 256, requires_grad=True)
    out = head(spatial_embed, freq_embed)

    loss = out["fused_logits"].sum() + out["spatial_logits"].sum() + out["freq_logits"].sum()
    loss.backward()

    assert spatial_embed.grad is not None and torch.isfinite(spatial_embed.grad).all()
    assert freq_embed.grad is not None and torch.isfinite(freq_embed.grad).all()
    assert head.spatial_aux_head.weight.grad is not None
    assert head.freq_aux_head.weight.grad is not None
    assert head.gate[0].weight.grad is not None


def test_respects_custom_num_classes():
    head = _make_head(num_classes=5)
    out = head(torch.randn(2, 384), torch.randn(2, 256))
    assert out["fused_logits"].shape == (2, 5)
    assert out["spatial_logits"].shape == (2, 5)
    assert out["freq_logits"].shape == (2, 5)

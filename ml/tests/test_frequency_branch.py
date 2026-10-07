import torch

from src.models.frequency_branch import FrequencyBranch


def test_output_shape_matches_embed_dim():
    branch = FrequencyBranch(embed_dim=256)
    x = torch.randn(4, 1, 224, 224)
    out = branch(x)
    assert out.shape == (4, 256)


def test_handles_batch_size_one():
    branch = FrequencyBranch(embed_dim=256)
    out = branch(torch.randn(1, 1, 224, 224))
    assert out.shape == (1, 256)


def test_custom_embed_dim_and_channels():
    branch = FrequencyBranch(conv_channels=(16, 32), embed_dim=64)
    out = branch(torch.randn(2, 1, 224, 224))
    assert out.shape == (2, 64)


def test_no_nan_or_inf_in_output():
    branch = FrequencyBranch()
    out = branch(torch.randn(4, 1, 224, 224))
    assert torch.isfinite(out).all()


def test_gradients_flow_to_first_conv_layer():
    branch = FrequencyBranch()
    x = torch.randn(2, 1, 224, 224, requires_grad=True)
    out = branch(x)
    out.sum().backward()
    first_conv = branch.blocks[0].conv
    assert first_conv.weight.grad is not None
    assert torch.isfinite(first_conv.weight.grad).all()
    assert first_conv.weight.grad.abs().sum() > 0

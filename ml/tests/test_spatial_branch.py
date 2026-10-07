import torch

from src.models.spatial_branch import SpatialBranch


def test_output_shape_matches_native_embed_dim():
    branch = SpatialBranch(pretrained=False, embed_dim=384)
    out = branch(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 384)


def test_handles_batch_size_one():
    branch = SpatialBranch(pretrained=False, embed_dim=384)
    out = branch(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 384)


def test_projects_when_embed_dim_differs_from_backbone():
    branch = SpatialBranch(pretrained=False, embed_dim=128)
    assert isinstance(branch.project, torch.nn.Linear)
    out = branch(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 128)


def test_identity_projection_when_embed_dim_matches_backbone():
    branch = SpatialBranch(pretrained=False, embed_dim=384)
    assert isinstance(branch.project, torch.nn.Identity)


def test_freeze_backbone_disables_grad():
    branch = SpatialBranch(pretrained=False, embed_dim=384, freeze_backbone=True)
    assert all(not p.requires_grad for p in branch.backbone.parameters())


def test_unfrozen_backbone_receives_gradients():
    branch = SpatialBranch(pretrained=False, embed_dim=384, freeze_backbone=False)
    out = branch(torch.randn(2, 3, 224, 224))
    out.sum().backward()
    some_param = next(branch.backbone.parameters())
    assert some_param.grad is not None


def test_pretrained_weights_load_and_produce_finite_output():
    branch = SpatialBranch(pretrained=True, embed_dim=384)
    out = branch(torch.randn(1, 3, 224, 224))
    assert torch.isfinite(out).all()

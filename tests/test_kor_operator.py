import torch

from models.kor_operator import KORBlock


def test_kor_shape_gradient_and_finiteness():
    block = KORBlock(dim=16, heads=2, temperature=0.2, dropout=0.0)
    x = torch.randn(2, 12, 16, requires_grad=True)
    output, diagnostics = block(x)
    assert output.shape == x.shape
    assert diagnostics["rank_scores"].shape == (2, 2, 12)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert block.rank_net[1].weight.grad is not None
    assert torch.isfinite(block.rank_net[1].weight.grad).all()


def test_hard_and_soft_paths_execute():
    block = KORBlock(dim=8, heads=1)
    x = torch.randn(1, 5, 8)
    soft, _ = block(x, "soft")
    hard, _ = block(x, "hard")
    assert soft.shape == hard.shape == x.shape

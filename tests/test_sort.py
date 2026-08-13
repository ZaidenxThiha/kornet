import torch

from models.differentiable_sort import NeuralSorter


def test_hard_sort_permutation_orders_values():
    scores = torch.tensor([[[2.0, -1.0, 3.0]]])
    permutation = NeuralSorter(mode="hard")(scores, descending=True)
    ordered = permutation @ scores.unsqueeze(-1)
    assert ordered.flatten().tolist() == [3.0, 2.0, -1.0]
    assert torch.allclose(permutation.sum(-1), torch.ones(1, 1, 3))


def test_soft_sort_has_gradient_and_valid_rows():
    scores = torch.randn(2, 4, 7, requires_grad=True)
    permutation = NeuralSorter(temperature=0.2)(scores)
    assert permutation.shape == (2, 4, 7, 7)
    assert torch.allclose(permutation.sum(-1), torch.ones(2, 4, 7), atol=1e-5)
    permutation.square().sum().backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_sinkhorn_iterations_make_soft_permutation_nearly_doubly_stochastic():
    scores = torch.randn(2, 3, 7)
    unrefined = NeuralSorter(0.2, "soft", sinkhorn_iterations=0)(scores)
    permutation = NeuralSorter(0.2, "soft", sinkhorn_iterations=10)(scores)
    assert torch.allclose(permutation.sum(-1), torch.ones(2, 3, 7), atol=1e-4)
    target = torch.ones(2, 3, 7)
    assert (permutation.sum(-2) - target).abs().mean() < (unrefined.sum(-2) - target).abs().mean()

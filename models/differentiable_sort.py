from __future__ import annotations

import torch
from torch import nn


class NeuralSorter(nn.Module):
    """NeuralSort relaxation with an optional exact argsort inference path.

    Input scores have shape [B, H, N]; output permutations are [B, H, N, N],
    where rows are ordered positions and columns are original token positions.
    Complexity is O(B H N^2), which is why KORNet caps N explicitly.
    """

    def __init__(
        self, temperature: float = 0.1, mode: str = "soft", sinkhorn_iterations: int = 0
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if mode not in {"soft", "hard"}:
            raise ValueError("sort mode must be 'soft' or 'hard'")
        self.temperature = temperature
        self.mode = mode
        if sinkhorn_iterations < 0:
            raise ValueError("sinkhorn_iterations must be non-negative")
        self.sinkhorn_iterations = sinkhorn_iterations

    @staticmethod
    def hard_permutation(scores: torch.Tensor, descending: bool) -> torch.Tensor:
        indices = torch.argsort(scores, dim=-1, descending=descending)
        return torch.nn.functional.one_hot(indices, scores.shape[-1]).to(scores.dtype)

    def soft_permutation(self, scores: torch.Tensor, descending: bool) -> torch.Tensor:
        if not descending:
            scores = -scores
        n = scores.shape[-1]
        pairwise = (scores.unsqueeze(-1) - scores.unsqueeze(-2)).abs().sum(dim=-1)
        ranks = torch.arange(1, n + 1, device=scores.device, dtype=scores.dtype)
        scaling = n + 1 - 2 * ranks
        logits = scaling.view(*([1] * (scores.ndim - 1)), n, 1) * scores.unsqueeze(-2)
        logits = logits - pairwise.unsqueeze(-2)
        permutation = torch.softmax(logits / self.temperature, dim=-1)
        for _ in range(self.sinkhorn_iterations):
            permutation = permutation / permutation.sum(dim=-2, keepdim=True).clamp_min(1e-8)
            permutation = permutation / permutation.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return permutation

    def forward(self, scores: torch.Tensor, descending: bool = True, mode: str | None = None):
        selected = mode or self.mode
        if selected == "hard":
            return self.hard_permutation(scores, descending)
        return self.soft_permutation(scores, descending)

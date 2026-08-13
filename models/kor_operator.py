from __future__ import annotations

import torch
from torch import nn

from .differentiable_sort import NeuralSorter


class KORBlock(nn.Module):
    """Differentiable opposing-ranking update with a scalar adaptive gate per sample."""

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        temperature: float = 0.1,
        sort_mode: str = "soft",
        dropout: float = 0.1,
        ranking: str = "learned",
        opposing_subtraction: bool = True,
        sinkhorn_iterations: int = 0,
    ) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError("heads must be >= 1")
        self.dim, self.heads = dim, heads
        self.ranking = ranking
        self.opposing_subtraction = opposing_subtraction
        self.rank_net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, heads))
        self.sorter = NeuralSorter(temperature, sort_mode, sinkhorn_iterations)
        self.combine = nn.Linear(dim * heads, dim)
        self.phi = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim // 2 or 1), nn.GELU(), nn.Linear(dim // 2 or 1, 1)
        )
        self.norm = nn.LayerNorm(dim)

    def _scores(self, x: torch.Tensor) -> torch.Tensor:
        if self.ranking == "magnitude":
            base = x.norm(dim=-1, keepdim=True)
            return base.expand(-1, -1, self.heads).transpose(1, 2)
        return self.rank_net(x).transpose(1, 2)

    def forward(self, x: torch.Tensor, sort_mode: str | None = None):
        scores = self._scores(x)  # [B, H, N]
        p_down = self.sorter(scores, True, sort_mode)
        p_up = self.sorter(scores, False, sort_mode)
        expanded = x.unsqueeze(1).expand(-1, self.heads, -1, -1)
        down = torch.matmul(p_down, expanded)
        up = torch.matmul(p_up, expanded)
        ranked = down - up if self.opposing_subtraction else down
        # Bring each head back to original token coordinates before combining heads.
        restored = torch.matmul(p_down.transpose(-1, -2), ranked)
        difference = self.combine(restored.transpose(1, 2).flatten(2))
        update = self.phi(difference)
        pooled = torch.cat([x.mean(1), difference.abs().mean(1)], dim=-1)
        gate = torch.sigmoid(self.gate(pooled)).unsqueeze(1)
        output = self.norm(x + gate * update)
        diagnostics = {
            "rank_scores": scores,
            "rank_difference": difference,
            "gate": gate.squeeze(-1).squeeze(-1),
        }
        return output, diagnostics

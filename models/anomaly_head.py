from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AnomalyHead(nn.Module):
    def __init__(self, dim: int, score_weights=(1.0, 0.2, 0.1), momentum: float = 0.99):
        super().__init__()
        self.register_buffer("prototype", torch.zeros(dim))
        self.register_buffer("prototype_initialized", torch.tensor(False))
        self.momentum = momentum
        self.register_buffer("score_weights", torch.tensor(score_weights, dtype=torch.float32))

    @torch.no_grad()
    def update_prototype(self, tokens: torch.Tensor) -> None:
        center = tokens.detach().mean(dim=(0, 1))
        if not bool(self.prototype_initialized):
            self.prototype.copy_(center)
            self.prototype_initialized.fill_(True)
        else:
            self.prototype.mul_(self.momentum).add_(center, alpha=1 - self.momentum)

    def forward(self, tokens, dynamics, rank_difference, shapes, output_size, used_iterations=None):
        prototype = self.prototype.to(tokens.dtype)
        token_distance = (tokens - prototype).pow(2).mean(-1).sqrt()
        attractor_score = token_distance.mean(-1)
        if dynamics:
            stacked_dynamics = torch.stack(dynamics, dim=1)
            if used_iterations is None:
                dynamic_score = stacked_dynamics.mean(1)
            else:
                dynamic_score = stacked_dynamics.sum(1) / used_iterations.to(tokens).clamp_min(1)
        else:
            dynamic_score = torch.zeros_like(attractor_score)
        rank_score = rank_difference.abs().mean(dim=(1, 2))
        components = torch.stack([attractor_score, dynamic_score, rank_score], dim=-1)
        image_score = (components * self.score_weights.to(components)).sum(-1)
        maps, offset = [], 0
        for h, w in shapes:
            count = h * w
            level = token_distance[:, offset : offset + count].reshape(-1, 1, h, w)
            maps.append(F.interpolate(level, output_size, mode="bilinear", align_corners=False))
            offset += count
        anomaly_map = torch.stack(maps).mean(0)
        return image_score, anomaly_map, components

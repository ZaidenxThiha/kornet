from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AnomalyHead(nn.Module):
    def __init__(
        self,
        dim: int,
        score_weights=(1.0, 0.2, 0.1),
        momentum: float = 0.99,
        token_count: int | None = None,
        spatial_weight: float = 0.0,
        initial_residual_weight: float = 0.0,
        topk_fraction: float = 1.0,
        variance_floor: float = 1e-4,
    ):
        super().__init__()
        if not 0 <= spatial_weight <= 1:
            raise ValueError("spatial_weight must be between 0 and 1")
        if not 0 <= initial_residual_weight <= 1:
            raise ValueError("initial_residual_weight must be between 0 and 1")
        if not 0 < topk_fraction <= 1:
            raise ValueError("topk_fraction must be in (0, 1]")
        if spatial_weight and not token_count:
            raise ValueError("token_count is required when spatial_weight is enabled")
        self.register_buffer("prototype", torch.zeros(dim))
        self.register_buffer("prototype_initialized", torch.tensor(False))
        self.momentum = momentum
        self.spatial_weight = spatial_weight
        self.initial_residual_weight = initial_residual_weight
        self.topk_fraction = topk_fraction
        self.variance_floor = variance_floor
        self.register_buffer("score_weights", torch.tensor(score_weights, dtype=torch.float32))
        if spatial_weight:
            self.register_buffer("spatial_prototype", torch.zeros(token_count, dim))
            self.register_buffer("spatial_variance", torch.ones(token_count, dim))
        if initial_residual_weight:
            self.register_buffer("initial_prototype", torch.zeros(dim))
            if spatial_weight:
                self.register_buffer("initial_spatial_prototype", torch.zeros(token_count, dim))
                self.register_buffer("initial_spatial_variance", torch.ones(token_count, dim))

    @torch.no_grad()
    def update_prototype(
        self, tokens: torch.Tensor, initial_tokens: torch.Tensor | None = None
    ) -> None:
        center = tokens.detach().mean(dim=(0, 1))
        if not bool(self.prototype_initialized):
            self.prototype.copy_(center)
            if self.spatial_weight:
                self._initialize_spatial(tokens, "")
            if self.initial_residual_weight:
                initial = initial_tokens if initial_tokens is not None else tokens
                self.initial_prototype.copy_(initial.detach().mean(dim=(0, 1)))
                if self.spatial_weight:
                    self._initialize_spatial(initial, "initial_")
            self.prototype_initialized.fill_(True)
        else:
            self.prototype.mul_(self.momentum).add_(center, alpha=1 - self.momentum)
            if self.spatial_weight:
                self._update_spatial(tokens, "")
            if self.initial_residual_weight:
                initial = initial_tokens if initial_tokens is not None else tokens
                initial_center = initial.detach().mean(dim=(0, 1))
                self.initial_prototype.mul_(self.momentum).add_(
                    initial_center, alpha=1 - self.momentum
                )
                if self.spatial_weight:
                    self._update_spatial(initial, "initial_")

    def _initialize_spatial(self, tokens: torch.Tensor, prefix: str) -> None:
        detached = tokens.detach()
        getattr(self, f"{prefix}spatial_prototype").copy_(detached.mean(0))
        getattr(self, f"{prefix}spatial_variance").copy_(
            detached.var(0, unbiased=False).clamp_min(self.variance_floor)
        )

    def _update_spatial(self, tokens: torch.Tensor, prefix: str) -> None:
        detached = tokens.detach()
        mean = detached.mean(0)
        variance = detached.var(0, unbiased=False).clamp_min(self.variance_floor)
        getattr(self, f"{prefix}spatial_prototype").mul_(self.momentum).add_(
            mean, alpha=1 - self.momentum
        )
        getattr(self, f"{prefix}spatial_variance").mul_(self.momentum).add_(
            variance, alpha=1 - self.momentum
        )

    def _distance(self, tokens: torch.Tensor, prefix: str = "") -> torch.Tensor:
        prototype = getattr(self, f"{prefix}prototype").to(tokens)
        global_distance = (tokens - prototype).pow(2).mean(-1).sqrt()
        if not self.spatial_weight:
            return global_distance
        spatial_prototype = getattr(self, f"{prefix}spatial_prototype").to(tokens)
        spatial_variance = getattr(self, f"{prefix}spatial_variance").to(tokens)
        spatial_distance = (
            ((tokens - spatial_prototype).pow(2) / spatial_variance.clamp_min(self.variance_floor))
            .mean(-1)
            .sqrt()
        )
        return (1 - self.spatial_weight) * global_distance + self.spatial_weight * spatial_distance

    def forward(
        self,
        tokens,
        dynamics,
        rank_difference,
        shapes,
        output_size,
        used_iterations=None,
        initial_tokens=None,
    ):
        token_distance = self._distance(tokens)
        if self.initial_residual_weight:
            initial = initial_tokens if initial_tokens is not None else tokens
            initial_distance = self._distance(initial, "initial_")
            token_distance = (
                self.initial_residual_weight * initial_distance
                + (1 - self.initial_residual_weight) * token_distance
            )
        topk = max(1, round(token_distance.shape[1] * self.topk_fraction))
        attractor_score = token_distance.topk(topk, dim=1).values.mean(1)
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

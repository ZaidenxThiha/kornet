from __future__ import annotations

import torch
from torch import nn

from .anomaly_head import AnomalyHead
from .backbone import FeatureBackbone, MultiScaleTokenizer
from .kor_operator import KORBlock


class KORNet(nn.Module):
    def __init__(
        self,
        backbone="resnet18",
        pretrained=True,
        feature_dim=256,
        rank_heads=4,
        sort_mode="soft",
        sort_temperature=0.1,
        sinkhorn_iterations=5,
        max_tokens=256,
        max_iterations=7,
        train_iterations=4,
        min_iterations=1,
        adaptive_stop=True,
        convergence_threshold=1e-3,
        convergence_patience=2,
        dropout=0.1,
        multi_scale=True,
        ranking="learned",
        opposing_subtraction=True,
        score_weights=(1.0, 0.2, 0.1),
        spatial_weight=0.0,
        initial_residual_weight=0.0,
        topk_fraction=1.0,
        variance_floor=1e-4,
    ) -> None:
        super().__init__()
        self.backbone = FeatureBackbone(backbone, pretrained)
        self.tokenizer = MultiScaleTokenizer(
            self.backbone.channels, feature_dim, max_tokens, multi_scale
        )
        self.kor = KORBlock(
            feature_dim,
            rank_heads,
            sort_temperature,
            sort_mode,
            dropout,
            ranking,
            opposing_subtraction,
            sinkhorn_iterations,
        )
        self.anomaly_head = AnomalyHead(
            feature_dim,
            score_weights,
            token_count=self.tokenizer.token_count,
            spatial_weight=spatial_weight,
            initial_residual_weight=initial_residual_weight,
            topk_fraction=topk_fraction,
            variance_floor=variance_floor,
        )
        self.max_iterations = max_iterations
        self.train_iterations = train_iterations
        self.min_iterations = min_iterations
        self.adaptive_stop = adaptive_stop
        self.convergence_threshold = convergence_threshold
        self.convergence_patience = convergence_patience
        self.feature_dim = feature_dim

    def forward(
        self,
        images: torch.Tensor,
        adaptive: bool | None = None,
        iterations: int | None = None,
        sort_mode: str | None = None,
    ) -> dict[str, torch.Tensor | list]:
        tokens, shapes = self.tokenizer(self.backbone(images))
        initial = tokens
        use_adaptive = (
            self.adaptive_stop and not self.training if adaptive is None else bool(adaptive)
        )
        limit = iterations or (self.train_iterations if self.training else self.max_iterations)
        limit = min(limit, self.max_iterations)
        batch = images.shape[0]
        active = torch.ones(batch, dtype=torch.bool, device=images.device)
        stable_count = torch.zeros(batch, dtype=torch.long, device=images.device)
        used_iterations = torch.full((batch,), limit, dtype=torch.long, device=images.device)
        dynamics, states, rank_history, gates = [], [tokens], [], []
        last_diag = None
        final_rank_difference = torch.zeros_like(tokens)
        eps = torch.finfo(tokens.dtype).eps
        for step in range(limit):
            candidate, diag = self.kor(tokens, sort_mode)
            delta = (candidate - tokens).flatten(1).norm(dim=1) / (
                tokens.flatten(1).norm(dim=1) + eps
            )
            if use_adaptive:
                candidate = torch.where(active[:, None, None], candidate, tokens)
                delta = torch.where(active, delta, torch.zeros_like(delta))
            tokens = candidate
            dynamics.append(delta)
            states.append(tokens)
            rank_history.append(diag["rank_scores"])
            gates.append(diag["gate"])
            last_diag = diag
            if use_adaptive:
                final_rank_difference = torch.where(
                    active[:, None, None], diag["rank_difference"], final_rank_difference
                )
            else:
                final_rank_difference = diag["rank_difference"]
            if use_adaptive and step + 1 >= self.min_iterations:
                stable_count = torch.where(
                    active & (delta < self.convergence_threshold),
                    stable_count + 1,
                    torch.zeros_like(stable_count),
                )
                newly_done = active & (stable_count >= self.convergence_patience)
                used_iterations = torch.where(newly_done, step + 1, used_iterations)
                active = active & ~newly_done
                if not bool(active.any()):
                    break
        assert last_diag is not None
        image_score, anomaly_map, score_components = self.anomaly_head(
            tokens,
            dynamics,
            final_rank_difference,
            shapes,
            images.shape[-2:],
            used_iterations,
            initial,
        )
        stability_next, _ = self.kor(tokens, sort_mode)
        return {
            "image_score": image_score,
            "anomaly_map": anomaly_map,
            "score_components": score_components,
            "tokens": tokens,
            "initial_tokens": initial,
            "states": states,
            "deltas": torch.stack(dynamics, dim=1),
            "iterations": used_iterations,
            "rank_scores": rank_history,
            "gates": torch.stack(gates, dim=1),
            "stability_delta": stability_next - tokens,
            "token_shapes": shapes,
        }

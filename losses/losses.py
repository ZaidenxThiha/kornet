from __future__ import annotations

from itertools import pairwise

import torch
from torch import nn


class KORNetLoss(nn.Module):
    """Normal-only objective with a hinge variance term to discourage collapse."""

    def __init__(self, compactness=1.0, convergence=0.05, stability=0.05, variance=0.01):
        super().__init__()
        self.weights = {
            "compactness": compactness,
            "convergence": convergence,
            "stability": stability,
            "variance": variance,
        }

    def forward(
        self,
        output: dict,
        prototype: torch.Tensor,
        prototype_initialized: bool = True,
    ) -> dict[str, torch.Tensor]:
        tokens = output["tokens"]
        compactness = (
            (tokens - prototype.to(tokens)).pow(2).mean()
            if prototype_initialized
            else tokens.new_zeros(())
        )
        states = output["states"]
        late_transitions = [(b - a).pow(2).mean() for a, b in pairwise(states)]
        convergence = (
            torch.stack(late_transitions[len(late_transitions) // 2 :]).mean()
            if late_transitions
            else tokens.new_zeros(())
        )
        stability = output["stability_delta"].pow(2).mean()
        feature_std = tokens.flatten(0, 1).std(dim=0, unbiased=False)
        variance = torch.relu(0.1 - feature_std).mean()
        losses = {
            "compactness": compactness,
            "convergence": convergence,
            "stability": stability,
            "variance": variance,
            "feature_std": feature_std.mean().detach(),
        }
        losses["total"] = sum(self.weights[name] * losses[name] for name in self.weights)
        return losses

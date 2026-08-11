from __future__ import annotations

import copy

import torch
from torch import nn

from .kornet import KORNet


class CNNOnly(KORNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.kor

    def forward(self, images: torch.Tensor, **kwargs):
        tokens, shapes = self.tokenizer(self.backbone(images))
        zeros = tokens.new_zeros(images.shape[0])
        rank_difference = tokens.new_zeros(tokens.shape)
        score, anomaly_map, components = self.anomaly_head(
            tokens, [], rank_difference, shapes, images.shape[-2:]
        )
        return {
            "image_score": score,
            "anomaly_map": anomaly_map,
            "score_components": components,
            "tokens": tokens,
            "initial_tokens": tokens,
            "states": [tokens],
            "deltas": zeros[:, None],
            "iterations": torch.zeros(images.shape[0], dtype=torch.long, device=images.device),
            "rank_scores": [],
            "gates": zeros[:, None],
            "stability_delta": torch.zeros_like(tokens),
            "token_shapes": shapes,
        }


class AttentionBaseline(KORNet):
    def __init__(self, *args, rank_heads=4, feature_dim=256, **kwargs):
        super().__init__(*args, rank_heads=rank_heads, feature_dim=feature_dim, **kwargs)
        valid_heads = rank_heads if feature_dim % rank_heads == 0 else 1
        self.attention = nn.MultiheadAttention(feature_dim, valid_heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(feature_dim)
        del self.kor

    def forward(self, images: torch.Tensor, **kwargs):
        tokens, shapes = self.tokenizer(self.backbone(images))
        update, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        final = self.attention_norm(tokens + update)
        delta = (final - tokens).flatten(1).norm(dim=1) / (tokens.flatten(1).norm(dim=1) + 1e-8)
        rank_difference = final - tokens
        score, anomaly_map, components = self.anomaly_head(
            final, [delta], rank_difference, shapes, images.shape[-2:]
        )
        return {
            "image_score": score,
            "anomaly_map": anomaly_map,
            "score_components": components,
            "tokens": final,
            "initial_tokens": tokens,
            "states": [tokens, final],
            "deltas": delta[:, None],
            "iterations": torch.ones_like(delta, dtype=torch.long),
            "rank_scores": [],
            "gates": torch.ones_like(delta)[:, None],
            "stability_delta": torch.zeros_like(final),
            "token_shapes": shapes,
        }


def build_model(config: dict) -> nn.Module:
    options = copy.deepcopy(config)
    variant = options.pop("variant", "kornet_adaptive")
    if variant == "cnn":
        return CNNOnly(**options)
    if variant == "attention":
        return AttentionBaseline(**options)
    if variant in {"kor_single", "kornet_fixed", "kornet_adaptive"}:
        if variant == "kor_single":
            options.update(max_iterations=1, train_iterations=1, adaptive_stop=False)
        elif variant == "kornet_fixed":
            options["adaptive_stop"] = False
        return KORNet(**options)
    raise ValueError(f"Unknown model variant: {variant}")

from __future__ import annotations

import warnings
from typing import ClassVar

import torch
from torch import nn


class FeatureBackbone(nn.Module):
    """Torchvision backbone exposing three spatial feature levels."""

    CHANNELS: ClassVar = {
        "resnet18": (128, 256, 512),
        "efficientnet_b0": (40, 112, 320),
        "convnext_tiny": (192, 384, 768),
    }

    def __init__(self, name: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()
        if name not in self.CHANNELS:
            raise ValueError(f"Unknown backbone {name!r}; choose from {sorted(self.CHANNELS)}")
        self.name = name
        self.channels = self.CHANNELS[name]
        self.body = self._build(name, pretrained)

    @staticmethod
    def _build(name: str, pretrained: bool):
        from torchvision import models

        try:
            if name == "resnet18":
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                model = models.resnet18(weights=weights)
                return nn.ModuleList(
                    [
                        nn.Sequential(
                            model.conv1, model.bn1, model.relu, model.maxpool, model.layer1
                        ),
                        model.layer2,
                        model.layer3,
                        model.layer4,
                    ]
                )
            if name == "efficientnet_b0":
                weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                model = models.efficientnet_b0(weights=weights)
                return model.features
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            model = models.convnext_tiny(weights=weights)
            return model.features
        except Exception as exc:
            if not pretrained:
                raise
            warnings.warn(
                f"Could not load pretrained {name} weights ({exc}); using random weights."
            )
            return FeatureBackbone._build(name, False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.name == "resnet18":
            x = self.body[0](x)
            f1 = self.body[1](x)
            f2 = self.body[2](f1)
            f3 = self.body[3](f2)
            return [f1, f2, f3]
        wanted = (3, 5, 7)
        outputs = []
        for index, layer in enumerate(self.body):
            x = layer(x)
            if index in wanted:
                outputs.append(x)
        return outputs


class MultiScaleTokenizer(nn.Module):
    def __init__(
        self, channels: tuple[int, ...], dim: int, max_tokens: int, multi_scale: bool = True
    ):
        super().__init__()
        selected = channels if multi_scale else channels[-1:]
        self.feature_indices = list(range(len(channels))) if multi_scale else [-1]
        self.projections = nn.ModuleList(nn.Conv2d(c, dim, 1) for c in selected)
        self.max_tokens = max_tokens
        per_level = max(1, max_tokens // len(selected))
        self.target_side = max(1, int(per_level**0.5))
        self.token_count = len(selected) * self.target_side**2

    def forward(self, features: list[torch.Tensor]):
        chosen = [features[i] for i in self.feature_indices]
        tokens, shapes = [], []
        for projection, feature in zip(self.projections, chosen):
            x = projection(feature)
            h = w = self.target_side
            if x.device.type == "mps" and (x.shape[-2] % h != 0 or x.shape[-1] % w != 0):
                # MPS does not implement non-divisible adaptive pooling. Device-copy
                # operations retain autograd, preserving the exact pooling semantics.
                x = nn.functional.adaptive_avg_pool2d(x.cpu(), (h, w)).to(x.device)
            else:
                x = nn.functional.adaptive_avg_pool2d(x, (h, w))
            shapes.append((h, w))
            tokens.append(x.flatten(2).transpose(1, 2))
        return torch.cat(tokens, dim=1), shapes

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import torch
from torch.nn import functional as F

PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class InferenceProtocol:
    sort_mode: str = "hard"
    adaptive: bool = False
    gaussian_sigma: float = 0.0
    version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.sort_mode not in {"soft", "hard"}:
            raise ValueError("Inference sort_mode must be 'soft' or 'hard'")
        if self.gaussian_sigma < 0:
            raise ValueError("gaussian_sigma must be non-negative")

    def as_dict(self) -> dict:
        return asdict(self)


def resolve_inference_protocol(
    config: dict,
    *,
    sort_mode: str | None = None,
    adaptive: bool | None = None,
    gaussian_sigma: float | None = None,
) -> InferenceProtocol:
    evaluation = config.get("evaluation", {})
    if evaluation.get("threshold_source", "validation") != "validation":
        raise ValueError("Only validation-derived threshold calibration is supported")
    protocol = InferenceProtocol(
        sort_mode=evaluation.get("sort_mode", "hard"),
        adaptive=bool(evaluation.get("adaptive", False)),
        gaussian_sigma=float(evaluation.get("gaussian_sigma", 0.0)),
    )
    updates = {}
    if sort_mode is not None:
        updates["sort_mode"] = sort_mode
    if adaptive is not None:
        updates["adaptive"] = adaptive
    if gaussian_sigma is not None:
        updates["gaussian_sigma"] = gaussian_sigma
    return replace(protocol, **updates)


def protocol_matches(metrics: dict, protocol: InferenceProtocol) -> bool:
    return metrics.get("protocol", {}).get("inference") == protocol.as_dict()


def calibrated_image_threshold(metrics: dict, protocol: InferenceProtocol) -> float | None:
    if not protocol_matches(metrics, protocol):
        return None
    return metrics.get("image", {}).get("threshold")


def gaussian_smooth(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply exportable Gaussian smoothing to BCHW anomaly maps."""
    if sigma <= 0:
        return maps
    radius = max(1, int(4.0 * sigma + 0.5))
    coordinates = torch.arange(-radius, radius + 1, device=maps.device, dtype=maps.dtype)
    kernel_1d = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    channels = maps.shape[1]
    horizontal = kernel_1d.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel_1d.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    padded = F.pad(maps, (radius, radius, 0, 0), mode="reflect")
    smoothed = F.conv2d(padded, horizontal, groups=channels)
    padded = F.pad(smoothed, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(padded, vertical, groups=channels)

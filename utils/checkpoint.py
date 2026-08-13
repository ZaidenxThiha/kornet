from __future__ import annotations

from pathlib import Path

import torch


def load_checkpoint_payload(path, map_location="cpu") -> dict:
    """Load state-dict checkpoints without allowing arbitrary pickle globals."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Checkpoint must be a dictionary containing a model state")
    return checkpoint


def confined_checkpoint(path, roots) -> Path:
    candidate = Path(path).resolve()
    allowed = [Path(root).resolve() for root in roots]
    if not any(candidate == root or root in candidate.parents for root in allowed):
        raise ValueError("Checkpoint path is outside the configured checkpoint roots")
    if candidate.suffix not in {".pt", ".pth"} or not candidate.is_file():
        raise ValueError("Checkpoint must be an existing .pt or .pth file")
    return candidate


def save_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    epoch=0,
    best_metric=None,
    config=None,
    metric_name=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "metric_name": metric_name,
        "config": config,
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
    }
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu", strict=True):
    checkpoint = load_checkpoint_payload(path, map_location)
    model.load_state_dict(checkpoint["model"], strict=strict)
    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    path, model, optimizer=None, scheduler=None, epoch=0, best_metric=None, config=None
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config,
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
    }
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu", strict=True):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=strict)
    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint

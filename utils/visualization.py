from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
    low, high = float(anomaly_map.min()), float(anomaly_map.max())
    return (anomaly_map - low) / (high - low + 1e-8)


def heatmap_overlay(image: np.ndarray, anomaly_map: np.ndarray, alpha=0.45) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.max() > 1:
        image /= 255.0
    heat = plt.get_cmap("turbo")(normalize_map(anomaly_map))[..., :3]
    return np.clip((1 - alpha) * image + alpha * heat, 0, 1)


def save_heatmap(image, anomaly_map, output_path, title=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = heatmap_overlay(image, anomaly_map)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[1].imshow(anomaly_map, cmap="turbo")
    axes[1].set_title("Anomaly map")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_training(history: list[dict], output_path):
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(7, 4))
    for key in ("train_loss", "val_score_mean", "feature_std"):
        if key in history[0]:
            axis.plot(epochs, [row.get(key, np.nan) for row in history], label=key)
    axis.set_xlabel("Epoch")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

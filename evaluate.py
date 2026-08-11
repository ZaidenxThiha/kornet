from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_datasets
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.config import load_config, save_json
from utils.metrics import (
    calibrate_threshold,
    classification_metrics,
    pixel_metrics,
    summarize_iterations,
)
from utils.profiling import estimate_macs, model_size_mb, parameter_count


@torch.inference_mode()
def collect(model, loader, device, adaptive=True, sort_mode=None, gaussian_sigma=0.0):
    scores, labels, maps, masks, iterations, latencies = [], [], [], [], [], []
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch in tqdm(loader, desc="evaluate", leave=False):
        images = batch["image"].to(device)
        start = time.perf_counter()
        output = model(images, adaptive=adaptive, sort_mode=sort_mode)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - start) * 1000 / images.shape[0]
        batch_maps = output["anomaly_map"].squeeze(1).cpu().numpy()
        if gaussian_sigma > 0:
            batch_maps = np.stack([gaussian_filter(item, gaussian_sigma) for item in batch_maps])
        scores.extend(output["image_score"].cpu().tolist())
        labels.extend(batch["label"].cpu().tolist())
        maps.extend(batch_maps)
        masks.extend(batch["mask"].squeeze(1).cpu().numpy())
        iterations.extend(output["iterations"].cpu().tolist())
        latencies.extend([elapsed] * images.shape[0])
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
    )
    return {
        "scores": scores,
        "labels": labels,
        "maps": maps,
        "masks": masks,
        "iterations": iterations,
        "latencies": latencies,
        "peak_gpu_memory_mb": peak_memory,
    }


def evaluate(checkpoint_path, config_path=None, output_path=None, sort_mode=None, fixed=False):
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = load_config(config_path) if config_path else raw.get("config")
    if not config:
        raise ValueError("Config is absent from checkpoint; pass --config")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = build_datasets(config["dataset"])
    batch_size = config["training"].get("batch_size", 16)
    workers = config["dataset"].get("num_workers", 4)
    loaders = {
        name: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
        )
        for name, ds in datasets.items()
        if name in {"val", "test"}
    }
    model = build_model(config["model"]).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    sigma = config.get("evaluation", {}).get("gaussian_sigma", 0.0)
    validation = collect(model, loaders["val"], device, adaptive=False, sort_mode=sort_mode)
    threshold = calibrate_threshold(validation["scores"])
    pixel_threshold = calibrate_threshold(np.asarray(validation["maps"]).reshape(-1))
    test = collect(
        model,
        loaders["test"],
        device,
        adaptive=not fixed,
        sort_mode=sort_mode,
        gaussian_sigma=sigma,
    )
    profile_sample = next(iter(loaders["val"]))["image"][:1].to(device)
    macs = estimate_macs(model, profile_sample)
    metrics = {
        "model": config["model"].get("variant", "kornet_adaptive"),
        "dataset": config["dataset"]["name"],
        "category": config["dataset"]["category"],
        "image": classification_metrics(test["labels"], test["scores"], threshold),
        "pixel": pixel_metrics(test["masks"], test["maps"], pixel_threshold),
        "efficiency": {
            "parameters": parameter_count(model),
            "model_size_mb": model_size_mb(model),
            "latency_ms_mean": float(np.mean(test["latencies"])),
            "latency_device": device.type,
            "peak_gpu_memory_mb": test["peak_gpu_memory_mb"],
            "macs": macs,
            "flops_estimate": 2 * macs if macs is not None else None,
            **summarize_iterations(test["iterations"]),
            "sort_mode": sort_mode or config["model"].get("sort_mode", "soft"),
        },
        "protocol": {
            "threshold": "99th percentile of held-out normal validation scores",
            "test_used_for_calibration": False,
            "gaussian_sigma": sigma,
        },
    }
    output_path = Path(output_path or Path(checkpoint_path).with_name("metrics.json"))
    save_json(metrics, output_path)
    print(f"Metrics written to {output_path.resolve()}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--sort-mode", choices=["soft", "hard"])
    parser.add_argument("--fixed", action="store_true", help="Disable adaptive stopping")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.config, args.output, args.sort_mode, args.fixed)


if __name__ == "__main__":
    main()

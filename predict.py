from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from datasets.transforms import build_transform
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.device import default_device, synchronize
from utils.visualization import save_heatmap


@torch.inference_mode()
def predict(checkpoint, image_path, output_dir="outputs", threshold=None, sort_mode="hard"):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    if threshold is None:
        metrics_path = Path(checkpoint).with_name("metrics.json")
        if metrics_path.exists():
            threshold = json.loads(metrics_path.read_text()).get("image", {}).get("threshold")
    device = default_device()
    model = build_model(config["model"]).to(device).eval()
    load_checkpoint(checkpoint, model, map_location=device)
    original = Image.open(image_path).convert("RGB")
    size = config["dataset"].get("image_size", 256)
    tensor, _ = build_transform(size)(original)
    start = time.perf_counter()
    output = model(tensor[None].to(device), sort_mode=sort_mode)
    synchronize(device)
    latency = (time.perf_counter() - start) * 1000
    score = float(output["image_score"][0])
    anomaly_map = output["anomaly_map"][0, 0].cpu().numpy()
    resized = np.asarray(original.resize((size, size)))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization = output_dir / f"{Path(image_path).stem}_prediction.png"
    save_heatmap(resized, anomaly_map, visualization, f"score={score:.4f}")
    result = {
        "image": str(Path(image_path).resolve()),
        "score": score,
        "status": "DEFECT"
        if threshold is not None and score >= threshold
        else ("NORMAL" if threshold is not None else "UNCALIBRATED"),
        "threshold": threshold,
        "iterations": int(output["iterations"][0]),
        "deltas": output["deltas"][0].cpu().tolist(),
        "latency_ms": latency,
        "visualization": str(visualization.resolve()),
    }
    (output_dir / f"{Path(image_path).stem}_prediction.json").write_text(
        json.dumps(result, indent=2)
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--sort-mode", choices=["soft", "hard"], default="hard")
    args = parser.parse_args()
    print(
        json.dumps(
            predict(args.checkpoint, args.image, args.output_dir, args.threshold, args.sort_mode),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

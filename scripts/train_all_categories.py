from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import train
from utils.config import load_config

CATEGORIES = {
    "mvtec": [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ],
    "visa": [
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--categories", nargs="*")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    args = parser.parse_args()
    base = load_config(args.config)
    categories = args.categories or CATEGORIES.get(base["dataset"]["name"])
    if not categories:
        raise ValueError("Supply --categories for this dataset")
    for category in categories:
        for seed in args.seeds:
            config = copy.deepcopy(base)
            config["dataset"].update(category=category, seed=seed)
            config["seed"] = seed
            config["experiment_name"] = (
                f"{config['model']['variant']}_{config['dataset']['name']}_{category}"
            )
            train(config)


if __name__ == "__main__":
    main()

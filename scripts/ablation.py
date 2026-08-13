from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import train
from utils.config import load_config

ABLATIONS = {
    "no_opposing": {"model.opposing_subtraction": False},
    "no_recursion": {
        "model.max_iterations": 1,
        "model.train_iterations": 1,
        "model.adaptive_stop": False,
    },
    "heads_1": {"model.rank_heads": 1},
    "heads_2": {"model.rank_heads": 2},
    "heads_4": {"model.rank_heads": 4},
    "heads_8": {"model.rank_heads": 8},
    "iterations_1": {"model.max_iterations": 1, "model.train_iterations": 1},
    "iterations_2": {"model.max_iterations": 2, "model.train_iterations": 2},
    "iterations_3": {"model.max_iterations": 3, "model.train_iterations": 3},
    "iterations_5": {"model.max_iterations": 5, "model.train_iterations": 4},
    "iterations_7": {"model.max_iterations": 7, "model.train_iterations": 4},
    "iterations_10": {"model.max_iterations": 10, "model.train_iterations": 5},
    "fixed": {"model.adaptive_stop": False},
    "soft_sort": {"evaluation.sort_mode": "soft"},
    "single_scale": {"model.multi_scale": False},
    "magnitude_ranking": {"model.ranking": "magnitude"},
    "no_convergence_loss": {"loss.convergence": 0.0},
}


def set_nested(config, dotted_key, value):
    parent = config
    pieces = dotted_key.split(".")
    for piece in pieces[:-1]:
        parent = parent[piece]
    parent[pieces[-1]] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiments", nargs="*", choices=sorted(ABLATIONS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    base = load_config(args.config)
    experiments = args.experiments or list(ABLATIONS)
    manifest = []
    for name in experiments:
        config = copy.deepcopy(base)
        for key, value in ABLATIONS[name].items():
            set_nested(config, key, value)
        config["experiment_name"] = (
            f"ablation_{name}_{config['dataset']['name']}_{config['dataset']['category']}"
        )
        manifest.append(
            {"name": name, "changes": ABLATIONS[name], "experiment_name": config["experiment_name"]}
        )
        if not args.dry_run:
            train(config)
    Path("runs").mkdir(exist_ok=True)
    Path("runs/ablation_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

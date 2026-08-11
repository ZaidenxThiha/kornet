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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna is optional. Install it with: pip install -e '.[research]'"
        ) from exc
    base = load_config(args.config)

    def objective(trial):
        config = copy.deepcopy(base)
        config["experiment_name"] = f"optuna_trial_{trial.number}"
        config["training"]["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-5, 5e-4, log=True
        )
        config["model"]["feature_dim"] = trial.suggest_categorical("feature_dim", [128, 192, 256])
        config["model"]["rank_heads"] = trial.suggest_categorical("rank_heads", [1, 2, 4, 8])
        config["model"]["sort_temperature"] = trial.suggest_float(
            "sort_temperature", 0.03, 0.5, log=True
        )
        config["model"]["train_iterations"] = trial.suggest_int("train_iterations", 1, 7)
        config["model"]["max_iterations"] = max(config["model"]["train_iterations"], 7)
        config["model"]["convergence_threshold"] = trial.suggest_float(
            "convergence_threshold", 1e-4, 1e-2, log=True
        )
        config["model"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
        config["model"]["max_tokens"] = trial.suggest_categorical("max_tokens", [64, 144, 256])
        config["loss"]["convergence"] = trial.suggest_float("convergence_weight", 0.0, 0.2)
        checkpoint = train(config)
        state = __import__("torch").load(checkpoint, map_location="cpu", weights_only=False)
        return -float(state["best_metric"])  # validation-normal objective; final test is never read

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)
    output = Path(base.get("output_dir", "runs")) / "best_params.json"
    output.write_text(json.dumps(study.best_params, indent=2))
    print(f"Best validation-only parameters written to {output}")


if __name__ == "__main__":
    main()

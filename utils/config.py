from __future__ import annotations

import json
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    config.setdefault("dataset", {})["seed"] = config.get("seed", 42)
    return config


def save_config(config: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def save_json(data: dict | list, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)

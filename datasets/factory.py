from __future__ import annotations

from .mvtec import build_mvtec
from .mvtec_ad2 import build_mvtec_ad2
from .visa import build_visa


def build_datasets(config: dict):
    builders = {"mvtec": build_mvtec, "visa": build_visa, "mvtec_ad2": build_mvtec_ad2}
    name = config["name"]
    if name not in builders:
        raise ValueError(f"Unsupported dataset {name!r}; choose from {sorted(builders)}")
    return builders[name](
        config["root"],
        config["category"],
        int(config.get("image_size", 256)),
        float(config.get("val_fraction", 0.1)),
        int(config.get("seed", 42)),
    )

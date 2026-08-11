from pathlib import Path

import numpy as np
from PIL import Image

from datasets.mvtec import build_mvtec


def save_image(path: Path, value=128):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((24, 24, 3), value, dtype=np.uint8)).save(path)


def test_mvtec_loader_splits_and_masks(tmp_path):
    root = tmp_path / "mvtec" / "bottle"
    for index in range(5):
        save_image(root / "train" / "good" / f"{index}.png")
    save_image(root / "test" / "good" / "normal.png")
    save_image(root / "test" / "scratch" / "bad.png", 200)
    mask = root / "ground_truth" / "scratch" / "bad_mask.png"
    mask.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((24, 24), 255, dtype=np.uint8)).save(mask)
    datasets = build_mvtec(tmp_path / "mvtec", "bottle", 32, val_fraction=0.2, seed=7)
    assert len(datasets["train"]) == 4
    assert len(datasets["val"]) == 1
    assert len(datasets["test"]) == 2
    anomalous = datasets["test"][1]
    assert anomalous["image"].shape == (3, 32, 32)
    assert anomalous["mask"].sum() == 32 * 32
    assert anomalous["label"] == 1

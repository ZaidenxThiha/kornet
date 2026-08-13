from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from datasets.common import AnomalyDataset, Sample, normal_train_val_split, safe_path
from datasets.mvtec import build_mvtec
from datasets.transforms import build_transform


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
    assert anomalous["mask_valid"]


def test_missing_anomaly_mask_is_explicitly_invalid(tmp_path):
    image = tmp_path / "bad.png"
    save_image(image)
    dataset = AnomalyDataset(
        [Sample(image, 1, "scratch", tmp_path / "missing.png")],
        build_transform(16),
        "bottle",
        "test",
    )
    sample = dataset[0]
    assert not sample["mask_valid"]
    assert sample["mask"].sum() == 0


def test_train_val_split_rejects_overlap_edge_cases():
    samples = [Sample(Path(f"{index}.png"), 0, "good") for index in range(4)]
    train, val = normal_train_val_split(samples, 0.25, 42)
    assert set(train).isdisjoint(val)
    with pytest.raises(ValueError, match="strictly between"):
        normal_train_val_split(samples, 0, 42)
    with pytest.raises(ValueError, match="At least two"):
        normal_train_val_split(samples[:1], 0.5, 42)


def test_dataset_paths_cannot_escape_root(tmp_path):
    assert safe_path(tmp_path, "bottle", "category") == (tmp_path / "bottle").resolve()
    with pytest.raises(ValueError, match="escapes"):
        safe_path(tmp_path, "../secret", "category")
    with pytest.raises(ValueError, match="relative"):
        safe_path(tmp_path, "/tmp/secret", "VisA CSV image")

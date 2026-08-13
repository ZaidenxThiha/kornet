from __future__ import annotations

import csv
from pathlib import Path

from .common import AnomalyDataset, Sample, image_files, normal_train_val_split, safe_path
from .transforms import build_transform


def _from_split_csv(root: Path, category: str) -> tuple[list[Sample], list[Sample]] | None:
    candidates = [root / "split_csv" / "1cls.csv", root / "split_csv" / "2cls_fewshot.csv"]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        return None
    train, test = [], []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("object") != category:
                continue
            image = safe_path(root, row["image"], "VisA CSV image")
            label = int(row.get("label", "normal") not in {"normal", "good", "0"})
            mask_value = row.get("mask", "")
            sample = Sample(
                image,
                label,
                "anomaly" if label else "good",
                safe_path(root, mask_value, "VisA CSV mask") if mask_value else None,
            )
            (train if row.get("split", "test") == "train" else test).append(sample)
    return train, test


def build_visa(root: str | Path, category: str, image_size: int, val_fraction=0.1, seed=42):
    root = Path(root).resolve()
    safe_path(root, category, "category")
    parsed = _from_split_csv(root, category)
    if parsed is None:
        data_root = safe_path(root, Path(category) / "Data", "category")
        if not data_root.exists():
            raise FileNotFoundError(
                f"VisA not found at {root.resolve()}. Download it under the terms of its official "
                "release, preserving split_csv/1cls.csv or <category>/Data."
            )
        normal = [Sample(p, 0, "good") for p in image_files(data_root / "Images" / "Normal")]
        anomaly_images = image_files(data_root / "Images" / "Anomaly")
        masks = {p.stem: p for p in image_files(data_root / "Masks" / "Anomaly")}
        parsed = (normal, [Sample(p, 1, "anomaly", masks.get(p.stem)) for p in anomaly_images])
    normal, anomalies = parsed
    normal = [s for s in normal if s.label == 0]
    train, val = normal_train_val_split(normal, val_fraction, seed)
    test_normals = [s for s in parsed[1] if s.label == 0]
    test = test_normals + [s for s in anomalies if s.label == 1]
    if not test_normals:
        # A folder-only release has no official split. Keep a deterministic normal holdout for test.
        train, held_out = normal_train_val_split(train, max(val_fraction, 0.1), seed + 1)
        test = held_out + test
    return {
        "train": AnomalyDataset(train, build_transform(image_size, True), category, "train"),
        "val": AnomalyDataset(val, build_transform(image_size), category, "val"),
        "test": AnomalyDataset(test, build_transform(image_size), category, "test"),
    }

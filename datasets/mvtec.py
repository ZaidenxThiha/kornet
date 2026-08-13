from __future__ import annotations

from pathlib import Path

from .common import AnomalyDataset, Sample, image_files, normal_train_val_split, safe_path
from .transforms import build_transform


def _test_samples(category_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    test_root = category_root / "test"
    for defect_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        defect = defect_dir.name
        for image in image_files(defect_dir):
            if defect == "good":
                samples.append(Sample(image, 0, defect))
            else:
                mask = category_root / "ground_truth" / defect / f"{image.stem}_mask.png"
                samples.append(Sample(image, 1, defect, mask))
    return samples


def build_mvtec(root: str | Path, category: str, image_size: int, val_fraction=0.1, seed=42):
    category_root = safe_path(root, category, "category")
    if not (category_root / "train" / "good").exists():
        raise FileNotFoundError(
            f"MVTec AD category not found at {category_root}. Place the official dataset under "
            f"{Path(root).resolve()}/<category>/train/good."
        )
    normal = [Sample(p, 0, "good") for p in image_files(category_root / "train" / "good")]
    train, val = normal_train_val_split(normal, val_fraction, seed)
    return {
        "train": AnomalyDataset(train, build_transform(image_size, True), category, "train"),
        "val": AnomalyDataset(val, build_transform(image_size), category, "val"),
        "test": AnomalyDataset(
            _test_samples(category_root), build_transform(image_size), category, "test"
        ),
    }

from __future__ import annotations

from pathlib import Path

from .common import AnomalyDataset, Sample, image_files, normal_train_val_split, safe_path
from .transforms import build_transform


def build_mvtec_ad2(root: str | Path, category: str, image_size: int, val_fraction=0.1, seed=42):
    category_root = safe_path(root, category, "category")
    train_root = category_root / "train" / "good"
    if not train_root.exists():
        raise FileNotFoundError(
            f"MVTec AD 2 not found at {category_root.resolve()}. Request/download it from the "
            "official MVTec AD 2 page and preserve the supplied directory structure."
        )
    normal = [Sample(p, 0, "good") for p in image_files(train_root)]
    train, val = normal_train_val_split(normal, val_fraction, seed)
    test: list[Sample] = []
    # The public release may use test_public and test_private; evaluation only uses available labels.
    for split_name in ("test", "test_public"):
        split_root = category_root / split_name
        if not split_root.exists():
            continue
        for defect_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            label = int(defect_dir.name != "good")
            for image in image_files(defect_dir):
                mask_candidates = list(
                    (category_root / "ground_truth" / defect_dir.name).glob(f"{image.stem}*")
                )
                test.append(
                    Sample(
                        image,
                        label,
                        defect_dir.name,
                        mask_candidates[0] if mask_candidates else None,
                    )
                )
    return {
        "train": AnomalyDataset(train, build_transform(image_size, True), category, "train"),
        "val": AnomalyDataset(val, build_transform(image_size), category, "val"),
        "test": AnomalyDataset(test, build_transform(image_size), category, "test"),
    }

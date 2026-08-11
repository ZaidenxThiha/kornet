from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    image: Path
    label: int
    defect_type: str
    mask: Path | None = None


class AnomalyDataset(Dataset):
    def __init__(self, samples: list[Sample], transform, category: str, split: str) -> None:
        if not samples:
            raise RuntimeError(f"No samples found for category={category!r}, split={split!r}")
        self.samples = samples
        self.transform = transform
        self.category = category
        self.split = split

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = Image.open(sample.image).convert("RGB")
        mask = (
            Image.open(sample.mask).convert("L") if sample.mask and sample.mask.exists() else None
        )
        image_t, mask_t = self.transform(image, mask)
        return {
            "image": image_t,
            "mask": mask_t,
            "label": sample.label,
            "path": str(sample.image),
            "category": self.category,
            "defect_type": sample.defect_type,
        }


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def normal_train_val_split(samples: list[Sample], val_fraction: float, seed: int):
    """Split normal training samples deterministically; test data is never touched."""
    import random

    items = samples.copy()
    random.Random(seed).shuffle(items)
    if len(items) < 2 or val_fraction <= 0:
        return items, items[: min(1, len(items))]
    n_val = max(1, round(len(items) * val_fraction))
    n_val = min(n_val, len(items) - 1)
    return items[n_val:], items[:n_val]

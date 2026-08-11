from __future__ import annotations

import random
from collections.abc import Callable

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class JointTransform:
    """Apply spatial operations consistently to an RGB image and a binary mask."""

    def __init__(self, size: int, train: bool = False, allow_flip: bool = True) -> None:
        self.size = size
        self.train = train
        self.allow_flip = allow_flip

    def __call__(self, image: Image.Image, mask: Image.Image | None = None):
        image = image.convert("RGB")
        mask = (mask or Image.new("L", image.size, 0)).convert("L")
        image = TF.resize(image, [self.size, self.size], InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.size, self.size], InterpolationMode.NEAREST)
        if self.train:
            if self.allow_flip and random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            angle = random.uniform(-5.0, 5.0)
            image = TF.rotate(image, angle, InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, InterpolationMode.NEAREST)
            image = TF.adjust_brightness(image, random.uniform(0.9, 1.1))
            image = TF.adjust_contrast(image, random.uniform(0.9, 1.1))
            image = TF.adjust_saturation(image, random.uniform(0.95, 1.05))
        image_t = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        mask_t = (TF.pil_to_tensor(mask).float() / 255.0 > 0.5).float()
        return image_t, mask_t


def build_transform(size: int, train: bool = False, allow_flip: bool = True) -> Callable:
    return JointTransform(size, train, allow_flip)


def denormalize(image: torch.Tensor) -> torch.Tensor:
    mean = image.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = image.new_tensor(IMAGENET_STD)[:, None, None]
    return (image * std + mean).clamp(0, 1)

"""ADE20K dataset for segmentation."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as T

log = logging.getLogger(__name__)

NUM_CLASSES = 150
IGNORE_LABEL = 255

ResizeMode = Literal["center_crop", "squish"]


def make_val_transform(
    size: int,
    mode: ResizeMode,
) -> Callable[[Image.Image, Image.Image], tuple[Tensor, Tensor]]:
    """Create joint (img, mask) transform for validation.

    Handles reduce_zero_label: 0→255 (ignore), 1-150→0-149 (classes).
    """
    if mode == "center_crop":
        img_transform = T.Compose([
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])
        mask_transform = T.Compose([
            T.Resize(size, T.InterpolationMode.NEAREST),
            T.CenterCrop(size),
        ])
    else:  # squish
        img_transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])
        mask_transform = T.Resize((size, size), T.InterpolationMode.NEAREST)

    def transform(img: Image.Image, mask: Image.Image) -> tuple[Tensor, Tensor]:
        img_t = img_transform(img)
        assert isinstance(img_t, Tensor)  # T.Compose typing is weak
        mask_t = torch.from_numpy(np.array(mask_transform(mask))).long()
        # reduce_zero_label: 0→255, 1-150→0-149
        mask_t = torch.where((mask_t >= 1) & (mask_t <= 150), mask_t - 1, IGNORE_LABEL)
        return img_t, mask_t

    return transform


class ADE20kDataset(Dataset[tuple[Tensor, Tensor]]):
    """ADE20K segmentation dataset."""

    def __init__(
        self,
        root: Path,
        split: str,
        transform: Callable[[Image.Image, Image.Image], tuple[Tensor, Tensor]],
    ) -> None:
        img_dir = root / "images" / split
        ann_dir = root / "annotations" / split
        self.imgs = sorted(img_dir.glob("*.jpg"))
        self.anns = [ann_dir / (p.stem + ".png") for p in self.imgs]
        self.transform = transform
        assert len(self.imgs) > 0, f"No images in {img_dir}"
        log.info(f"ADE20k {split}: {len(self.imgs)} images")

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        img = Image.open(self.imgs[i]).convert("RGB")
        mask = Image.open(self.anns[i])
        return self.transform(img, mask)

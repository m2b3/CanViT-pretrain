"""ADE20K dataset for segmentation."""

from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torch.utils.data import Dataset

NUM_CLASSES = 150
IGNORE_LABEL = 255
_MEAN = torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1)


class ADE20kDataset(Dataset[tuple[Tensor, Tensor]]):
    """ADE20K segmentation dataset."""

    def __init__(self, root: Path, split: str, size: int, augment: bool) -> None:
        self.size = size
        self.load_size = size * 2 if augment else size
        self.transform = A.Compose([A.HorizontalFlip(p=0.5), A.RandomCrop(size, size)]) if augment else None

        img_dir = root / "images" / split
        ann_dir = root / "annotations" / split
        self.imgs = sorted(img_dir.glob("*.jpg"))
        self.anns = [ann_dir / (p.stem + ".png") for p in self.imgs]
        assert len(self.imgs) > 0, f"No images in {img_dir}"

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        img = np.array(
            Image.open(self.imgs[i]).convert("RGB").resize((self.load_size, self.load_size), Image.Resampling.BILINEAR)
        )
        mask = np.array(Image.open(self.anns[i]).resize((self.load_size, self.load_size), Image.Resampling.NEAREST))

        if self.transform:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]

        img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img_t = (img_t - _MEAN) / _STD
        mask_t = torch.from_numpy(mask.astype(np.int64))
        valid = (mask_t >= 1) & (mask_t <= 150)
        return img_t, torch.where(valid, mask_t - 1, IGNORE_LABEL)

"""ADE20K dataset for segmentation."""

import logging
from pathlib import Path

import numpy as np
import torch
from dinov3.eval.segmentation.transforms import make_segmentation_train_transforms
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as T

from canvit_pretrain.train.transforms import val_transform

log = logging.getLogger(__name__)

NUM_CLASSES = 150
IGNORE_LABEL = 255


class ADE20kDataset(Dataset[tuple[Tensor, Tensor]]):
    """ADE20K segmentation dataset."""

    def __init__(
        self,
        root: Path,
        split: str,
        size: int,
        *,
        augment: bool,
        aug_scale_range: tuple[float, float],
        aug_flip_prob: float,
    ) -> None:
        self.size = size
        self._augment = augment
        self._img_transform = val_transform(size) if not augment else None
        self._mask_spatial = T.Compose([T.Resize(size, T.InterpolationMode.NEAREST), T.CenterCrop(size)])

        if augment:
            self.transform = make_segmentation_train_transforms(
                img_size=size,
                random_img_size_ratio_range=list(aug_scale_range),
                crop_size=(size, size),
                flip_prob=aug_flip_prob,
                reduce_zero_label=True,
            )

        img_dir = root / "images" / split
        ann_dir = root / "annotations" / split
        self.imgs = sorted(img_dir.glob("*.jpg"))
        self.anns = [ann_dir / (p.stem + ".png") for p in self.imgs]

        assert len(self.imgs) > 0, f"No images in {img_dir}"
        log.info(f"ADE20k {split}: {len(self.imgs)} images, size={size}, augment={augment}")

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        img = Image.open(self.imgs[i]).convert("RGB")
        mask = Image.open(self.anns[i])

        if self._augment:
            img_t, mask_t = self.transform(img, mask)
            mask_t = mask_t.squeeze(0)  # (1, H, W) → (H, W)
        else:
            img_t = self._img_transform(img)
            mask_t = torch.from_numpy(np.array(self._mask_spatial(mask))).long()
            # reduce_zero_label: 0→255, 1-150→0-149
            mask_t = torch.where((mask_t >= 1) & (mask_t <= 150), mask_t - 1, IGNORE_LABEL)

        return img_t, mask_t

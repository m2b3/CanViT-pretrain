"""ADE20K dataset for segmentation."""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from dinov3.eval.segmentation.transforms import make_segmentation_train_transforms
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as T

log = logging.getLogger(__name__)

NUM_CLASSES = 150
IGNORE_LABEL = 255

TransformMode = Literal["center_crop", "squish"]


def _make_val_transforms(
    size: int, mode: TransformMode
) -> tuple[T.Compose, T.Compose | T.Resize]:
    """Create image and mask transforms for validation/evaluation."""
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
    return img_transform, mask_transform


class ADE20kDataset(Dataset[tuple[Tensor, Tensor]]):
    """ADE20K segmentation dataset.

    Args:
        root: Path to ADE20K root (contains images/ and annotations/)
        split: "training" or "validation"
        size: Target image size
        augment: If True, use training augmentations. If False, use val transforms.
        transform_mode: "center_crop" or "squish" (only used when augment=False)
        aug_scale_range: Scale range for training augmentation
        aug_flip_prob: Flip probability for training augmentation
    """

    def __init__(
        self,
        root: Path,
        split: str,
        size: int,
        *,
        augment: bool,
        transform_mode: TransformMode = "center_crop",
        aug_scale_range: tuple[float, float] = (0.5, 2.0),
        aug_flip_prob: float = 0.5,
    ) -> None:
        self.size = size
        self._augment = augment

        if augment:
            # Joint image+mask augmentation (DINOv3 style)
            self._joint_augment = make_segmentation_train_transforms(
                img_size=size,
                random_img_size_ratio_range=list(aug_scale_range),
                crop_size=(size, size),
                flip_prob=aug_flip_prob,
                reduce_zero_label=True,
            )
            self._img_transform = None
            self._mask_transform = None
        else:
            # Separate image/mask transforms for validation
            self._joint_augment = None
            self._img_transform, self._mask_transform = _make_val_transforms(size, transform_mode)

        img_dir = root / "images" / split
        ann_dir = root / "annotations" / split
        self.imgs = sorted(img_dir.glob("*.jpg"))
        self.anns = [ann_dir / (p.stem + ".png") for p in self.imgs]

        assert len(self.imgs) > 0, f"No images in {img_dir}"
        mode_str = "augment=True" if augment else f"transform={transform_mode}"
        log.info(f"ADE20k {split}: {len(self.imgs)} images, size={size}, {mode_str}")

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        img = Image.open(self.imgs[i]).convert("RGB")
        mask = Image.open(self.anns[i])

        if self._augment:
            assert self._joint_augment is not None
            img_t, mask_t = self._joint_augment(img, mask)
            mask_t = mask_t.squeeze(0)  # (1, H, W) → (H, W)
        else:
            assert self._img_transform is not None and self._mask_transform is not None
            img_t = self._img_transform(img)
            mask_t = torch.from_numpy(np.array(self._mask_transform(mask))).long()
            # reduce_zero_label: 0→255, 1-150→0-149
            mask_t = torch.where((mask_t >= 1) & (mask_t <= 150), mask_t - 1, IGNORE_LABEL)

        return img_t, mask_t

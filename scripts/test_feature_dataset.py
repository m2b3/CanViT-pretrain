"""Smoke test for FeatureDataset.

Usage:
    source slurm/env.sh
    uv run python scripts/test_feature_dataset.py \
        --shards-dir $FEATURES_DIR/dinov3_vitb16/512/shards \
        --image-root $IN21K_DIR
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from avp_vit.train.data import IMAGENET_MEAN, IMAGENET_STD
from avp_vit.train.feature_dataset import FeatureDataset


def pca_rgb(patches: torch.Tensor) -> torch.Tensor:
    """Project patch features to RGB via PCA. Input: [n_patches, dim]."""
    from sklearn.decomposition import PCA
    patches_np = patches.float().numpy()
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(patches_np)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return torch.from_numpy(rgb)


def denormalize(img: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization. [3, H, W] -> [H, W, 3] in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img * std + mean
    return img.permute(1, 2, 0).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--viz-idx", type=int, default=0, help="Index to visualize")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--throughput-batches", type=int, default=100, help="Batches for throughput test")
    args = parser.parse_args()

    print(f"Loading from: {args.shards_dir}")
    ds = FeatureDataset(args.shards_dir, args.image_root)

    print("\n=== Metadata ===")
    meta = ds.get_metadata()
    for k, v in meta.items():
        print(f"  {k}: {v}")

    print("\n=== Dataset ===")
    print(f"  Total images: {len(ds):,}")
    print(f"  Shards: {ds.n_shards}")

    print("\n=== Sample access ===")
    for idx in [0, 1, len(ds) // 2, len(ds) - 1]:
        sample = ds[idx]
        print(f"  idx={idx}: image={sample.image.shape}, patches={sample.patches.shape}, cls={sample.cls.shape}, class={sample.class_idx}")

    print("\n=== Random access benchmark ===")
    indices = torch.randint(0, len(ds), (1000,)).tolist()
    t0 = time.perf_counter()
    for i in indices:
        _ = ds[i]
    elapsed = time.perf_counter() - t0
    print(f"  1000 random accesses: {elapsed:.2f}s ({elapsed*1000:.1f}ms total, {elapsed:.4f}ms/access)")

    print("\n=== DataLoader test ===")
    print(f"  batch_size={args.batch_size}, num_workers={args.num_workers}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Quick sanity check (first 5 batches)
    for i, batch in enumerate(loader):
        if i >= 5:
            break
        images, patches, cls, class_idx = batch
        print(f"  Batch {i}: images={images.shape}, patches={patches.shape}, cls={cls.shape}")

    print(f"\n=== Throughput test ({args.throughput_batches} batches) ===")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    n_images = 0
    t0 = time.perf_counter()
    for i, batch in enumerate(loader):
        if i >= args.throughput_batches:
            break
        n_images += batch[0].shape[0]
    elapsed = time.perf_counter() - t0
    throughput = n_images / elapsed
    print(f"  {n_images:,} images in {elapsed:.2f}s")
    print(f"  Throughput: {throughput:,.0f} images/sec")

    # PCA visualization
    import matplotlib.pyplot as plt

    print(f"\n=== PCA Visualization (idx={args.viz_idx}) ===")
    sample = ds[args.viz_idx]
    rel_path = ds.get_path(args.viz_idx)

    grid_size = int(ds.n_patches ** 0.5)
    pca_features = pca_rgb(sample.patches).reshape(grid_size, grid_size, 3)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Use transformed image from dataset (matches features spatially)
    axes[0].imshow(denormalize(sample.image))
    axes[0].set_title(f"Transformed: {rel_path}")
    axes[0].axis("off")

    axes[1].imshow(pca_features)
    axes[1].set_title(f"PCA of patches ({grid_size}x{grid_size})")
    axes[1].axis("off")

    plt.tight_layout()
    out_path = Path("feature_dataset_test.png")
    plt.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")

    print("\n✓ Smoke test passed")


if __name__ == "__main__":
    main()

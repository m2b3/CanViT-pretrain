"""Prove cross-process CUDA non-determinism in bf16 autocast.

Run twice as separate processes and compare outputs.
"""

import argparse
import sys
from pathlib import Path

import torch
from canvit.hub import create_backbone
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_features(image_path: Path, ckpt_path: Path, size: int) -> torch.Tensor:
    """Run bf16 autocast inference, return raw output."""
    device = torch.device("cuda")

    # Load and transform
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    # Load model
    teacher = create_backbone("dinov3_vitb16", weights=str(ckpt_path))
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Inference
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        feats = teacher.forward_norm_features(tensor)
        return feats.patches[0].cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True, help="Save output tensor here")
    parser.add_argument("--compare", type=Path, help="Compare to this saved tensor")
    args = parser.parse_args()

    patches = get_features(args.image, args.ckpt, args.size)
    print(f"Output shape: {patches.shape}, dtype: {patches.dtype}")
    print(f"Output range: [{patches.min():.4f}, {patches.max():.4f}]")

    torch.save(patches, args.output)
    print(f"Saved to: {args.output}")

    if args.compare:
        other = torch.load(args.compare, weights_only=True)
        diff = (patches.float() - other.float()).abs()
        print(f"\nComparison to {args.compare}:")
        print(f"  max diff:  {diff.max().item():.6f}")
        print(f"  mean diff: {diff.mean().item():.6f}")
        if diff.max().item() == 0:
            print("  IDENTICAL - no cross-process variance")
        else:
            print("  DIFFERENT - cross-process variance confirmed")


if __name__ == "__main__":
    main()

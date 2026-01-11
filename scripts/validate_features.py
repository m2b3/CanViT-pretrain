"""Validate exported features against fresh inference.

Usage (in interactive session):
    source slurm/env.sh
    uv run python scripts/validate_features.py \
        --shard $FEATURES_DIR/dinov3_vitb16/512/shards/00000.pt \
        --image-root $IN21K_DIR \
        --teacher-ckpt $DINOV3_VITB16_CKPT \
        --idx 42
"""

import argparse
from pathlib import Path

import torch
from canvit.hub import create_backbone
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_and_transform(path: Path, size: int) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img)


def compare(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    diff = (a.float() - b.float()).abs()
    print(f"  {name}: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--teacher-ckpt", type=Path, required=True)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--teacher-model", type=str, default="dinov3_vitb16")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load shard
    shard = torch.load(args.shard, map_location="cpu", weights_only=False, mmap=True)
    stored_patches = shard["patches"][args.idx]  # bfloat16
    rel_path = shard["paths"][args.idx]
    print(f"Image: {rel_path}")
    print(f"Stored dtype: {stored_patches.dtype}")

    # Load image
    image_path = args.image_root / rel_path
    img_tensor = load_and_transform(image_path, shard["image_size"]).unsqueeze(0).to(device)

    # Load teacher
    teacher = create_backbone(args.teacher_model, weights=str(args.teacher_ckpt))
    teacher = teacher.to(device).eval()

    with torch.no_grad():
        # Run inference at different precisions
        print("\n=== Fresh inference ===")

        # float32 inference (ground truth)
        feats_f32 = teacher.forward_norm_features(img_tensor)
        patches_f32 = feats_f32.patches[0].cpu()  # float32
        print(f"float32 output range: [{patches_f32.min():.4f}, {patches_f32.max():.4f}]")

        # bfloat16 autocast (what export uses)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            feats_bf16 = teacher.forward_norm_features(img_tensor)
            patches_bf16_raw = feats_bf16.patches[0].cpu()  # f32 from autocast
            patches_bf16_stored = patches_bf16_raw.to(torch.bfloat16)  # matches export path

        print("\n=== Precision comparison ===")

        # Cast ground truth to different dtypes
        patches_to_bf16 = patches_f32.to(torch.bfloat16)
        patches_to_f16 = patches_f32.to(torch.float16)

        print("Quantization error (f32 → dtype → f32):")
        compare("f32 → bf16", patches_f32, patches_to_bf16.float())
        compare("f32 → f16 ", patches_f32, patches_to_f16.float())

        print("\nAutocast vs f32 (same run):")
        compare("bf16 autocast raw vs f32", patches_f32, patches_bf16_raw)
        compare("bf16 autocast→bf16 vs f32", patches_f32, patches_bf16_stored)

        print("\nStored (bf16) vs fresh:")
        compare("stored vs f32              ", stored_patches, patches_f32)
        compare("stored vs f32→bf16         ", stored_patches, patches_to_bf16)
        compare("stored vs autocast raw     ", stored_patches, patches_bf16_raw)
        compare("stored vs autocast→bf16   ", stored_patches, patches_bf16_stored)  # exact export path

        # Run autocast twice to check determinism
        print("\n=== Determinism check ===")
        with torch.autocast(device.type, dtype=torch.bfloat16):
            run1 = teacher.forward_norm_features(img_tensor).patches[0].cpu()
            run2 = teacher.forward_norm_features(img_tensor).patches[0].cpu()
        compare("run1 vs run2 (same session)", run1, run2)

        # dtype precision info
        print("\n=== Dtype info ===")
        print(f"bfloat16: 8 exp bits, 7 mantissa bits, eps={torch.finfo(torch.bfloat16).eps:.6f}")
        print(f"float16:  5 exp bits, 10 mantissa bits, eps={torch.finfo(torch.float16).eps:.6f}")
        print(f"float32: 8 exp bits, 23 mantissa bits, eps={torch.finfo(torch.float32).eps:.8f}")


if __name__ == "__main__":
    main()

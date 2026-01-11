"""Export teacher features for IN21k."""

import gc
import hashlib
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path

import pyarrow.parquet as pq
import torch
import tyro
from canvit.hub import create_backbone
from PIL import Image, ImageFile
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = False
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Config:
    shard: int | None = None
    start_shard: int | None = None
    end_shard: int | None = None
    parquet: Path | None = None
    image_root: Path | None = None
    out_dir: Path | None = None
    teacher_ckpt: Path | None = None
    teacher_model: str = "dinov3_vitb16"
    shard_size: int = 4096
    batch_size: int = 64
    num_workers: int = 8
    image_size: int = 512


def log_gpu(label: str) -> None:
    a = torch.cuda.memory_allocated() / 1e9
    log.info(f"GPU [{label}]: {a:.2f}GB")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


class ImageDataset(Dataset):
    def __init__(self, root: Path, paths: list[str], size: int):
        self.root = root
        self.paths = paths
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, int, bool]:
        path = self.root / self.paths[idx]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                img = Image.open(path).convert("RGB")
                img.load()
            return self.transform(img), idx, True
        except Exception as e:
            log.warning(f"Bad image {path}: {e}")
            return torch.full((3, self.size, self.size), float("nan")), idx, False


def main(cfg: Config) -> None:
    device = torch.device("cuda")

    # Resolve paths
    image_root = cfg.image_root or Path(os.environ["AVP_TRAIN_DIR"])
    parquet_path = cfg.parquet or Path(os.environ["AVP_INDEX_DIR"]) / f"{image_root.name}.parquet"
    out_dir = cfg.out_dir or Path(os.environ["AVP_FEATURES_DIR"])
    teacher_ckpt = cfg.teacher_ckpt or Path(os.path.expanduser(os.environ["AVP_TEACHER_CKPT"]))
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"out_dir: {out_dir}")

    # Load parquet
    table = pq.read_table(parquet_path)
    n_images = len(table)
    n_shards = ceil(n_images / cfg.shard_size)
    parquet_hash = sha256_file(parquet_path)
    log.info(f"Parquet: {n_images:,} images, {n_shards} shards")

    # Shard range
    if cfg.shard is not None:
        start, end = cfg.shard, cfg.shard + 1
    elif cfg.start_shard is not None and cfg.end_shard is not None:
        start, end = cfg.start_shard, min(cfg.end_shard, n_shards)
    else:
        raise ValueError("Specify --shard or --start-shard/--end-shard")
    log.info(f"Shards: [{start}, {end})")

    # Meta
    meta_path = out_dir / "meta.json"
    expected = {"schema_version": 1, "shard_size": cfg.shard_size, "n_images": n_images,
                "parquet_sha256": parquet_hash, "teacher_model": cfg.teacher_model, "image_size": cfg.image_size}
    if meta_path.exists():
        with open(meta_path) as f:
            existing = json.load(f)
        for k, v in expected.items():
            assert existing.get(k) == v, f"meta.json mismatch: {k}"
    else:
        with open(meta_path, "w") as f:
            json.dump({**expected, "created_at": datetime.now(timezone.utc).isoformat()}, f)

    # Load teacher
    teacher = create_backbone(cfg.teacher_model, weights=str(teacher_ckpt)).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    patch_size = teacher.patch_size_px
    embed_dim = teacher.embed_dim
    n_patches = (cfg.image_size // patch_size) ** 2
    log.info(f"Teacher: {embed_dim}d, {n_patches} patches")
    log_gpu("after teacher")

    # Warmup
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        teacher.forward_norm_features(torch.randn(1, 3, cfg.image_size, cfg.image_size, device=device))
    torch.cuda.synchronize()
    log_gpu("after warmup")

    # Export
    for shard_id in range(start, end):
        shard_path = shards_dir / f"{shard_id:05d}.pt"
        if shard_path.exists():
            log.info(f"Shard {shard_id}: exists, skip")
            continue

        # Slice
        start_idx = shard_id * cfg.shard_size
        end_idx = min(start_idx + cfg.shard_size, n_images)
        n = end_idx - start_idx
        paths = table.slice(start_idx, n).column("path").to_pylist()
        class_idxs = table.slice(start_idx, n).column("class_idx").to_pylist()
        log.info(f"Shard {shard_id}: {n} images")

        # Allocate
        patches = torch.empty(n, n_patches, embed_dim, dtype=torch.bfloat16, device=device)
        cls = torch.empty(n, embed_dim, dtype=torch.bfloat16, device=device)
        failed = []

        # Inference
        loader = DataLoader(ImageDataset(image_root, paths, cfg.image_size),
                            batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
        t0 = time.perf_counter()
        idx = 0

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for imgs, indices, ok in tqdm(loader, desc=f"Shard {shard_id}", leave=False):
                for i, success in zip(indices.tolist(), ok.tolist()):
                    if not success:
                        failed.append(i)
                imgs = imgs.to(device)
                feats = teacher.forward_norm_features(imgs)
                bs = imgs.shape[0]
                patches[idx:idx+bs] = feats.patches.to(torch.bfloat16)
                cls[idx:idx+bs] = feats.cls.to(torch.bfloat16)
                idx += bs

        elapsed = time.perf_counter() - t0
        log.info(f"  {n/elapsed:.0f} img/s, {len(failed)} failed")
        log_gpu("before save")

        # Save
        tmp = shard_path.with_suffix(".tmp")
        torch.save({
            "patches": patches, "cls": cls, "paths": paths,
            "class_idxs": torch.tensor(class_idxs, dtype=torch.int32),
            "shard_id": shard_id, "start_idx": start_idx, "failed_indices": failed,
            "parquet_sha256": parquet_hash, "teacher_model": cfg.teacher_model,
            "image_size": cfg.image_size, "dtype": "torch.bfloat16",
        }, tmp)
        tmp.rename(shard_path)
        log.info(f"  Wrote {shard_path.name}")

        # Cleanup
        del patches, cls, loader
        gc.collect()
        torch.cuda.empty_cache()
        log_gpu("after cleanup")

    log.info("Done")


if __name__ == "__main__":
    main(tyro.cli(Config))

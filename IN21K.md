# ImageNet-21k Support for SLURM/Compute Canada

## Context

Adapting `scripts/train_scene_match/` to run on Alliance Canada with ImageNet-21k (IN21k), which has **no train/val split**.

**Problem**: IN21k is at `/project/.../winter21_whole/` with 13M+ images in 19k class folders. Read-only filesystem, limited inodes. Need to split into train/val programmatically.

**Solution**: Parquet index files listing which images belong to train vs val. Fast loading (~0.5s) vs walking filesystem (~20-30 min).

---

## What Was Implemented

### Location: `avp_vit/train/data/indexed/`

```
avp_vit/train/data/indexed/
├── __init__.py   # Pure functions (72 lines)
└── test.py       # 5 tests
```

### Exports

```python
from avp_vit.train.data.indexed import (
    SCHEMA_VERSION,       # int = 1
    IndexMetadata,        # Dataclass for parquet metadata
    SplitIndex,           # Dataclass: train/val Path pair
    parse_index_metadata, # Parse raw parquet metadata
    split_by_class,       # Core splitting logic
)
```

### `split_by_class(class_to_images, val_ratio, seed) -> (train, val)`

- Input: `{class_name: [image_filenames]}`
- Output: `(train_records, val_records)` as `[(relative_path, class_name), ...]`
- Guarantees: deterministic, at least 1 val per class, no data loss

---

## What Remains To Be Done

### 1. `generate_split_index` function (in `indexed/__init__.py`)

```python
def generate_split_index(
    root: Path,
    output_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> SplitIndex:
    """Generate train/val parquet indices. Uses tqdm."""
    # Walk root, build class_to_images dict
    # Call split_by_class(...)
    # Write parquet: columns [path, class_name, class_idx]
    # Metadata: schema_version, root_name, split, val_ratio, seed, n_samples, n_classes, generated_at
```

### 2. `IndexedImageFolder` class (in `indexed/__init__.py`)

```python
class IndexedImageFolder(ImageFolder):
    def __init__(self, root: str, index_file: Path, transform=None):
        # Read parquet, validate via parse_index_metadata()
        # ASSERT root_name matches (crash on mismatch)
        # Build self.samples, self.classes, self.class_to_idx
        # Log everything
```

### 3. `ensure_split_index` function (in `indexed/__init__.py`)

```python
def ensure_split_index(root, output_dir, val_ratio, seed) -> SplitIndex:
    """Return index paths, auto-generate if missing."""
```

### 4. Update `scripts/train_scene_match/config.py`

Add fields:
```python
index_dir: Path = Path("indices")
val_ratio: float = 0.1
split_seed: int = 42
```

### 5. Update `scripts/train_scene_match/data.py`

Modify `create_loaders` function (~line 132):

```python
def create_loaders(
    cfg: Config, stages: dict[int, ResolutionStage]
) -> tuple[dict[int, InfiniteLoader], dict[int, InfiniteLoader]]:

    # NEW: Detect if indexing needed
    needs_index = cfg.train_dir.resolve() == cfg.val_dir.resolve()

    if needs_index:
        log.info(f"train_dir == val_dir → indexed split mode")
        log.info(f"  val_ratio={cfg.val_ratio}, seed={cfg.split_seed}")
        from avp_vit.train.data.indexed import IndexedImageFolder, ensure_split_index
        index = ensure_split_index(cfg.train_dir, cfg.index_dir, cfg.val_ratio, cfg.split_seed)
    else:
        log.info(f"Separate train/val directories → ImageFolder mode")
        index = None

    # ... existing loop ...
    for G, stage in stages.items():
        # Replace ImageFolder with conditional:
        if index is not None:
            train_dataset = IndexedImageFolder(
                str(cfg.train_dir), index.train,
                train_transform(scene_size_px, (cfg.crop_scale_min, 1.0))
            )
            val_dataset = IndexedImageFolder(
                str(cfg.val_dir), index.val,
                val_transform(scene_size_px)
            )
        else:
            train_dataset = ImageFolder(...)  # existing code
            val_dataset = ImageFolder(...)    # existing code
```

---

## Design Decisions

1. **Crash on root_name mismatch**: Silent mismatch = silent data corruption
2. **Auto-generate if missing**: First run slow (~20 min), subsequent fast (~0.5s)
3. **Schema versioning**: Hard fail if version mismatch
4. **At least 1 val per class**: `max(1, int(n * ratio))`
5. **Parquet format**: ~60MB for 13M entries, pyarrow dependency OK

---

## Usage (Once Complete)

**ILSVRC** (unchanged):
```bash
uv run python -m scripts.train_scene_match \
    --train-dir /datasets/ILSVRC/.../train \
    --val-dir /datasets/ILSVRC/.../val
```

**IN21k**:
```bash
uv run python -m scripts.train_scene_match \
    --train-dir /project/.../winter21_whole \
    --val-dir /project/.../winter21_whole \
    --index-dir ./indices
```

---

## Performance

| Operation | Time |
|-----------|------|
| Index generation (one-time) | ~15-30 min |
| Index file size | ~60 MB each |
| IndexedImageFolder init | ~0.5 sec |
| ImageFolder init (13M files) | ~20-30 min |

---

## Tests

```bash
uv run pytest avp_vit/train/data/indexed/test.py -v
```

5 tests: metadata parsing, schema validation, determinism, no data loss, stratification.

---

## SLURM Notes

- Copy ImageNet to `$SLURM_TMPDIR` for fast I/O
- `COMET_OFFLINE_DIRECTORY=$SCRATCH/comet_offline`
- `TORCH_COMPILE_CACHE_DIR=$SCRATCH/torch_compile_cache`
- All config overridable via CLI (tyro)

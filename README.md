# AVP-ViT

Active Vision Pretraining with Vision Transformers.

## Repository Structure

```
avp_vit/
  train/
    __main__.py           # Entry: tyro CLI → Optuna study → train()
    config.py             # Config dataclass (ALL hyperparameters live here)
    loop.py               # train() and training_loop(): setup, resume, main loop
    step.py               # training_step(): TBPTT chunks, branches, losses
    data.py               # Loaders (raw images or precomputed features)
    feature_dataset.py    # IterableDataset for precomputed shards
    norm.py               # PositionAwareNorm: fixed per-position stats
    model.py              # Model creation, compilation
    scheduler.py          # Warmup → constant LR
    viz/                  # Validation, PCA, Comet logging
  checkpoint/
    __init__.py           # CheckpointData TypedDict, save/load functions

inference_app/            # Streamlit demo
  gpu_worker.py           # ← Reference implementation for checkpoint loading
  app.py                  # UI (no torch)
  rendering.py            # Viz helpers (numpy/PIL only)

scripts/                  # Analysis and utilities
  export_features.py      # Precompute teacher features to shards
  flops.py                # FLOP analysis
  bench_*.py              # Benchmarks
  validate_in1k/          # ImageNet evaluation

slurm/
  env.sh                  # ← Environment setup (paths). Source first.
  train.sbatch            # Production training (job arrays)

canvit                    # Model architecture (SEPARATE REPO)
                          # Source: .venv/.../site-packages/canvit/
                          # Or: ~/code/CanViT, github.com/m2b3/CanViT
                          # Check uv.lock for commit, pyproject.toml for branch
```

## Key Implementation Details

**Training step architecture** (`step.py`):
- Truncated BPTT with configurable `chunk_size` (default 2)
- Independent branches (FULL-start vs RANDOM-start) - no `retain_graph`
- Memory is O(chunk_size), not O(trajectory_length)
- t=0: FULL or RANDOM viewpoint based on branch type
- t≥1: RANDOM or POLICY (half each, shuffled, if policy enabled)
- Optional gradient checkpointing on odd timesteps

**Normalization** (`norm.py`):
- `PositionAwareNorm`: fixed per-position per-dimension stats [n_tokens, embed_dim]
- Computed once from first feature shard, then frozen
- No train/eval difference - always uses fixed stats
- Required for inference (model outputs normalized features)

**Feature shards** (`feature_dataset.py`):
- IterableDataset - each worker loads own shards, no fork issues
- Workers interleave: worker i gets shards i, i+nw, i+2*nw...
- `failed_indices` field: samples with NaN features (auto-skipped)
- `.clone()` required when yielding mmap'd tensors

**Job array resume** (`loop.py`):
- `run_name` derived from SLURM_ARRAY_JOB_ID → consistent across array
- Auto-finds `latest.pt` symlink in run_dir
- Comet experiment continuation via `comet_id` in checkpoint
- FAILED marker prevents infinite crash loops → `scancel` on crash
- `training_config_history`: tracks config changes across resumes

**Forced FlashAttention** (`loop.py`):
- Only FlashAttention enabled (mem_efficient and math backends disabled)

## Deployment Strategy

| Environment | Use for | Hardware |
|-------------|---------|----------|
| **Local** (macOS) | Development, code review, quick non-GPU tests | M4 Pro |
| **crockett** | Quick CUDA tests, interactive debugging | RTX 4090 (not always online) |
| **Nibi cluster** | All serious training, evaluation, feature export | H100 (via SLURM) |

**Workflow**: Develop locally → quick CUDA test on crockett if needed → submit to Nibi.

**Cluster paths**: Single source of truth is `slurm/env.sh`. See `SLURM.md` for setup (deploy keys, accounts).

## Documentation

| File | Content |
|------|---------|
| `CLAUDE.md` | Development guidelines, session startup |
| `SLURM.md` | HPC setup, deploy keys, job management |
| `docs/philosophy.md` | Vision, design choices, training philosophy |

## Commands

```bash
# Cluster setup
source slurm/env.sh

# Training
sbatch slurm/train.sbatch                    # Production
uv run -m avp_vit.train --help               # All options

# Inference (reference implementation for loading + using checkpoints)
uv run streamlit run inference_app/__main__.py
```

## Pitfalls

- **DINOv3 ≠ DINOv2** - different model, not a typo.
- **Coordinate conventions vary** - check canvit `viewpoint/`, `coords/`.
- **GPU syncs kill throughput** - `.item()`, `.cpu()`, logging in hot loops.
- **Read canvit source** - model lives there, not here.

# AVP-ViT

Active Vision Pretraining with Vision Transformers.

## Context

Research project, started September 2024. Mostly one person. High cognitive load.

This repo (`avp-vit`) is experimental - training, viz, monitoring code that evolves rapidly. We accept the mess, gradually clean/extract/modularize.

`canvit` (separate repo, in venv) is the core architecture - stabler, cleaner API, geared for future public release. **Will not merge back.** The split is intentional: core arch evolves slower than experiment code.

Everything can change. Be ready.

## Vision

**Goal**: Active Vision Foundation Models. Scalable, general-purpose, task-agnostic active vision pretraining - done right. Fast training, fast inference, clean gradient signal, smart design.

**Three pillars:**
- **Seeing**: Single-step behavior. Given a glimpse, can you make sense of it?
- **Integration**: Spatiotemporal filling-in from partial views across time. Recurrence, not pooling - enables top-down feedback.
- **Deciding where to look**: Policy. Use what you've seen to inform what to look at next.

## Design Choices

**Sequential glimpses**: Each glimpse is a crop at (center, scale). Canvas and CLS persist across steps - recurrence enables top-down feedback, not just bottom-up aggregation.

**Fixed canvas size**: Bounds memory. Glimpse positions vary, canvas dimensions don't grow.

**Efficiency**: Small glimpse = cheap per step. Canvas cross-attention cheaper than full self-attention. See `scripts/flops.py`.

**Distillation**: Student reconstructs frozen teacher (DINOv3) features. Clean gradient signal.

## Training Philosophy

**Generality**: Model should work with ANY viewpoint sequence - random, fixed, learned, whatever. Training uses diverse viewpoint types to ensure this.

**Exploration via structure**: Multiple branches with different viewpoint types. Policy can be deterministic (no noise injection) because other branches provide diversity. Backbone trains on all branches, stays robust.

**Maximize teacher signal**: Teacher inference is expensive. Don't waste it. Full-view branches ensure we extract maximum value from each image.

**Difficulty gradient**: Partial views from the start (not just after seeing full image) forces real filling-in. The hard case, not the easy case.

**Policy is opportunistic**: Cheap to train alongside everything else - just another branch. Even if policy isn't good yet, it:
- Shapes representations for task-specific policy finetuning later
- Keeps VPE token policy-relevant during pretraining
- Gives something to monitor during iteration

**How to evaluate policy**: full→policy should beat full→random and full→full. "You've seen everything once, now where should you look to do even better?" If it can't beat random second look, policy isn't helping yet.

**Current state (Jan 2026)**: Policy learning is experimental. Might take long pretraining to start mattering. The branching structure may evolve.

## Repository Structure

```
avp_vit/                    # Main package
  train/                    # Training infrastructure
    config.py               # ← THE training config (dataclass with all hyperparams)
    loop.py                 # Main training loop, resume logic
    step.py                 # Single training step (forward, loss, backward)
    data.py                 # Data loading (raw images or precomputed features)
    feature_dataset.py      # IterableDataset for precomputed teacher features
    norm.py                 # Position-aware normalization (required for inference)
    viz/                    # Visualization (PCA, metrics, Comet logging)
  checkpoint/               # Checkpoint save/load (TypedDict schema in __init__.py)

inference_app/              # Streamlit demo for interactive inference

scripts/                    # One-off utilities
  flops.py                  # FLOP analysis
  export_features.py        # Precompute teacher features (for fast training)
  bench_*.py                # Benchmarks (latency, throughput, loaders)
  validate_in1k/            # ImageNet-1k evaluation
  train_ade20k_probe.py     # Segmentation probe training

slurm/                      # HPC infrastructure
  env.sh                    # ← Environment setup (paths, caches) - source this first
  train.sbatch              # Production training (job arrays)
  export_features.sh        # Feature precomputation jobs

canvit                      # Model architecture (SEPARATE REPO)
                            # Source: ~/code/CanViT or https://github.com/m2b3/CanViT/
                            # Installed as SSH dep (deploy keys, see SLURM.md)
                            # Key modules: hub.py, viewpoint/, coords/
```

## Documentation

| File | Content |
|------|---------|
| `CLAUDE.md` | Development guidelines, session startup |
| `SLURM.md` | Distributed training on HPC clusters |
| `POLICY.md` | Policy learning design notes |
| `avp_vit/train/config.py` | All training hyperparameters (dataclass) |
| `avp_vit/checkpoint/__init__.py` | Checkpoint schema (`CheckpointData` TypedDict) |

## Commands

```bash
# Setup (on cluster)
source slurm/env.sh

# Training
sbatch slurm/train.sbatch              # Production (SLURM job arrays)
uv run -m avp_vit.train --help         # Local / see all options

# Inference demo (reference implementation for checkpoint loading + model usage)
uv run streamlit run inference_app/__main__.py
```

## Pitfalls

- **DINOv3 exists and is not DINOv2** - not a typo, not the same. Don't assume.
- **Coordinate conventions vary** - internal vs external APIs vs PyTorch grid_sample. Check canvit: `viewpoint/`, `coords/`.
- **GPU syncs kill performance** - `.item()`, `.cpu()`, logging inside hot loops. Keep logging outside.
- **Read canvit source before assuming model behavior** - it's not in this repo.

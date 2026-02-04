# CanViT-train

Pretraining and evaluation for CanViT.

## Structure

```
canvit_pretrain/    # Pretraining
canvit_eval/        # Evaluation (IN1k, ADE20k)
drac_imagenet/      # Fast indexed ImageFolder for IN21k
scripts/            # Utilities (benchmarks, feature export)
slurm/              # SLURM job scripts (source env.sh first)
```

## Entry Points

```bash
# Pretraining
uv run -m canvit_pretrain.train --help

# IN1k probe evaluation
uv run -m canvit_eval.in1k --help

# ADE20k segmentation
uv run -m canvit_eval.ade20k --help
```

## SLURM (Nibi cluster)

**Setup** (one-time):
```bash
# Clone to scratch
cd ~/scratch
git clone git@github.com:m2b3/CanViT-train.git

# SSH config: add Host github.com-canvit-train with deploy key
# gitconfig: rewrite git@github.com:m2b3/CanViT-train.git → use that host

# HuggingFace auth (required for private model repos)
uvx hf auth login
```

**Usage**:
```bash
cd ~/scratch/CanViT-train

# All scripts source slurm/env.sh (sets $IN1K_VAL_DIR, $HF_HOME, etc.)
sbatch slurm/eval_in1k.sbatch              # IN1k evaluation (~20min)
sbatch slurm/eval_ade20k.sbatch            # ADE20k probe training (~2hr)
sbatch slurm/train.sbatch                  # Pretraining (long)
bash slurm/interactive.sh                  # Interactive GPU session
```

**Key files**:
- `slurm/env.sh` - Environment setup (dataset paths, caches, Comet key)
- `outputs/` - Evaluation results (parquet files)
- `logs/` - Job stdout/stderr

## See Also

- `CLAUDE.md` - Development guidelines
- `canvit` package - Core model architecture (separate repo)

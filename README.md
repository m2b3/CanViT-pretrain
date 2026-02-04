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

## SLURM

```bash
source slurm/env.sh
sbatch slurm/train.sbatch
```

## See Also

- `CLAUDE.md` - Development guidelines
- `canvit` package - Core model architecture (separate repo)

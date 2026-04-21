# CanViT-pretrain

Pretraining for CanViT: passive-to-active dense latent distillation from DINOv3.

## Prerequisites

### Env

```bash
uv sync
```
Requires Python 3.12+. Deps (including git-pulled `canvit-pytorch`, `dinov3`, etc.) live in `pyproject.toml`.

### Credentials

Read from the launching shell via `direnv` (`.envrc.nibi` is the Nibi template):

- **`HF_TOKEN`** — required. Pulls the DINOv3 teacher from HF Hub.
- **`COMET_API_KEY`** — optional. Without it, Comet logging is skipped.

### Data + cache paths

```bash
cp .envrc.nibi .envrc && direnv allow
```

`.envrc.nibi` documents every variable (IN21k images, IN1k val, parquet indexes, features dir, checkpoints, HF/Torch caches) with the Nibi defaults. Re-tailor per cluster as needed.

## Workflow

Two stages: export teacher features once per (teacher, dataset, resolution); then repeated training runs that consume the cached features.

### Export DINOv3 teacher features

Build the shuffled parquet index, then submit the export array job:

```bash
uv run python scripts/build_shuffled_index.py \
  --image-root $IN21K_IMAGE_DIR --index-dir $INDEX_DIR --dataset in21k

sbatch --array=0-99%20 slurm/export_features.sh
```

See `slurm/export_features.sh` for the teacher/image-size/shard constants and `scripts/export_in21k_features.py`'s header comment for atomic-write guarantees, resume semantics, and monitor commands.

### Train

```bash
sbatch slurm/train.sbatch [--flag value ...]
```

Every field of `canvit_pretrain.train.config.Config` is a `--kebab-case-flag` via `tyro`:
```bash
uv run python -m canvit_pretrain.train --help
```

The sbatch passes `"$@"` straight through. See `slurm/train.sbatch`'s header for array-sizing, run-name derivation, and override examples (CanViT-S, resuming, quick smoke).

Switching teacher / resolution / dataset requires re-running the feature export.

### Ablations

Each ablation is a variant of the flagship run (same harness, one parameter
changed). Submit individually:

```bash
bash slurm/ablations/baseline.sh
bash slurm/ablations/no-bptt.sh
# ... one script per variant
```

Shared config (array size, warmup, run-name prefix) lives in
`slurm/ablations/common.sh`; each variant script documents what it changes
in a one-line header comment.

## Interactive dev

```bash
bash slurm/interactive.sh
```
`salloc` wrapper. Inside the allocation, `.envrc` is sourced by `slurm/env.sh`; run the same `uv run python -m canvit_pretrain.train ...` the sbatch runs.

## Repository layout

```bash
uv run pypatree
```

## Related repos

| Repo | Role |
|------|------|
| [CanViT-PyTorch](https://github.com/m2b3/CanViT-PyTorch) (public, canonical) | Core model + policies |
| [CanViT-specialize](https://github.com/m2b3/CanViT-specialize) | Probes, datasets, IoU metrics, IN1K finetuning |
| [CanViT-eval](https://github.com/m2b3/CanViT-eval) | Evaluation (produces `.pt` result files) |
| [dinov3-in1k-probes](https://github.com/yberreby/dinov3-in1k-probes) | DINOv3 IN1K linear probes (baselines) |
| [CanViT-Toward-AVFMs](https://github.com/m2b3/CanViT-Toward-AVFMs) | Paper pipeline |

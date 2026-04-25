# CanViT-pretrain

Passive-to-active dense latent distillation
of CanViT


from DINOv3.

Model lives in [CanViT-PyTorch](https://github.com/m2b3/CanViT-PyTorch).

This repository
was originally designed to run on [the Nibi SLURM cluster](https://docs.alliancecan.ca/wiki/Nibi)
using its [hosted ImageNet-21k `winter21_whole` replica](https://docs.alliancecan.ca/wiki/ImageNet).

## Setup

```bash
cp .envrc.example .envrc && direnv allow
# Edit .envrc to adapt to your environment.
```

Please ensure that `HF_TOKEN` and `COMET_API_KEY` are set.

## Run

Export DINOv3 teacher features once:

```bash
uv run python scripts/build_shuffled_index.py \
  --image-root $IN21K_IMAGE_DIR --index-dir $INDEX_DIR --dataset in21k
sbatch --array=0-99%20 slurm/export_features.sh
```

Pretraining:

```bash
sbatch slurm/train.sbatch [--flag value ...]
```

Ablations:

```bash
bash slurm/ablations/baseline.sh
bash slurm/ablations/no-bptt.sh
# ...
```

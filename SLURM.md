# SLURM Deployment Guide (Nibi)

## Storage

| Location | Persistence | Use for |
|----------|-------------|---------|
| `$SLURM_TMPDIR` | Job only (fast local SSD) | uv cache |
| `$SCRATCH` | Persistent | Checkpoints, HF/torch caches, dataset index |
| `$HOME` | Persistent (quota-limited) | Small files (API keys, SSH keys) |

## Paths

All paths defined in `slurm/env.sh` - source it first. Key exports:
- `IN21K_DIR`, `IN1K_VAL_DIR` - datasets
- `CHECKPOINTS_DIR`, `FEATURES_DIR` - storage
- `DINOV3_VITB16_CKPT` - teacher checkpoint

## Accounts

| Account | Use for |
|---------|---------|
| `def-skrishna` | CPU-only quick tests |
| `rrg-skrishna_gpu` | GPU jobs (H100) |

---

## Pre-flight (One-time Setup)

### 1. SSH Deploy Keys

Both repos are private. SSH config (`~/.ssh/config`):
```
Host github.com-canvit
    HostName github.com
    User git
    IdentityFile ~/.ssh/canvit_deploy
    IdentitiesOnly yes

Host github.com-avp-vit
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_rsa_avp_vit
    IdentitiesOnly yes

Host github.com
    IdentityFile ~/.ssh/nibi_deploy_key
    IdentitiesOnly yes
```

Git URL rewrites (`~/.gitconfig`):
```bash
git config --global url."git@github.com-avp-vit:m2b3/avp-vit.git".insteadOf "git@github.com:m2b3/avp-vit.git"
git config --global url."ssh://git@github.com-canvit/m2b3/CanViT.git".insteadOf "ssh://git@github.com/m2b3/CanViT.git"
```

### 2. Directories

```bash
mkdir -p ~/scratch/avp-vit/logs $SCRATCH/avp_checkpoints $SCRATCH/in21k_index
```

### 3. Comet API Key

```bash
echo "your-api-key" > ~/comet_api_key.txt
chmod 600 ~/comet_api_key.txt
```

---

## Smoke Test (CPU)

Before using GPU allocation, verify everything works:

```bash
srun --account=def-skrishna --mem=32G --cpus-per-task=4 --time=0:30:00 --pty bash
```

```bash
# Env setup (same as sbatch)
export PATH=$HOME/.local/bin:$PATH
module load java/17.0.6
export HF_HOME="$SCRATCH/.huggingface"
export TORCH_HOME="$SCRATCH/.torch"
cd ~/scratch/avp-vit

# 1. uv sync (tests canvit SSH access)
uv sync

# 2. Imports
uv run python -c "import canvit; import dinov3_probes; import drac_imagenet; print('imports ok')"

# 3. Paths exist
ls ~/projects/def-skrishna/checkpoints/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
ls /datashare/imagenet/winter21_whole | head -3
ls /datashare/imagenet/ILSVRC2012/val | head -3

# 4. Teacher loads (CPU)
uv run python -c "
from canvit.hub import create_backbone
import os
b = create_backbone('dinov3_vitb16', weights=os.path.expanduser('~/projects/def-skrishna/checkpoints/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'))
print(f'Teacher: {b.embed_dim}d, {b.n_blocks} blocks')
"

# 5. HF probe loads
uv run python -c "
from dinov3_probes import DINOv3LinearClassificationHead
p = DINOv3LinearClassificationHead.from_pretrained('yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe')
print('Probe loaded')
"
```

---

## Optional Pre-caching

### Dataset Index

Pre-build parquet index (otherwise built on first train run):
```bash
source slurm/env.sh
uv run python -c "
from pathlib import Path; from drac_imagenet import IndexedImageFolder; import os
IndexedImageFolder(Path(os.environ['AVP_TRAIN_DIR']), Path(os.environ['AVP_INDEX_DIR']), None)
"
```

### HF Probe

```bash
source slurm/env.sh
uv run python -c "
from dinov3_probes import DINOv3LinearClassificationHead
DINOv3LinearClassificationHead.from_pretrained('yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe')
"
```

---

## Submit (GPU)

```bash
cd ~/scratch/avp-vit
sbatch slurm/train.sbatch --n-trials 1
sbatch slurm/train.sbatch --n-trials 1 --batch-size 128
```

Entry point uses Optuna. `--n-trials 1` = single run (no search).

---

## SIGUSR1 Checkpoint

Saves checkpoint after current step (useful for preemption):
```bash
scancel --signal=USR1 --batch $SLURM_JOB_ID
```

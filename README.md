# CanViT-pretrain

Pretraining for CanViT: passive-to-active dense latent distillation from DINOv3.

## Prerequisites

### Python env

```bash
uv sync
```

Requires Python 3.12+ (`.python-version`). Direct dependencies pulled from git: `canvit-pytorch`, `dinov3`, `dinov3-in1k-probes`, `ybml`. See `pyproject.toml` for the full list including `torch`, `comet-ml`, `huggingface-hub`, `tyro`, `optuna`, `polars`, `pyarrow`, `opencv-python-headless`.

### Credentials

Two secrets are read from the launching shell (via `direnv` + `.envrc`; `.envrc.nibi` is a template for the Nibi cluster):

- **`HF_TOKEN`** — used to pull the DINOv3 teacher checkpoint from HuggingFace Hub. `.envrc.nibi` reads (in priority order) `~/hf_token.txt`, then `$HF_HOME/token`.
- **`COMET_API_KEY`** — optional. Without it, Comet logging is skipped. `.envrc.nibi` reads `~/comet_api_key.txt`.

### Env vars / paths (template in `.envrc.nibi`)

These are required by the code at runtime. Copy the template and tailor for your cluster:

```bash
cp .envrc.nibi .envrc
direnv allow
```

| Variable | What it points at | Nibi value (in `.envrc.nibi`) |
|---|---|---|
| `IN21K_IMAGE_DIR` | ImageNet-21k image tree (winter21 whole) | `/datashare/imagenet/winter21_whole` |
| `IN1K_VAL_IMAGE_DIR` | IN1k val set (pretraining-time eval probe) | `/datashare/imagenet/ILSVRC2012/val` |
| `INDEX_DIR` | Parquet indexes of the training set (expected file `{INDEX_DIR}/in21k-shuffled.parquet`) | `$SCRATCH/dataset_indexes` |
| `FEATURES_DIR` | Precomputed teacher features. Layout is `{FEATURES_DIR}/{dataset}/dinov3_vitb16/{image_size}/shards/*.pt` (set by `slurm/export_features.sh`). | `~/projects/def-skrishna/dinov3_dense_features` |
| `CHECKPOINTS_DIR` | Training-run output root. Run directory is `{CHECKPOINTS_DIR}/{run_name}/`. | `~/projects/def-skrishna/checkpoints` |
| `HF_HOME` | HuggingFace cache (teacher weights). Put on `$SCRATCH` to persist across SLURM jobs. | `$SCRATCH/.huggingface` |
| `TORCH_HOME` / `TORCH_COMPILE_CACHE_DIR` | Torch + torch.compile caches | `$SCRATCH/.torch{,_compile_cache}` |

`.envrc.nibi` also defines `ADE20K_ZIP`, `SA1B_TAR_DIR`, `SA1B_LINKS`, `SA1B_FEATURES_DIR` for sibling experiments; not required for the default IN21k pretraining flow.

## Workflow

Two phases: upfront teacher-feature export (once per teacher/dataset/resolution), then repeated training runs that consume the cached features.

### Phase 1 — Export DINOv3 teacher features

```bash
sbatch --array=0-99%20 slurm/export_features.sh
```

The sbatch calls `scripts/export_in21k_features.py` with: teacher `facebook/dinov3-vitb16-pretrain-lvd1689m`, `IMAGE_SIZE=512`, `SHARD_SIZE=4096`, `SHARDS_PER_JOB=36`. Inputs: `$INDEX_DIR/in21k-shuffled.parquet` + `$IN21K_IMAGE_DIR`. Output: `$FEATURES_DIR/in21k/dinov3_vitb16/512/shards/shard_NNNN.pt`.

Recovery semantics (per the script's header):
- Atomic writes — shards land as `.tmp` then renamed to `.pt`; partial corruption is structurally prevented.
- Resume-safe — existing `.pt` files are skipped. Re-submit the same array to fill gaps.
- Self-describing — each shard carries `parquet_path`, `parquet_sha256`, `teacher_repo_id`, `image_size`, `shard_size`, `dtype`, `embed_dim`, `n_patches`, plus `git_commit`.

Monitor commands (from the script's header comment):
```bash
squeue -u $USER                                                  # job status
ls $FEATURES_DIR/in21k/dinov3_vitb16/512/shards/*.pt | wc -l     # completed shards
grep -l "Error\|Traceback" logs/export_*.out                     # failures
```

### Phase 2 — Pretraining

```bash
sbatch slurm/train.sbatch
```

The sbatch (comment at top of file) describes the default array as *"401 jobs × 4992 steps ≈ 2M steps (~8 days)"*. Each array task sources `slurm/env.sh`, runs `uv sync`, then `exec uv run python -m canvit_pretrain.train` with `--feature-base-dir $FEATURES_DIR/in21k --feature-image-root $IN21K_IMAGE_DIR --val-dir $IN1K_VAL_IMAGE_DIR --train-index-dir $INDEX_DIR --ckpt-dir $CHECKPOINTS_DIR` plus any extra flags you pass to `sbatch slurm/train.sbatch ...`. The task name + `$SLURM_ARRAY_JOB_ID` derives a stable `RUN_NAME=train-${SLURM_ARRAY_JOB_ID}` so subsequent array tasks resume from the same run directory.

Override patterns (from the sbatch usage block):
```bash
# CanViT-S variant:
sbatch slurm/train.sbatch --backbone-name vits16

# Continue existing run:
sbatch --array=0-99%1 slurm/train.sbatch --run-name train-XXXXXXX

# Quick test:
sbatch --array=0-2%1 --time=00:10:00 slurm/train.sbatch --steps-per-job 10
```

Every field in `canvit_pretrain.train.config.Config` becomes a `--kebab-case-flag` via `tyro`; the sbatch passes `"$@"` through to the Python command.

### Flagship hyperparameters

Defaults in `canvit_pretrain/train/config.py` (`Config` dataclass) for the published 2 M-step CanViT-B pretraining:

| Field | Default |
|---|---|
| `teacher_repo_id` | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| `backbone_name` | `vitb16` |
| `glimpse_grid_size` | 8 (so $128^2$ px glimpses at 16 px patches) |
| `canvas_patch_grid_size` | 32 |
| `scene_resolution` | 512 |
| `batch_size` | 64 |
| `peak_lr` | 4e-4 |
| `warmup_steps` | 100 000 |
| `cosine_total_steps` | `None` (constant LR after warmup; set to enable cosine decay) |
| `weight_decay` | 1e-4 (note in config: `1e-4` was used for the published 2M-step flagship run) |
| `min_viewpoint_scale` | 0.05 |
| `n_full_start_branches` | 1 (F-IID branch) |
| `n_random_start_branches` | 1 (R-IID branch) |
| `chunk_size` | 2 |
| `continue_prob` | 0.5 |
| `enable_scene_patches_loss` | True |
| `enable_scene_cls_loss` | True |
| `grad_clip` | 1.0 |
| `steps_per_job` | 4992 |
| `compile` | True |
| `amp` | True (bf16 autocast) |
| `val_every` | 1000 |
| `log_every` | 20 |
| `comet_project` | `canvit-pretrain` |

Teacher features for a new teacher / scene resolution / dataset combination require re-running Phase 1 first.

## Interactive dev

```bash
bash slurm/interactive.sh   # `salloc` wrapper (allocates a node and drops into a shell)
```

Inside the allocation, `.envrc` is loaded by `source slurm/env.sh` (which is what the sbatch also does). From there, the same `uv run python -m canvit_pretrain.train …` command the sbatch runs is available directly.

## Repository layout

```
canvit_pretrain/train/
  __main__.py, config.py, loop.py, step.py, model.py, scheduler.py,
  probe.py, transforms.py, viewpoint.py, ema.py, data/, viz/
canvit_pretrain/checkpoint/     # CheckpointData format
scripts/
  export_in21k_features.py      # Phase 1 entry
  push_ablation_checkpoints.py
  bench.py, bench_dataloader.py, inspect_ckpt.py, validate_features.py
slurm/
  train.sbatch                  # Phase 2 entry (array job)
  export_features.sh            # Phase 1 entry (array job)
  env.sh                        # sources .envrc, uv cache/venv, uv sync
  interactive.sh                # salloc wrapper
  ablations/                    # per-ablation sbatch variants
.envrc.nibi                     # env-var template (copy to .envrc on Nibi)
```

## Related repos

| Repo | Role |
|------|------|
| [CanViT-PyTorch](https://github.com/m2b3/CanViT-PyTorch) (public, canonical) | Core model + policies |
| [CanViT-specialize](https://github.com/m2b3/CanViT-specialize) | Probe training + datasets + IoU metrics + IN1K finetuning |
| [CanViT-eval](https://github.com/m2b3/CanViT-eval) | Evaluation (produces `.pt` result files) |
| [dinov3-in1k-probes](https://github.com/yberreby/dinov3-in1k-probes) | DINOv3 IN1K linear probes used as baselines |
| [CanViT-Toward-AVFMs](https://github.com/m2b3/CanViT-Toward-AVFMs) | Paper (`.pt` → JSON → PDF) |

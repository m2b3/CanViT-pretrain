# SA-1B Continual Pretraining

**Goal**: Continually pretrain CanViT-B flagship checkpoint (2M steps on IN21k @ 512px) at **1024px on SA-1B**.
First run: get loss on Comet, verify the model works at higher resolution.

**Branch**: `sa1b` (worktree: `~/code/CanViT-train-SA1B`)

---

## Architecture Decisions

### HF Seed Checkpoint
- **Source**: `canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02` on HF Hub
- **Format**: `config.json` + `model.safetensors` (239 keys) — NOT CheckpointData `.pt`
- **Config**: `additive` update mode, `enable_vpe=True`, `gate_bias_init=None`, `teacher_dim=768`
- **Solution**: `hf_seed_ckpt` config option. Downloads from HF, extracts state_dict + config, overrides `cfg.model`, proceeds with seed mode (step=0, fresh optimizer).
- **Grid size change**: 32 → 64. Only 6 standardizer keys mismatch (3 missing for "64", 3 unexpected for "32"). All 233 core weights load perfectly. Handled by existing `strict=False` + regex filter.
- **IMPORTANT**: `cfg.model` defaults are `convex` + `gate_bias_init=-2.0`. The HF model is `additive` + `gate_bias_init=None`. Using defaults crashes. The `hf_seed_ckpt` path MUST override `cfg.model` with the HF config.

### Storage Model
| What | Where | Persistence |
|---|---|---|
| SA-1B tars (~10.5 GB each) | `~/projects/def-akrish/datasets/sa1b/tars/` (NFS) | Permanent |
| SA-1B feature shards (~65 GB each) | `$SA1B_FEATURES_DIR/sa1b/dinov3_vitb16/1024/shards/` (NFS) | Permanent |
| Checkpoints | `$CHECKPOINTS_DIR/{run_name}/` (NFS) | Permanent |

### Data Loading Strategy
- Feature `.pt` shards on NFS, read with `mmap=True` (cheap, no copy).
- **Images read directly from mmap'd tar files** — no extraction step. `TarImageReader` builds a `{name: (offset, size)}` index via `tarfile` (~2s per 70 GB tar), then reads images via mmap slicing (~18ms/img).
- All DataLoader workers share mmap pages via fork COW. No RAM duplication.
- `--feature-base-dir "$SA1B_FEATURES_DIR/sa1b"` → shards path auto-constructed: `{base}/dinov3_vitb16/1024/shards/`.
- `--tar-dir "$SA1B_TAR_DIR"` → images read directly from tar files.
- Shard→tar mapping: `sa_000020.pt` → `sa_000020.tar` (via `shard_path.stem + ".tar"`).
- Shard loader iterates all shards, starts at `start_shard`, processes `steps_per_job` steps.

### Image Resolution
- **Teacher export**: 1024px input → features at 1024px (baked at export time).
- **Training images**: loaded at `scene_resolution=1024px`, same transform as export (`val_transform`).
- **Student glimpses**: 128px² crops from the loaded scene image. Higher scene resolution = more fine-grained info when student looks "up close".
- The dataloader image resolution affects glimpse pixel richness and CPU/GPU memory, NOT alignment with the teacher features (those are already fixed from export).
- No assertion currently validates `scene_resolution == G * patch_size`. For SA-1B: 1024 == 64 × 16 = 1024.

### Tar Structure
- SA-1B tars from Meta have `./` prefix: entries like `./sa_226692.jpg` (flat, no subdirectory).
- `TarImageReader._build_index` strips the prefix: `member.name.split("/", 1)[-1]`.
- **export_features.py** lists members with `tar tf` then extracts by exact path — unaffected by glob issues.

### Job Array Design
- `%1` concurrency (1 job at a time, like IN21k).
- `steps_per_job` = multiple of `batches_per_shard`. batches_per_shard = 11186 // 64 = **174**.
- Default: 1218 steps/job (7 shards). Override via `--steps-per-job`.
- No extraction step — Python handles everything. Startup is near-instant.

### Key Numbers
| Metric | Value | Source |
|---|---|---|
| Images per tar | ~11,186 | tar tvf on sa_000020 |
| JPEG size per tar | ~10.3 GB | tar tvf estimate |
| Feature shard size | ~65-70 GB | sa_000020.pt on def-areynaud |
| batch_size | 64 | config default |
| batches_per_shard | 174 | 11186 // 64 |
| shards_per_job | 7 | Default (1024px is ~4x more expensive/step) |
| steps_per_job | 1218 | 174 × 7 |
| Tar index build | ~2s | tarfile iteration on 70 GB tar |
| Image read (mmap) | ~18ms/img | mmap slice → BytesIO → PIL decode |
| Canvas grid size | 64 | 1024 / 16 |
| Scene resolution | 1024px | Target for SA-1B |
| VRAM estimate | <80 GB | 18.4 GB @ 512/32 × ~4x, plus constant factors |

---

## What's DONE

### Training Pipeline (commit `e1ef147`)
- `sa1b/train.sh` — SLURM array job. No extraction — images read from mmap'd tars.
- `canvit_pretrain/train/data/tar_images.py` — `TarImageReader`: tarfile index + mmap reads.
- `canvit_pretrain/train/data/shards.py` — `AllShardsDataset` supports `tar_dir` (SA-1B) or `image_root` (IN21k).
- `hf_seed_ckpt` config option in `loop.py` — downloads from HF, overrides cfg.model, seed mode.
- `strict=False` + regex filter for grid-size-change standardizer mismatches.
- Standardizers travel with model state_dict via `model.standardizers(G)`.

### Export Pipeline
- `sa1b/export_features.py` — 1 tar → 1 shard. Atomic save. Idempotent.
- `sa1b/export_features.sh` — SLURM array job. `gpu:h100:1`, 32G RAM, 16 CPUs, 10 min.
- `sa1b/submit_export.sh` — Auto-submits missing shards. `--dry-run` supported. **No lock on concurrent runs.**
- First shard exported: `sa_000020.pt` (70,394 MB, 11186 images).

### Download
- `sa1b/download.py` running on Nibi (tmux `sa1b-dl`). ~73 MB/s.
- Non-contiguous tar indices. submit_export.sh handles this.

### Env Setup
- `.envrc.nibi` committed. `SA1B_TAR_DIR`, `SA1B_FEATURES_DIR`, `CANVIT_FLAGSHIP_CKPT` defined.
- `sa1b/sa1b_links.tsv` committed (1000 image tars).

---

## What's PENDING

### 1. Re-export shards in tar order
- Old shards used **alphabetical** path order → random I/O across 70GB tars during training.
- New export (commit `d1bf94b`) saves paths in **tar file order** = sequential reads.
- Old shards preserved in `shards.old_alphabetical/`. New shards dir ready for re-export.
- **Job 9084285** (single shard sa_000020): PENDING. Verify output, then export remaining shards.

### 2. Training smoketest with tar-ordered shards
- Once export job completes, run training with new shard to compare data bottleneck.
- **Job 9084008 result**: Ran 172/175 steps, loss 1.07→1.06. Crashed with FileNotFoundError because we `mv`'d the shards dir while the job was running. Not a code bug — self-inflicted.
- Previous attempts: 9079933 (OOM), 9081051 (`.envrc` set-e bug), 9082081 (time limit — extraction ate all GPU time).

### 3. Full training run
- **REQUIRES HUMAN VALIDATION**: user must check Comet loss, VRAM, logs from smoketest before proceeding.
- Once validated, user submits with `--array=0-N%1` for real training.
- Warmup set to 2000 steps in train.sh (down from default 100K).

---

## Tensions / Open Questions

1. **Shuffling**: SA-1B images within a tar are geographically correlated (contiguous IDs = same region). No shuffle for now — verified visually that 4 sequential images look diverse enough. Revisit if loss curves show issues.
2. **Variable shard sizes**: `ShardedFeatureLoader` assumes uniform shard sizes (line 132). SA-1B shards are ~11,186 but not guaranteed identical. Should be fine for first run.
3. **Batch size at 1024px**: 64 @ 512px uses 18.4 GB. At 1024px/grid64, canvas is 4x larger. ~50-70 GB estimated. Should fit H100 80GB. May need to reduce if OOM.
4. **Training memory**: 48G requested. Shards and tars are mmap'd (shared via COW). Export needs 32G (mmap buffers).
5. **Export mmap optimization**: DONE (`8636703`). numpy.memmap on SLURM_TMPDIR, torch.from_numpy for save. `--mem` 96G → 32G.
6. **Warmup for continual pretraining**: Config has 100K warmup steps. For seed from pretrained model, this is very long (LR stays near 1e-7 for ~100K steps). May want shorter warmup or no warmup.
7. **`scene_resolution` vs `G * patch_size`**: These are independent. `scene_resolution` = pixel size of loaded training images. `G * patch_size` = teacher input resolution (baked at export time). They don't have to match — student sees images at `scene_resolution`, teacher features are precomputed.

---

## Bugs Found and Fixed

| Date | Commit | Bug | Impact |
|---|---|---|---|
| 2026-02-23 | `d1bf94b` | Export shards saved paths in alphabetical order (not tar order) | Random I/O across 70GB tars during training. Causes data loading bottleneck. |
| 2026-02-23 | `956f501` | End-of-job `save_checkpoint()` passed removed `scene_norm_state`/`cls_norm_state` kwargs | Would crash with TypeError after every training job — ALL training wasted |
| 2026-02-23 | `e1ef147` | Tar extraction blocked training for 8+ min on 10-min GPU job | Only 9 seconds of actual training before TIME LIMIT kill. Replaced with mmap tar reading. |
| 2026-02-23 | `956f501` | `train.sh` tar extraction: `'*.jpg'` doesn't match `./sa_226692.jpg` without `--wildcards` | Zero images extracted, training crashes on first image load |
| 2026-02-23 | `956f501` | `ShardedFeatureLoader.__init__` loaded first shard without `mmap=True` | ~70 GB loaded into RAM just to call `len()`, risking OOM on 96G nodes |
| 2026-02-23 | `47b6c9c` | `.envrc` credential `&&` chains return exit 1 under `set -e` | ALL submitted jobs (train + 47 exports) died in <1s. Zero work done. |
| 2026-02-23 | `47b6c9c` | `init_normalizer_stats_from_shard` loaded shard to GPU without mmap | Would OOM trying to load 70GB shard to H100. Even CPU would need 141GB float32. |

---

## Changelog

| Date | Commit | What |
|---|---|---|
| 2026-02-23 | `d1bf94b` | Export shards in tar order (not alphabetical), add data/gpu timing |
| 2026-02-23 | `e1ef147` | Read SA-1B images directly from mmap'd tars (no extraction) |
| 2026-02-23 | `5b1e45a` | Make normalizer_max_samples a config parameter |
| 2026-02-23 | `47b6c9c` | Fix .envrc set-e crash + normalizer shard OOM |
| 2026-02-23 | `a8813b6` | Reduce train.sh: workers 16→4, cpus 16→8, steps 4872→1218 |
| 2026-02-23 | `8636703` | Mmap export buffers, reduce --mem 96G→32G (export+train) |
| 2026-02-23 | `f3051a9` | Remove dead build_parquet.py |
| 2026-02-23 | `956f501` | Fix 3 bugs: checkpoint crash, tar extraction, shard OOM |
| 2026-02-23 | `2a2e316` | Add `hf_seed_ckpt` config + loop support, plan_job.py, train.sh |
| 2026-02-23 | `3539567` | Add --max-concurrent to submit_export.sh |
| 2026-02-22 | `21e9bbc` | Unify standardizers: model.standardizers(G) in loop |
| 2026-02-22 | various | Export pipeline (export_features.py/.sh, submit_export.sh) |

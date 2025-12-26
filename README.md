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

**Current state (Dec 2025)**: Policy learning is experimental. Might take long pretraining to start mattering. The branching structure may evolve.

## Architecture Split

- **`avp_vit/`** = training code, checkpointing, this repo
- **`canvit`** = model architecture, lives in `.venv/.../site-packages/canvit/`

**Before assuming anything about the model, read canvit source.** It's a separate dependency.

## Discovery

```bash
git ls-files                    # this repo's structure
uv run pypatree                 # this repo with signatures
ls .venv/.../site-packages/canvit/  # model architecture
```

## Entry Points

- Training: `avp_vit/train/__main__.py`
- Inference demo: `scripts/inference_app.py` (Streamlit, good reference)
- Config: `avp_vit/train/config.py`
- FLOP analysis: `scripts/flops.py`

## Pitfalls

- **DINOv3 exists and is not DINOv2** - not a typo, not the same. Don't assume. Ask or check the code.
- **Never hardcode, never assume** - model names, dimensions, conventions. Read the actual code.
- **Training-inference mismatch is subtle** - deep learning is brittle. Sometimes we aim for zero-shot generalization, sometimes we accept it's not possible. Be explicit about assumptions.
- **Coordinate conventions vary** - internal vs external APIs vs PyTorch grid_sample. Check canvit source in venv (or https://github.com/m2b3/CanViT/ - private for now): `viewpoint/` and `coords/`.
- **Normalizer states required for inference** - checkpoints include `scene_norm_state`, `cls_norm_state`. See `avp_vit/train/norm.py`.
- **GPU syncs kill performance** - `.item()`, `.cpu()`, logging inside hot loops. Keep logging outside.
- **Logging is essential** - but outside hot loops. We want visibility, not sync stalls.

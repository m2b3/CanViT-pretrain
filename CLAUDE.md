# AVP-ViT Development Guide

## Context

Research project, started September 2024. Mostly one person. High cognitive load.

This repo (`avp-vit`) is experimental - training, viz, monitoring code that evolves rapidly. We accept the mess, gradually clean/extract/modularize.

`canvit` (separate repo, in venv) is the core architecture - stabler, cleaner API, geared for future public release. **Will not merge back.** The split is intentional: core arch evolves slower than experiment code.

Everything can change. Be ready.

See @README.md for structure, entry points, and implementation details.

## Session Startup

**Do this EVERY session before anything else:**

Load the pytorch skill first, then:
```bash
git status
git log --oneline -10
uv run pypatree
```

Check canvit source in venv - model architecture lives there, not here.

**If anything is unclear or doesn't match expectations, STOP and ask the user.** Do not proceed with confusion.

## Principles

- Measure twice, cut once
- Verify, don't assume
  - NEVER GUESS APIs.
- State hypotheses before investigating
- If you change code that would make README.md misleading, update it

## DINOv3 Notes

**DINOv3 ≠ DINOv2.** Different model, different codebase.

- **Patch size is 16px**, always. Do NOT assume 14px like DINOv2.
- Access via `.patch_size_px` attribute on any `DINOv3Backbone` instance (whether called `backbone`, `teacher`, etc.)
- **Never hardcode patch size.** Always read from the backbone object.

**To verify DINOv3 API:**
```bash
uv run ipython -c "
from canvit import create_backbone
bb = create_backbone('dinov3_vitb16', pretrained=False)
print(dir(bb))  # see available attributes
"
```

Or read source: `.venv/lib/python3.12/site-packages/dinov3/` and `canvit/backbone/dinov3.py`.

## CanViT Model Architecture

**⚠️ VERIFY THIS SECTION AGAINST SOURCE CODE** — it can go stale. Check `avp_vit/__init__.py` and canvit source in `.venv/lib/python3.12/site-packages/canvit/`.

**Class hierarchy**:
```
CanViT (canvit/model/base/)
  → init_state, init_canvas, forward, forward_reduce, get_spatial

CanViTForPretraining (canvit/model/pretraining/)
  → predict_teacher_scene, predict_scene_teacher_cls
```

**RecurrentState** (passed between timesteps):
```python
RecurrentState(
    canvas: Tensor,        # [B, n_canvas_regs + G², canvas_dim]
    recurrent_cls: Tensor  # [B, 1, local_dim]
)
```

**Canvas layout**: `[registers | spatial]`. Use `model.get_spatial(canvas)` to extract spatial tokens.

**Viewpoint coordinates**: `centers` is `[cy, cx]` — **y first**, normalized to `[-1, 1]`.

## Commands

```bash
uv run pypatree                              # structure
uv run -m avp_vit.train                      # training
COMET_API_KEY=$(cat ~/comet_api_key.txt) uv run ...
uv run ipython -c "..."                      # quick experiments
```

## Conventions

- NEVER `git add -A` or `git add -u`
- Directory structure: `avp_vit/mymodule/{__init__.py,test.py,...}`
- `assert isinstance(...)` over `cast` or `type: ignore`

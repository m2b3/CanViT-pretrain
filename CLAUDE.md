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

Check canvit source in venv - model architecture lives there, not here. Read `inference_app/` for how pieces connect.

**If anything is unclear or doesn't match expectations, STOP and ask the user.** Do not proceed with confusion.

## Principles

- Measure twice, cut once
- Verify, don't assume
  - NEVER GUESS APIs.
- State hypotheses before investigating
- If you change code that would make README.md misleading, update it

## Commands

```bash
uv run pypatree                              # structure
uv run -m avp_vit.train                      # training
uv run streamlit run inference_app/__main__.py # demo
COMET_API_KEY=$(cat ~/comet_api_key.txt) uv run ...
uv run ipython -c "..."                      # quick experiments
```

## Conventions

- NEVER `git add -A` or `git add -u`
- Directory structure: `avp_vit/mymodule/{__init__.py,test.py,...}`
- `assert isinstance(...)` over `cast` or `type: ignore`

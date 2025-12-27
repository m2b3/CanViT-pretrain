# AVP-ViT Development Guide

See @README.md for core principles and architecture split.

## Session Startup

**Do this EVERY session before anything else:**

Load the pytorch skill first, then:
```bash
git status
git log --oneline -10
uv run pypatree
```

Check canvit source in venv - model architecture lives there, not here. Read `scripts/inference_app.py` for how pieces connect.

**If anything is unclear or doesn't match expectations, STOP and ask the user.** Do not proceed with confusion.

## Principles

- Measure twice, cut once
- Verify, don't assume
- State hypotheses before investigating
- If you change code that would make README.md misleading, update it

## Commands

```bash
uv run pypatree                              # structure
uv run -m avp_vit.train                      # training
uv run streamlit run scripts/inference_app.py # demo
COMET_API_KEY=$(cat ~/comet_api_key.txt) uv run ...
uv run ipython -c "..."                      # quick experiments
```

## Conventions

- NEVER `git add -A` or `git add -u`
- Directory structure: `avp_vit/mymodule/{__init__.py,test.py,...}`
- `assert isinstance(...)` over `cast` or `type: ignore`

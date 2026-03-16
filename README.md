# CanViT-pretrain

Pretraining for CanViT: passive-to-active dense latent distillation from DINOv3.

## Usage

```bash
# Pretraining
uv run python -m canvit_pretrain.train --help

# Feature export (precompute DINOv3 teacher features)
uv run python scripts/export_in21k_features.py --help
```

## Architecture

```bash
uv run pypatree
```

## Related repos

| Repo | Role |
|------|------|
| [CanViT-PyTorch-Next](https://github.com/yberreby/CanViT-PyTorch-Next) | Core model + policies |
| [CanViT-probes](https://github.com/m2b3/CanViT-probes) | Probes, datasets, metrics, probe training |
| [CanViT-eval](https://github.com/m2b3/CanViT-eval) | Evaluation (produces .pt result files) |
| [CanViT-Toward-AVFMs](https://github.com/m2b3/CanViT-Toward-AVFMs) | Paper (.pt → JSON → PDF) |

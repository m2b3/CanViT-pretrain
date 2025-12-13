"""Tests for IJEPABackend - must match native forward exactly."""
import torch
from src.models.vision_transformer import vit_small

from . import IJEPABackend


def test_backend_forward_matches_native():
    """Backend block-by-block forward must exactly match native forward."""
    torch.manual_seed(42)
    backbone = vit_small(img_size=[112], patch_size=16)
    backbone.eval()

    backend = IJEPABackend(backbone)

    B = 2
    img = torch.randn(B, 3, 112, 112)

    with torch.no_grad():
        native_out = backbone(img, masks=None)

    x, H, W = backend.prepare_tokens(img)

    with torch.no_grad():
        for i in range(backend.n_blocks):
            x = backend.forward_block(i, x, rope=None)
        x = backbone.norm(x)

    assert torch.allclose(x, native_out, atol=1e-5)


def test_backend_properties():
    """Backend exposes correct properties from backbone."""
    backbone = vit_small(img_size=[112], patch_size=16)
    backend = IJEPABackend(backbone)

    assert backend.embed_dim == 384
    assert backend.num_heads == 6
    assert backend.n_blocks == 12
    assert backend.n_prefix_tokens == 0
    assert backend.rope_periods.shape[0] == backend.embed_dim // backend.num_heads // 4

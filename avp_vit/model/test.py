from typing import override

import torch
from torch import Tensor, nn

from avp_vit.backbone import ViTBackbone
from avp_vit.model import AVPConfig, AVPViT
from avp_vit.rope import make_rope_periods


class MockBackbone(ViTBackbone, nn.Module):
    """Minimal backbone for unit testing AVPViT without real weights."""

    _embed_dim: int
    _num_heads: int
    _n_blocks: int
    _rope_periods: Tensor
    _norm: nn.LayerNorm

    def __init__(self, embed_dim: int, num_heads: int, n_blocks: int) -> None:
        nn.Module.__init__(self)
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._n_blocks = n_blocks
        self._norm = nn.LayerNorm(embed_dim)
        head_dim = embed_dim // num_heads
        self.register_buffer("_rope_periods", make_rope_periods(head_dim))

    @property
    @override
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    @override
    def norm(self) -> nn.LayerNorm:
        return self._norm

    @property
    @override
    def num_heads(self) -> int:
        return self._num_heads

    @property
    @override
    def n_prefix_tokens(self) -> int:
        return 1

    @property
    @override
    def n_blocks(self) -> int:
        return self._n_blocks

    @property
    @override
    def rope_periods(self) -> Tensor:
        return self._rope_periods

    @property
    @override
    def rope_dtype(self) -> torch.dtype:
        return torch.float32

    @override
    def forward_block(
        self, idx: int, x: Tensor, rope: tuple[Tensor, Tensor] | None
    ) -> Tensor:
        return x

    @override
    def prepare_tokens(self, images: Tensor) -> tuple[Tensor, int, int]:
        raise NotImplementedError


def test_forward_shapes():
    embed_dim, num_heads, n_blocks = 64, 4, 2
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3)
    backbone = MockBackbone(embed_dim, num_heads, n_blocks)
    avp = AVPViT(backbone, cfg)

    B, n_prefix, n_patches = 2, 1, 9
    local = torch.randn(B, n_prefix + n_patches, embed_dim)
    centers = torch.rand(B, 2)
    scales = torch.rand(B)

    out_local, out_scene = avp(local, centers, scales)

    assert out_local.shape == local.shape
    assert out_scene.shape == (B, 16, embed_dim)


def test_gate_init():
    cfg = AVPConfig(scene_grid_size=4, gate_init=0.5)
    backbone = MockBackbone(64, 4, 2)
    avp = AVPViT(backbone, cfg)

    for g in avp.read_gate:
        assert (g == 0.5).all()
    for g in avp.write_gate:
        assert (g == 0.5).all()

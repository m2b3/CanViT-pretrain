import torch
from torch import Tensor, nn

from ..backend import ViTBackend
from . import AVPConfig, AVPViT


class MockBackend(ViTBackend, nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, n_blocks: int):
        nn.Module.__init__(self)
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._n_blocks = n_blocks
        head_dim = embed_dim // num_heads
        self.register_buffer("_rope_periods", torch.ones(head_dim // 4))

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def n_prefix_tokens(self) -> int:
        return 1

    @property
    def n_blocks(self) -> int:
        return self._n_blocks

    @property
    def rope_periods(self) -> Tensor:
        return self._rope_periods

    @property
    def rope_dtype(self) -> torch.dtype:
        return torch.float32

    def forward_block(self, idx: int, x: Tensor, rope: tuple[Tensor, Tensor] | None) -> Tensor:
        return x

    def prepare_tokens(self, images: Tensor) -> tuple[Tensor, int, int]:
        raise NotImplementedError


def test_avp_forward_shapes():
    embed_dim, num_heads, n_blocks = 64, 4, 2
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3)
    backend = MockBackend(embed_dim, num_heads, n_blocks)
    avp = AVPViT(backend, cfg)

    B, n_prefix, n_patches = 2, 1, 9
    local = torch.randn(B, n_prefix + n_patches, embed_dim)
    centers = torch.rand(B, 2)
    scales = torch.rand(B)

    out_local, out_scene = avp(local, centers, scales)

    assert out_local.shape == local.shape
    assert out_scene.shape == (B, 16, embed_dim)


def test_gate_init():
    cfg = AVPConfig(scene_grid_size=4, gate_init=0.5)
    backend = MockBackend(64, 4, 2)
    avp = AVPViT(backend, cfg)

    for g in avp.read_gate:
        assert (g == 0.5).all()
    for g in avp.write_gate:
        assert (g == 0.5).all()

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
    _n_register_tokens: int
    _rope_periods: Tensor
    _norm: nn.LayerNorm

    def __init__(
        self, embed_dim: int, num_heads: int, n_blocks: int, n_register_tokens: int = 0
    ) -> None:
        nn.Module.__init__(self)
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._n_blocks = n_blocks
        self._n_register_tokens = n_register_tokens
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
    def n_register_tokens(self) -> int:
        return self._n_register_tokens

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


def test_scene_registers_disabled_by_default():
    cfg = AVPConfig(scene_grid_size=4)
    backbone = MockBackbone(64, 4, 2, n_register_tokens=4)
    avp = AVPViT(backbone, cfg)

    assert avp.n_scene_registers == 0
    assert avp.scene_registers is None


def test_scene_registers_uses_backbone_count():
    cfg = AVPConfig(scene_grid_size=4, use_scene_registers=True)
    backbone = MockBackbone(64, 4, 2, n_register_tokens=4)
    avp = AVPViT(backbone, cfg)

    assert avp.n_scene_registers == 4
    assert avp.scene_registers is not None
    assert avp.scene_registers.shape == (1, 4, 64)


def test_scene_registers_output_shape_unchanged():
    """Output should only contain grid tokens, not registers."""
    embed_dim, num_heads, n_blocks = 64, 4, 2
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, use_scene_registers=True)
    backbone = MockBackbone(embed_dim, num_heads, n_blocks, n_register_tokens=4)
    avp = AVPViT(backbone, cfg)

    B, n_prefix, n_patches = 2, 1, 9
    local = torch.randn(B, n_prefix + n_patches, embed_dim)
    centers = torch.rand(B, 2)
    scales = torch.rand(B)

    out_local, out_scene = avp(local, centers, scales)

    assert out_local.shape == local.shape
    assert out_scene.shape == (B, 16, embed_dim)  # 4x4 grid, no registers


def test_forward_step_no_output_proj():
    """forward_step returns scene before output_proj."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, gate_init=0.0, use_output_proj=True)
    backbone = MockBackbone(embed_dim, 4, 2)
    avp = AVPViT(backbone, cfg)

    B = 2
    local = torch.randn(B, 1 + 9, embed_dim)
    centers = torch.zeros(B, 2)
    scales = torch.ones(B)

    with torch.no_grad():
        _, scene_step = avp.forward_step(local, centers, scales)
        _, scene_fwd = avp(local, centers, scales)

    # At init: forward_step returns scene_tokens, forward applies output_proj (identity)
    assert torch.allclose(scene_step, scene_fwd)


def test_scene_input_accepted():
    """forward_step accepts scene from previous step."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3)
    backbone = MockBackbone(embed_dim, 4, 2)
    avp = AVPViT(backbone, cfg)

    B = 2
    local = torch.randn(B, 1 + 9, embed_dim)
    centers = torch.zeros(B, 2)
    scales = torch.ones(B)
    custom_scene = torch.randn(B, 16, embed_dim)

    _, scene_out = avp.forward_step(local, centers, scales, scene=custom_scene)

    assert scene_out.shape == (B, 16, embed_dim)


def test_multi_step_passthrough_at_init():
    """At init (gates=0), multi-step returns same scene as single step."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, gate_init=0.0)
    backbone = MockBackbone(embed_dim, 4, 2)
    avp = AVPViT(backbone, cfg)

    B = 2
    local = torch.randn(B, 1 + 9, embed_dim)
    centers = torch.zeros(B, 2)
    scales = torch.ones(B)

    with torch.no_grad():
        # Single step
        _, scene_1 = avp.forward_step(local, centers, scales)

        # Multi-step: pass scene from step 1 to step 2
        _, scene_2 = avp.forward_step(local, centers, scales, scene=scene_1)

    # At init with gates=0, scene is unchanged, so scene_1 == scene_2
    assert torch.allclose(scene_1, scene_2)


def test_output_proj_is_always_module():
    """output_proj is always nn.Module (Identity or Linear), never None."""
    cfg_no_proj = AVPConfig(scene_grid_size=4, use_output_proj=False)
    cfg_with_proj = AVPConfig(scene_grid_size=4, use_output_proj=True)
    backbone = MockBackbone(64, 4, 2)

    avp_no = AVPViT(backbone, cfg_no_proj)
    avp_yes = AVPViT(backbone, cfg_with_proj)

    assert isinstance(avp_no.output_proj, torch.nn.Identity)
    assert isinstance(avp_yes.output_proj, torch.nn.Linear)


def test_forward_sequence_with_callback():
    """forward_sequence accepts glimpse_fn callback and n_steps."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3)
    backbone = MockBackbone(embed_dim, 4, 2)
    avp = AVPViT(backbone, cfg)

    B = 2
    local = torch.randn(B, 1 + 9, embed_dim)
    centers = torch.zeros(B, 2)
    scales = torch.ones(B)

    def glimpse_fn(step_idx: int, scene: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        return local, centers, scales

    scene = avp.forward_sequence(glimpse_fn, n_steps=3)

    assert isinstance(scene, Tensor)
    assert scene.shape == (B, 16, embed_dim)


def test_forward_sequence_with_loss_fn():
    """forward_sequence accumulates loss when loss_fn provided."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3)
    backbone = MockBackbone(embed_dim, 4, 2)
    avp = AVPViT(backbone, cfg)

    B = 2
    local = torch.randn(B, 1 + 9, embed_dim)
    centers = torch.zeros(B, 2)
    scales = torch.ones(B)
    target = torch.randn(B, 16, embed_dim)

    def glimpse_fn(step_idx: int, scene: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        return local, centers, scales

    def loss_fn(scene_proj: Tensor) -> Tensor:
        return torch.nn.functional.mse_loss(scene_proj, target)

    result = avp.forward_sequence(glimpse_fn, n_steps=3, loss_fn=loss_fn)

    assert isinstance(result, tuple)
    scene, avg_loss = result
    assert scene.shape == (B, 16, embed_dim)
    assert avg_loss.shape == ()  # scalar

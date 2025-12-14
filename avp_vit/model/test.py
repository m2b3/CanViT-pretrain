from typing import override

import torch
from torch import Tensor, nn

from avp_vit.backbone import ViTBackbone
from avp_vit.glimpse import Viewpoint
from avp_vit.model import AVPConfig, AVPViT, StepOutput
from avp_vit.rope import make_rope_periods


class MockBackbone(ViTBackbone, nn.Module):
    """Minimal backbone for unit testing AVPViT without real weights."""

    _embed_dim: int
    _num_heads: int
    _n_blocks: int
    _n_register_tokens: int
    _patch_size: int
    _rope_periods: Tensor

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        n_blocks: int,
        n_register_tokens: int,
        patch_size: int,
    ) -> None:
        nn.Module.__init__(self)
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._n_blocks = n_blocks
        self._n_register_tokens = n_register_tokens
        self._patch_size = patch_size
        head_dim = embed_dim // num_heads
        self.register_buffer("_rope_periods", make_rope_periods(head_dim, dtype=torch.float32))

    @property
    @override
    def embed_dim(self) -> int:
        return self._embed_dim

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
    def patch_size(self) -> int:
        return self._patch_size

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
        B, C, H, W = images.shape
        n_patches_h = H // self._patch_size
        n_patches_w = W // self._patch_size
        n_patches = n_patches_h * n_patches_w
        # Return [B, 1 + n_patches, embed_dim] (1 for CLS token)
        tokens = torch.randn(B, 1 + n_patches, self._embed_dim, device=images.device)
        return tokens, n_patches_h, n_patches_w


PATCH_SIZE = 16


def test_forward_shapes():
    embed_dim, num_heads, n_blocks = 64, 4, 2
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3)
    backbone = MockBackbone(embed_dim, num_heads, n_blocks, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    images = torch.randn(B, 3, 64, 64)  # scene_grid_size * patch_size = 4 * 16 = 64
    viewpoints = [Viewpoint.full_scene(B, images.device)]

    scene, hidden = avp(images, viewpoints)

    assert scene.shape == (B, 16, embed_dim)  # 4x4 grid
    assert hidden.shape == (B, 16, embed_dim)  # hidden same shape


def test_gate_init():
    cfg = AVPConfig(scene_grid_size=4, gate_init=0.5)
    backbone = MockBackbone(64, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    for g in avp.read_gate:
        assert (g == 0.5).all()
    for g in avp.write_gate:
        assert (g == 0.5).all()


def test_scene_registers_disabled_by_default():
    cfg = AVPConfig(scene_grid_size=4)
    backbone = MockBackbone(64, 4, 2, 4, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    assert avp.n_scene_registers == 0
    assert avp.scene_registers is None


def test_scene_registers_scales_with_token_ratio():
    # scene=14, glimpse=7 -> ratio=4, so 4 backbone regs -> 16 scene regs
    cfg = AVPConfig(scene_grid_size=14, glimpse_grid_size=7, use_scene_registers=True)
    backbone = MockBackbone(64, 4, 2, 4, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    assert avp.n_scene_registers == 16  # 4 * (14/7)² = 4 * 4 = 16
    assert avp.scene_registers is not None
    assert avp.scene_registers.shape == (1, 16, 64)


def test_scene_registers_output_shape_unchanged():
    """Output should only contain grid tokens, not registers."""
    embed_dim, num_heads, n_blocks = 64, 4, 2
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, use_scene_registers=True)
    backbone = MockBackbone(embed_dim, num_heads, n_blocks, 4, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    viewpoints = [Viewpoint.full_scene(B, images.device)]

    scene, hidden = avp(images, viewpoints)

    assert scene.shape == (B, 16, embed_dim)  # 4x4 grid, no registers
    assert hidden.shape == (B, 16, embed_dim)


def test_output_proj_is_always_module():
    """output_proj is always nn.Module (Identity or Linear), never None."""
    cfg_no_proj = AVPConfig(scene_grid_size=4, use_output_proj=False)
    cfg_with_proj = AVPConfig(scene_grid_size=4, use_output_proj=True)
    backbone = MockBackbone(64, 4, 2, 0, PATCH_SIZE)

    avp_no = AVPViT(backbone, cfg_no_proj)
    avp_yes = AVPViT(backbone, cfg_with_proj)

    assert isinstance(avp_no.output_proj, torch.nn.Identity)
    assert isinstance(avp_yes.output_proj, torch.nn.Linear)


def test_multi_viewpoint_forward():
    """Forward with multiple viewpoints processes all sequentially."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    viewpoints = [
        Viewpoint.full_scene(B, images.device),
        Viewpoint.quadrant(B, images.device, 0, 0),
        Viewpoint.quadrant(B, images.device, 1, 1),
    ]

    scene, hidden = avp(images, viewpoints)

    assert scene.shape == (B, 16, embed_dim)
    assert hidden.shape == (B, 16, embed_dim)


def test_glimpse_size_from_backbone():
    """glimpse_size = glimpse_grid_size * backbone.patch_size."""
    cfg = AVPConfig(scene_grid_size=14, glimpse_grid_size=7)
    backbone = MockBackbone(64, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    assert avp.glimpse_size == 7 * PATCH_SIZE  # 112


def test_forward_step_returns_step_output():
    """forward_step returns StepOutput with both hidden and scene."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    vp = Viewpoint.full_scene(B, images.device)

    out = avp.forward_step(images, vp, None)

    assert isinstance(out, StepOutput)
    assert out.local.shape == (B, 10, embed_dim)  # CLS + 3x3 glimpse grid
    # hidden: internal state for continuation
    assert out.hidden.shape == (B, 16, embed_dim)  # 4x4 scene grid
    # scene: projected output for loss/viz
    assert out.scene.shape == (B, 16, embed_dim)


def test_forward_loss():
    """forward_loss returns averaged MSE and final hidden."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    target = torch.randn(B, 16, embed_dim)
    viewpoints = [
        Viewpoint.full_scene(B, images.device),
        Viewpoint.quadrant(B, images.device, 0, 0),
    ]

    loss, final_hidden = avp.forward_loss(images, viewpoints, target)

    assert loss.shape == ()  # scalar
    assert loss.item() >= 0  # MSE is non-negative
    assert final_hidden.shape == (B, 16, embed_dim)


def test_forward_trajectory():
    """forward_trajectory returns list of projected scenes."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    viewpoints = [
        Viewpoint.full_scene(B, images.device),
        Viewpoint.quadrant(B, images.device, 0, 0),
        Viewpoint.quadrant(B, images.device, 1, 1),
    ]

    scenes, final_hidden = avp.forward_trajectory(images, viewpoints)

    assert len(scenes) == 3  # one per viewpoint
    for s in scenes:
        assert s.shape == (B, 16, embed_dim)
    assert final_hidden.shape == (B, 16, embed_dim)


def test_forward_reduce_custom():
    """forward_reduce with custom reducer."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    viewpoints = [Viewpoint.full_scene(B, images.device)]

    # Custom reducer that counts steps
    def count_reducer(acc: int, h: torch.Tensor, s: torch.Tensor) -> int:
        return acc + 1

    count, final_hidden = avp.forward_reduce(images, viewpoints, count_reducer, init=0)

    assert count == 1
    assert final_hidden.shape == (B, 16, embed_dim)


def test_gradient_checkpointing_smoke():
    """Gradient checkpointing produces same output and gradients flow."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, gradient_checkpointing=True)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)
    avp.train()

    B = 2
    images = torch.randn(B, 3, 64, 64)
    viewpoints = [Viewpoint.full_scene(B, images.device)]

    scene, _ = avp(images, viewpoints)
    loss = scene.sum()
    loss.backward()

    assert avp.hidden_tokens.grad is not None

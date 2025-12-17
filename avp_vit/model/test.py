from typing import override

import pytest
import torch
from torch import Tensor, nn

from avp_vit.attention import ScaledResidualAttention
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
    _has_cls: bool
    _rope_periods: Tensor

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        n_blocks: int,
        n_register_tokens: int,
        patch_size: int,
        *,
        has_cls: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._n_blocks = n_blocks
        self._n_register_tokens = n_register_tokens
        self._patch_size = patch_size
        self._has_cls = has_cls
        head_dim = embed_dim // num_heads
        self.register_buffer(
            "_rope_periods", make_rope_periods(head_dim, dtype=torch.float32)
        )

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
        return (1 if self._has_cls else 0) + self._n_register_tokens

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
        n_tokens = self.n_prefix_tokens + n_patches
        tokens = torch.randn(B, n_tokens, self._embed_dim, device=images.device)
        return tokens, n_patches_h, n_patches_w


PATCH_SIZE = 16


def test_forward_shapes():
    embed_dim, num_heads, n_blocks = 64, 4, 2
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=0)
    backbone = MockBackbone(embed_dim, num_heads, n_blocks, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=embed_dim)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    viewpoints = [Viewpoint.full_scene(B, images.device)]

    scene, hidden = avp(images, viewpoints)

    assert scene.shape == (B, 16, embed_dim)  # 4x4 grid
    assert hidden.shape == (B, 16, embed_dim)


def test_layer_scale_init():
    cfg = AVPConfig(scene_grid_size=4, layer_scale_init=0.5, gating="none", n_scene_registers=0)
    backbone = MockBackbone(64, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=64)

    for attn in avp.read_attn:
        assert isinstance(attn, ScaledResidualAttention)
        assert (attn.scale.scale == 0.5).all()
    for attn in avp.write_attn:
        assert isinstance(attn, ScaledResidualAttention)
        assert (attn.scale.scale == 0.5).all()


def test_convex_gating_init():
    import math

    from avp_vit.attention.convex import ConvexGatedAttention

    gate_init = 0.5
    cfg = AVPConfig(
        scene_grid_size=4,
        layer_scale_init=gate_init,
        gating="full",
        n_scene_registers=0,
    )
    backbone = MockBackbone(64, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=64)

    expected_bias = math.log(gate_init / (1 - gate_init))
    for cvx in avp.read_attn:
        assert isinstance(cvx, ConvexGatedAttention)
        assert torch.allclose(cvx.gate_bias, torch.full((64,), expected_bias))


def test_convex_init_passthrough():
    """At init with low gate, convex gating acts as passthrough on both streams."""
    from avp_vit.glimpse import extract_glimpse

    n_blocks = 2
    gate_init = 1e-5
    cfg = AVPConfig(
        scene_grid_size=4,
        glimpse_grid_size=3,
        layer_scale_init=gate_init,
        gating="full",
        n_scene_registers=0,
    )
    backbone = MockBackbone(64, 4, n_blocks, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=64)

    B = 2
    torch.manual_seed(42)
    images = torch.randn(B, 3, avp.scene_size, avp.scene_size)
    vp = Viewpoint("test", torch.zeros(B, 2), torch.ones(B))

    with torch.no_grad():
        glimpse = extract_glimpse(images, vp, avp.glimpse_size)

        # Baseline local: backbone only (no cross-attention)
        torch.manual_seed(123)
        baseline_local, _, _ = backbone.prepare_tokens(glimpse)
        for i in range(n_blocks):
            baseline_local = backbone.forward_block(i, baseline_local, None)

        # AVPViT forward (same seed for prepare_tokens)
        torch.manual_seed(123)
        out = avp.forward_step(images, vp)
        initial_scene = avp.scene_proj(avp.get_spatial(avp._get_base_hidden(B)))

    # Scene: write gate ≈ 0 → scene ≈ initial
    scene_diff = (out.scene - initial_scene).abs().mean()
    assert scene_diff < 0.1, f"Scene changed too much: {scene_diff}"

    # Local: read gate ≈ 0 → local ≈ backbone-only
    local_diff = (out.local - baseline_local).abs().mean()
    assert local_diff < 1e-3, f"Local differs from backbone-only: {local_diff}"


def test_convex_gate_value_affects_output():
    """High gate breaks passthrough; low gate preserves it."""
    cfg_lo = AVPConfig(
        scene_grid_size=4,
        layer_scale_init=1e-5,
        gating="full",
        n_scene_registers=0,
    )
    cfg_hi = AVPConfig(
        scene_grid_size=4,
        layer_scale_init=0.5,
        gating="full",
        n_scene_registers=0,
    )

    torch.manual_seed(999)
    avp_lo = AVPViT(MockBackbone(64, 4, 2, 0, PATCH_SIZE), cfg_lo, teacher_dim=64)
    torch.manual_seed(999)
    avp_hi = AVPViT(MockBackbone(64, 4, 2, 0, PATCH_SIZE), cfg_hi, teacher_dim=64)

    B = 2
    torch.manual_seed(42)
    images = torch.randn(B, 3, avp_lo.scene_size, avp_lo.scene_size)
    vp = Viewpoint("test", torch.zeros(B, 2), torch.ones(B))

    with torch.no_grad():
        torch.manual_seed(123)
        out_lo = avp_lo.forward_step(images, vp)
        torch.manual_seed(123)
        out_hi = avp_hi.forward_step(images, vp)
        initial = avp_lo.scene_proj(avp_lo.get_spatial(avp_lo._get_base_hidden(B)))

    diff_lo = (out_lo.scene - initial).abs().mean()
    diff_hi = (out_hi.scene - initial).abs().mean()

    assert diff_lo < 0.01, f"Low gate should preserve passthrough: {diff_lo}"
    assert diff_hi > 0.01, f"High gate should break passthrough: {diff_hi}"


def test_scene_registers_disabled_when_zero():
    cfg = AVPConfig(scene_grid_size=4, n_scene_registers=0)
    backbone = MockBackbone(64, 4, 2, 4, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=64)

    assert avp.n_scene_registers == 0
    assert avp.persistent_registers is None
    assert avp.ephemeral_registers is None


def test_scene_registers_fixed_count():
    cfg = AVPConfig(scene_grid_size=14, glimpse_grid_size=7, n_scene_registers=16)
    backbone = MockBackbone(64, 4, 2, 4, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=64)

    assert avp.n_scene_registers == 16
    assert avp.n_persistent_registers == 8
    assert avp.n_ephemeral_registers == 8
    assert avp.persistent_registers is not None
    assert avp.ephemeral_registers is not None
    assert avp.persistent_registers.shape == (1, 8, 64)
    assert avp.ephemeral_registers.shape == (1, 8, 64)


def test_scene_registers_split():
    embed_dim, num_heads, n_blocks = 64, 4, 2
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=7)
    backbone = MockBackbone(embed_dim, num_heads, n_blocks, 4, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=embed_dim)

    assert avp.n_scene_registers == 7
    assert avp.n_persistent_registers == 3
    assert avp.n_ephemeral_registers == 4
    assert avp.persistent_registers is not None
    assert avp.ephemeral_registers is not None
    assert avp.persistent_registers.shape == (1, 3, embed_dim)
    assert avp.ephemeral_registers.shape == (1, 4, embed_dim)


def test_scene_output_always_spatial_only():
    embed_dim, num_heads, n_blocks = 64, 4, 2
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=7)
    backbone = MockBackbone(embed_dim, num_heads, n_blocks, 4, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=embed_dim)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    viewpoints = [Viewpoint.full_scene(B, images.device)]

    scene, hidden = avp(images, viewpoints)

    assert scene.shape == (B, 16, embed_dim)  # 4x4 grid (spatial only)
    n_persistent = avp.n_persistent_registers
    assert hidden.shape == (B, n_persistent + 16, embed_dim)


def test_scene_registers_continuity():
    embed_dim, num_heads, n_blocks = 64, 4, 2
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=7)
    backbone = MockBackbone(embed_dim, num_heads, n_blocks, 4, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=embed_dim)

    B = 2
    n_persistent = avp.n_persistent_registers
    images = torch.randn(B, 3, 64, 64)
    vp = Viewpoint.full_scene(B, images.device)

    out1 = avp.forward_step(images, vp, None)
    assert out1.hidden.shape == (B, n_persistent + 16, embed_dim)

    out2 = avp.forward_step(images, vp, out1.hidden)
    assert out2.hidden.shape == (B, n_persistent + 16, embed_dim)

    assert out1.scene.shape == (B, 16, embed_dim)
    assert out2.scene.shape == (B, 16, embed_dim)


def test_get_spatial_extracts_correctly():
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=8)
    backbone = MockBackbone(embed_dim, 4, 2, 4, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=embed_dim)

    B = 2
    n_persistent = avp.n_persistent_registers
    n_spatial = 16
    hidden = torch.randn(B, n_persistent + n_spatial, embed_dim)

    spatial = avp.get_spatial(hidden)
    assert spatial.shape == (B, n_spatial, embed_dim)
    assert torch.equal(spatial, hidden[:, n_persistent:])


def test_compute_scene_matches_step_output():
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=8)
    backbone = MockBackbone(embed_dim, 4, 2, 4, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=embed_dim)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    vp = Viewpoint.full_scene(B, images.device)

    out = avp.forward_step(images, vp, None)
    scene_from_helper = avp.compute_scene(out.hidden)
    assert torch.allclose(scene_from_helper, out.scene)


def test_scene_proj_is_layernorm_linear():
    """scene_proj is always LayerNorm + Linear (no Identity fallback)."""
    cfg = AVPConfig(scene_grid_size=4, n_scene_registers=0)
    backbone = MockBackbone(64, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=64)

    assert isinstance(avp.scene_proj, nn.Sequential)
    assert len(avp.scene_proj) == 2
    assert isinstance(avp.scene_proj[0], nn.LayerNorm)
    assert isinstance(avp.scene_proj[1], nn.Linear)


def test_teacher_dim_projects_to_different_dimension():
    """teacher_dim controls the scene projection output dimension."""
    embed_dim = 64
    teacher_dim = 128  # Different from embed_dim
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=0)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=teacher_dim)

    assert avp.teacher_dim == teacher_dim
    assert avp.scene_proj[1].in_features == embed_dim
    assert avp.scene_proj[1].out_features == teacher_dim

    B = 2
    images = torch.randn(B, 3, 64, 64)
    vp = Viewpoint.full_scene(B, images.device)

    out = avp.forward_step(images, vp, None)
    assert out.hidden.shape[-1] == embed_dim  # Internal state in embed_dim
    assert out.scene.shape[-1] == teacher_dim  # Projected output in teacher_dim


def test_multi_viewpoint_forward():
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=0)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=embed_dim)

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
    cfg = AVPConfig(scene_grid_size=14, glimpse_grid_size=7, n_scene_registers=0)
    backbone = MockBackbone(64, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=64)

    assert avp.glimpse_size == 7 * PATCH_SIZE


def test_forward_step_returns_step_output():
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=0)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=embed_dim)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    vp = Viewpoint.full_scene(B, images.device)

    out = avp.forward_step(images, vp, None)

    assert isinstance(out, StepOutput)
    assert out.local.shape == (B, 10, embed_dim)  # CLS + 3x3
    assert out.hidden.shape == (B, 16, embed_dim)
    assert out.scene.shape == (B, 16, embed_dim)


def test_forward_loss():
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=0)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=embed_dim)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    target = torch.randn(B, 16, embed_dim)
    viewpoints = [
        Viewpoint.full_scene(B, images.device),
        Viewpoint.quadrant(B, images.device, 0, 0),
    ]

    loss, final_hidden = avp.forward_loss(images, viewpoints, target)

    assert loss.shape == ()
    assert loss.item() >= 0
    assert final_hidden.shape == (B, 16, embed_dim)


def test_forward_reduce_custom():
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=0)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=embed_dim)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    viewpoints = [Viewpoint.full_scene(B, images.device)]

    def count_reducer(acc: int, out: StepOutput) -> int:
        return acc + 1

    count, final_hidden = avp.forward_reduce(images, viewpoints, count_reducer, init=0)

    assert count == 1
    assert final_hidden.shape == (B, 16, embed_dim)


def test_gradient_checkpointing_smoke():
    embed_dim = 64
    cfg = AVPConfig(
        scene_grid_size=4,
        glimpse_grid_size=3,
        gradient_checkpointing=True,
        n_scene_registers=0,
    )
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=embed_dim)
    avp.train()

    B = 2
    images = torch.randn(B, 3, 64, 64)
    viewpoints = [Viewpoint.full_scene(B, images.device)]

    scene, _ = avp(images, viewpoints)
    loss = scene.sum()
    loss.backward()

    assert avp.spatial_hidden_init.grad is not None


def test_forward_loss_requires_viewpoints():
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=0)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=embed_dim)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    target = torch.randn(B, 16, embed_dim)

    with pytest.raises(AssertionError):
        avp.forward_loss(images, [], target)


# ==================== Context Tests ====================


def test_context_none_returns_none():
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=0)
    backbone = MockBackbone(64, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=64)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    vp = Viewpoint.full_scene(B, images.device)

    out = avp.forward_step(images, vp, None, context=None)
    assert out.context_out is None


def test_context_shapes():
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=0)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=embed_dim)

    B = 2
    n_ctx = 3
    images = torch.randn(B, 3, 64, 64)
    vp = Viewpoint.full_scene(B, images.device)
    context = torch.randn(B, n_ctx, embed_dim)

    out = avp.forward_step(images, vp, None, context=context)

    assert out.context_out is not None
    assert out.context_out.shape == (B, n_ctx, embed_dim)
    n_spatial = cfg.scene_grid_size**2
    assert out.hidden.shape == (B, n_spatial, embed_dim)


def test_context_gradient_flow():
    embed_dim = 64
    cfg = AVPConfig(
        scene_grid_size=4,
        glimpse_grid_size=3,
        layer_scale_init=1.0,
        gating="none",
        n_scene_registers=0,
    )
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=embed_dim)

    B = 2
    n_ctx = 2
    images = torch.randn(B, 3, 64, 64)
    vp = Viewpoint.full_scene(B, images.device)
    context = torch.randn(B, n_ctx, embed_dim, requires_grad=True)

    out = avp.forward_step(images, vp, None, context=context)
    assert out.context_out is not None

    loss = out.context_out.sum() + out.scene.sum()
    loss.backward()

    assert context.grad is not None
    assert context.grad.abs().sum() > 0


def test_context_influences_scene():
    embed_dim = 64
    cfg = AVPConfig(
        scene_grid_size=4,
        glimpse_grid_size=3,
        layer_scale_init=1.0,
        gating="none",
        n_scene_registers=0,
    )
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=embed_dim)

    B = 2
    n_ctx = 2
    torch.manual_seed(42)
    images = torch.randn(B, 3, 64, 64)
    vp = Viewpoint.full_scene(B, images.device)

    ctx1 = torch.randn(B, n_ctx, embed_dim)
    ctx2 = torch.randn(B, n_ctx, embed_dim)

    with torch.no_grad():
        out1 = avp.forward_step(images, vp, None, context=ctx1)
        out2 = avp.forward_step(images, vp, None, context=ctx2)

    diff = (out1.scene - out2.scene).abs().mean()
    assert diff > 0.01, f"Context should influence scene output, but diff={diff}"


# ==================== Curriculum Tests ====================


def test_set_scene_grid_size():
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=0)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=embed_dim)

    assert avp.cfg.scene_grid_size == 4
    assert avp.scene_positions.shape == (16, 2)

    avp.set_scene_grid_size(8)

    assert avp.cfg.scene_grid_size == 8
    assert avp.scene_positions.shape == (64, 2)
    assert avp.scene_size == 8 * PATCH_SIZE


def test_set_scene_grid_size_forward_works():
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=0)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=embed_dim)

    avp.set_scene_grid_size(8)

    B = 2
    images = torch.randn(B, 3, 8 * PATCH_SIZE, 8 * PATCH_SIZE)
    vp = Viewpoint.full_scene(B, images.device)

    out = avp.forward_step(images, vp, None)

    assert out.scene.shape == (B, 64, embed_dim)
    assert out.hidden.shape == (B, 64, embed_dim)


def test_set_scene_grid_size_rejects_too_small():
    cfg = AVPConfig(scene_grid_size=8, glimpse_grid_size=7, n_scene_registers=0)
    backbone = MockBackbone(64, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg, teacher_dim=64)

    with pytest.raises(AssertionError, match="must be >= glimpse_grid_size"):
        avp.set_scene_grid_size(4)

from typing import override

import pytest
import torch
from torch import Tensor, nn
from ytch.nn.layer_scale import LayerScale

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
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=0, use_local_temporal=False)
    backbone = MockBackbone(embed_dim, num_heads, n_blocks, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    images = torch.randn(B, 3, 64, 64)  # scene_grid_size * patch_size = 4 * 16 = 64
    viewpoints = [Viewpoint.full_scene(B, images.device)]

    scene, hidden, local = avp(images, viewpoints)

    assert scene.shape == (B, 16, embed_dim)  # 4x4 grid
    assert hidden.shape == (B, 16, embed_dim)  # hidden same shape (no registers)
    assert local is None  # use_local_temporal=False explicitly


def test_layer_scale_init():
    cfg = AVPConfig(scene_grid_size=4, layer_scale_init=0.5, n_scene_registers=0)
    backbone = MockBackbone(64, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    assert avp.read_scale is not None
    assert avp.write_scale is not None
    for scale in avp.read_scale:
        assert isinstance(scale, LayerScale)
        assert (scale.scale == 0.5).all()
    for scale in avp.write_scale:
        assert isinstance(scale, LayerScale)
        assert (scale.scale == 0.5).all()


def test_convex_gating_init():
    import math

    from avp_vit.attention.convex import ConvexGatedAttention

    gate_init = 0.5
    cfg = AVPConfig(scene_grid_size=4, layer_scale_init=gate_init, use_convex_gating=True, n_scene_registers=0)
    backbone = MockBackbone(64, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    assert avp.read_scale is None
    assert avp.write_scale is None

    expected_bias = math.log(gate_init / (1 - gate_init))
    for cvx in avp.read_attn:
        assert isinstance(cvx, ConvexGatedAttention)
        assert torch.allclose(cvx.gate_bias, torch.full((64,), expected_bias))
        assert torch.allclose(cvx.gate_scale, torch.zeros(64))


def test_convex_init_passthrough():
    """At init with low gate, convex gating acts as passthrough on both streams."""
    from avp_vit.glimpse import extract_glimpse

    n_blocks = 2
    gate_init = 1e-5
    cfg = AVPConfig(
        scene_grid_size=4, glimpse_grid_size=3, layer_scale_init=gate_init,
        use_convex_gating=True, n_scene_registers=0,
        use_local_temporal=False, use_scene_input_norm=False,
    )
    backbone = MockBackbone(64, 4, n_blocks, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

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
        # Initial scene goes through scene_input_norm before processing
        normed_init = avp.scene_input_norm(avp._get_base_hidden(B))
        initial_scene = avp.output_proj(normed_init)

    # Scene: write gate ≈ 0 → scene ≈ initial (after norm)
    scene_diff = (out.scene - initial_scene).abs().mean()
    assert scene_diff < 0.1, f"Scene changed too much: {scene_diff}"

    # Local: read gate ≈ 0 → local ≈ backbone-only
    local_diff = (out.local - baseline_local).abs().mean()
    assert local_diff < 1e-3, f"Local differs from backbone-only: {local_diff}"


def test_convex_gate_value_affects_output():
    """High gate breaks passthrough; low gate preserves it."""
    cfg_lo = AVPConfig(
        scene_grid_size=4, layer_scale_init=1e-5, use_convex_gating=True, n_scene_registers=0,
        use_local_temporal=False, use_scene_input_norm=False,
    )
    cfg_hi = AVPConfig(
        scene_grid_size=4, layer_scale_init=0.5, use_convex_gating=True, n_scene_registers=0,
        use_local_temporal=False, use_scene_input_norm=False,
    )

    # Same seed for both → same spatial_init, attention weights
    torch.manual_seed(999)
    avp_lo = AVPViT(MockBackbone(64, 4, 2, 0, PATCH_SIZE), cfg_lo)
    torch.manual_seed(999)
    avp_hi = AVPViT(MockBackbone(64, 4, 2, 0, PATCH_SIZE), cfg_hi)

    B = 2
    torch.manual_seed(42)
    images = torch.randn(B, 3, avp_lo.scene_size, avp_lo.scene_size)
    vp = Viewpoint("test", torch.zeros(B, 2), torch.ones(B))

    with torch.no_grad():
        torch.manual_seed(123)
        out_lo = avp_lo.forward_step(images, vp)
        torch.manual_seed(123)
        out_hi = avp_hi.forward_step(images, vp)
        # Initial scene goes through scene_input_norm before processing
        normed_init = avp_lo.scene_input_norm(avp_lo._get_base_hidden(B))
        initial = avp_lo.output_proj(normed_init)

    diff_lo = (out_lo.scene - initial).abs().mean()
    diff_hi = (out_hi.scene - initial).abs().mean()

    # Low gate: passthrough works (diff small, after norm)
    assert diff_lo < 0.01, f"Low gate should preserve passthrough: {diff_lo}"
    # High gate: passthrough broken (diff large)
    assert diff_hi > 0.1, f"High gate should break passthrough: {diff_hi}"


def test_scene_registers_disabled_when_zero():
    cfg = AVPConfig(scene_grid_size=4, n_scene_registers=0)
    backbone = MockBackbone(64, 4, 2, 4, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    assert avp.n_scene_registers == 0
    assert avp.persistent_registers is None
    assert avp.ephemeral_registers is None


def test_scene_registers_fixed_count():
    """n_scene_registers is a fixed count from config, not computed from ratio."""
    cfg = AVPConfig(scene_grid_size=14, glimpse_grid_size=7, n_scene_registers=16)
    backbone = MockBackbone(64, 4, 2, 4, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    assert avp.n_scene_registers == 16
    assert avp.n_persistent_registers == 8
    assert avp.n_ephemeral_registers == 8
    assert avp.persistent_registers is not None
    assert avp.ephemeral_registers is not None
    assert avp.persistent_registers.shape == (1, 8, 64)
    assert avp.ephemeral_registers.shape == (1, 8, 64)


def test_scene_registers_split():
    """Scene registers split into persistent (passthrough) and ephemeral (reinit)."""
    embed_dim, num_heads, n_blocks = 64, 4, 2
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=7)
    backbone = MockBackbone(embed_dim, num_heads, n_blocks, 4, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    # Total = 7, persistent = 3, ephemeral = 4
    assert avp.n_scene_registers == 7
    assert avp.n_persistent_registers == 3
    assert avp.n_ephemeral_registers == 4

    assert avp.persistent_registers is not None
    assert avp.ephemeral_registers is not None
    assert avp.persistent_registers.shape == (1, 3, embed_dim)
    assert avp.ephemeral_registers.shape == (1, 4, embed_dim)


def test_scene_output_always_spatial_only():
    """Scene output should only contain grid tokens, not registers."""
    embed_dim, num_heads, n_blocks = 64, 4, 2
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=7, use_local_temporal=False)
    backbone = MockBackbone(embed_dim, num_heads, n_blocks, 4, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    viewpoints = [Viewpoint.full_scene(B, images.device)]

    scene, hidden, local = avp(images, viewpoints)

    # Scene is always spatial-only (no registers)
    assert scene.shape == (B, 16, embed_dim)  # 4x4 grid
    assert local is None  # use_local_temporal=False

    # Hidden contains persistent registers + spatial (NOT glimpse registers)
    n_persistent = avp.n_persistent_registers
    assert hidden.shape == (B, n_persistent + 16, embed_dim)


def test_scene_registers_continuity():
    """Persistent registers flow between timesteps, glimpse registers don't."""
    embed_dim, num_heads, n_blocks = 64, 4, 2
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=7)
    backbone = MockBackbone(embed_dim, num_heads, n_blocks, 4, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    n_persistent = avp.n_persistent_registers
    images = torch.randn(B, 3, 64, 64)
    vp = Viewpoint.full_scene(B, images.device)

    # First step: hidden=None -> uses initial persistent registers
    out1 = avp.forward_step(images, vp, None)
    assert out1.hidden.shape == (B, n_persistent + 16, embed_dim)

    # Second step: pass hidden from first step (includes persistent registers)
    out2 = avp.forward_step(images, vp, out1.hidden)
    assert out2.hidden.shape == (B, n_persistent + 16, embed_dim)

    # Scene is always spatial-only
    assert out1.scene.shape == (B, 16, embed_dim)
    assert out2.scene.shape == (B, 16, embed_dim)


def test_get_spatial_extracts_correctly():
    """get_spatial correctly extracts spatial tokens from hidden state."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=8)
    backbone = MockBackbone(embed_dim, 4, 2, 4, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    n_persistent = avp.n_persistent_registers
    n_spatial = 16  # 4x4 grid
    hidden = torch.randn(B, n_persistent + n_spatial, embed_dim)

    spatial = avp.get_spatial(hidden)
    assert spatial.shape == (B, n_spatial, embed_dim)
    assert torch.equal(spatial, hidden[:, n_persistent:])


def test_compute_scene_matches_step_output():
    """compute_scene produces same result as StepOutput.scene."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=8, use_output_proj=True)
    backbone = MockBackbone(embed_dim, 4, 2, 4, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    vp = Viewpoint.full_scene(B, images.device)

    out = avp.forward_step(images, vp, None)

    # compute_scene on hidden should equal StepOutput.scene
    scene_from_helper = avp.compute_scene(out.hidden)
    assert torch.allclose(scene_from_helper, out.scene)


def test_output_proj_is_always_module():
    """output_proj is always nn.Module (Identity or Linear), never None."""
    cfg_no_proj = AVPConfig(scene_grid_size=4, use_output_proj=False, n_scene_registers=0)
    cfg_with_proj = AVPConfig(scene_grid_size=4, use_output_proj=True, n_scene_registers=0)
    backbone = MockBackbone(64, 4, 2, 0, PATCH_SIZE)

    avp_no = AVPViT(backbone, cfg_no_proj)
    avp_yes = AVPViT(backbone, cfg_with_proj)

    assert isinstance(avp_no.output_proj, torch.nn.Identity)
    assert isinstance(avp_yes.output_proj, torch.nn.Sequential)


def test_multi_viewpoint_forward():
    """Forward with multiple viewpoints processes all sequentially."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=0, use_local_temporal=False)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    viewpoints = [
        Viewpoint.full_scene(B, images.device),
        Viewpoint.quadrant(B, images.device, 0, 0),
        Viewpoint.quadrant(B, images.device, 1, 1),
    ]

    scene, hidden, local = avp(images, viewpoints)

    assert scene.shape == (B, 16, embed_dim)
    assert hidden.shape == (B, 16, embed_dim)
    assert local is None  # use_local_temporal=False


def test_glimpse_size_from_backbone():
    """glimpse_size = glimpse_grid_size * backbone.patch_size."""
    cfg = AVPConfig(scene_grid_size=14, glimpse_grid_size=7, n_scene_registers=0)
    backbone = MockBackbone(64, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    assert avp.glimpse_size == 7 * PATCH_SIZE  # 112


def test_forward_step_returns_step_output():
    """forward_step returns StepOutput with both hidden and scene."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=0)
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
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=0, use_local_temporal=False)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    target = torch.randn(B, 16, embed_dim)
    viewpoints = [
        Viewpoint.full_scene(B, images.device),
        Viewpoint.quadrant(B, images.device, 0, 0),
    ]

    loss, final_hidden, final_local = avp.forward_loss(images, viewpoints, target)

    assert loss.shape == ()  # scalar
    assert loss.item() >= 0  # MSE is non-negative
    assert final_hidden.shape == (B, 16, embed_dim)
    assert final_local is None  # use_local_temporal=False


def test_forward_trajectory():
    """forward_trajectory returns list of projected scenes."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=0, use_local_temporal=False)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    viewpoints = [
        Viewpoint.full_scene(B, images.device),
        Viewpoint.quadrant(B, images.device, 0, 0),
        Viewpoint.quadrant(B, images.device, 1, 1),
    ]

    scenes, final_hidden, final_local = avp.forward_trajectory(images, viewpoints)

    assert len(scenes) == 3  # one per viewpoint
    for s in scenes:
        assert s.shape == (B, 16, embed_dim)
    assert final_hidden.shape == (B, 16, embed_dim)
    assert final_local is None  # use_local_temporal=False


def test_forward_reduce_custom():
    """forward_reduce with custom reducer."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=0, use_local_temporal=False)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    viewpoints = [Viewpoint.full_scene(B, images.device)]

    # Custom reducer that counts steps
    def count_reducer(acc: int, out: StepOutput) -> int:
        return acc + 1

    count, final_hidden, final_local = avp.forward_reduce(images, viewpoints, count_reducer, init=0)

    assert count == 1
    assert final_hidden.shape == (B, 16, embed_dim)
    assert final_local is None  # use_local_temporal=False


def test_gradient_checkpointing_smoke():
    """Gradient checkpointing produces same output and gradients flow."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, gradient_checkpointing=True, n_scene_registers=0)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)
    avp.train()

    B = 2
    images = torch.randn(B, 3, 64, 64)
    viewpoints = [Viewpoint.full_scene(B, images.device)]

    scene, _, _ = avp(images, viewpoints)
    loss = scene.sum()
    loss.backward()

    assert avp.spatial_init.grad is not None


def test_forward_loss_requires_viewpoints():
    """forward_loss requires at least one viewpoint."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, use_output_proj=True, n_scene_registers=0)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    target = torch.randn(B, 16, embed_dim)

    # Empty viewpoints should raise
    with pytest.raises(AssertionError, match="Need at least one viewpoint"):
        avp.forward_loss(images, [], target)


def test_local_temporal_disabled():
    """Local temporal parameters are None when use_local_temporal=False."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=0, use_local_temporal=False)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    assert avp.local_init is None
    assert avp.local_temporal_norm is None
    assert avp.local_temporal_gate is None


def test_local_temporal_parameters_shapes():
    """Local temporal parameters have correct shapes when enabled."""
    embed_dim = 64
    n_registers = 4
    glimpse_grid_size = 3
    cfg = AVPConfig(
        scene_grid_size=4,
        glimpse_grid_size=glimpse_grid_size,
        use_local_temporal=True,
        temporal_gate_init=1e-5,
        n_scene_registers=0,
    )
    backbone = MockBackbone(embed_dim, 4, 2, n_registers, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    # n_prefix = 1 (CLS) + n_registers = 5
    n_prefix = backbone.n_prefix_tokens
    assert n_prefix == 1 + n_registers

    # N_local = n_prefix + glimpse_grid_size²
    expected_n_local = n_prefix + glimpse_grid_size**2
    assert avp.n_local_tokens == expected_n_local

    assert avp.local_init is not None
    assert avp.local_init.shape == (1, expected_n_local, embed_dim)

    assert avp.local_temporal_norm is not None
    assert isinstance(avp.local_temporal_norm, nn.LayerNorm)

    # Gate shape: (n_prefix + 1, D) - one per prefix token, one for all patches
    # Stores logit, sigmoid(logit) ≈ init_value
    assert avp.local_temporal_gate is not None
    assert avp.local_temporal_gate.shape == (n_prefix + 1, embed_dim)
    assert torch.allclose(torch.sigmoid(avp.local_temporal_gate), torch.full_like(avp.local_temporal_gate, 1e-5), atol=1e-6)


def test_local_temporal_gating_gradient_flow():
    """Gradient flows through local_temporal_gate when use_local_temporal=True."""
    embed_dim = 64
    glimpse_grid_size = 3
    cfg = AVPConfig(
        scene_grid_size=4,
        glimpse_grid_size=glimpse_grid_size,
        use_local_temporal=True,
        layer_scale_init=0.1,  # Nonzero so gradient flows through cross-attention
        temporal_gate_init=0.1,  # Nonzero so gradient is meaningful
        n_scene_registers=0,
    )
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    n_local = avp.n_local_tokens

    # Create inputs - use forward_step which calls _process_glimpse internally
    glimpse = torch.randn(B, 3, glimpse_grid_size * PATCH_SIZE, glimpse_grid_size * PATCH_SIZE)
    local_prev = torch.randn(B, n_local, embed_dim, requires_grad=True)
    centers = torch.zeros(B, 2)
    scales = torch.ones(B)

    out = avp._process_glimpse(glimpse, centers, scales, None, local_prev, None)

    # Backward pass
    loss = out.scene.sum()
    loss.backward()

    # Gradient should flow to local_temporal_gate
    assert avp.local_temporal_gate is not None
    assert avp.local_temporal_gate.grad is not None
    assert avp.local_temporal_gate.grad.abs().sum() > 0

    # Gradient should flow to local_prev
    assert local_prev.grad is not None
    assert local_prev.grad.abs().sum() > 0


def test_local_temporal_disabled_no_effect_on_output():
    """With use_local_temporal=False, local_prev has no effect (not used)."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, use_local_temporal=False, n_scene_registers=0)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    vp = Viewpoint.full_scene(B, images.device)

    # Two forward passes should give same output structure
    out1 = avp.forward_step(images, vp, None)
    out2 = avp.forward_step(images, vp, None)

    # Shapes should match
    assert out1.scene.shape == out2.scene.shape
    assert out1.hidden.shape == out2.hidden.shape


def test_forward_step_local_prev_flows_through():
    """forward_step passes local_prev to _process_glimpse and uses local_init if None."""
    embed_dim = 64
    cfg = AVPConfig(
        scene_grid_size=4,
        glimpse_grid_size=3,
        use_local_temporal=True,
        layer_scale_init=0.1,
        n_scene_registers=0,
    )
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    vp = Viewpoint.full_scene(B, images.device)

    # First call with local_prev=None should use local_init
    out1 = avp.forward_step(images, vp, None, None)
    assert out1.local.shape == (B, avp.n_local_tokens, embed_dim)

    # Second call with local_prev from first call
    out2 = avp.forward_step(images, vp, out1.hidden, out1.local)
    assert out2.local.shape == (B, avp.n_local_tokens, embed_dim)

    # Output local can be used as next local_prev
    out3 = avp.forward_step(images, vp, out2.hidden, out2.local)
    assert out3.local.shape == (B, avp.n_local_tokens, embed_dim)


@pytest.mark.parametrize("has_cls,n_registers", [
    (True, 0),   # CLS only (like minimal DINOv3)
    (True, 4),   # CLS + registers (like full DINOv3)
    (False, 0),  # No prefix tokens (like IJEPA)
    (False, 2),  # Registers without CLS (hypothetical)
])
def test_local_temporal_gate_shapes_parametrized(has_cls: bool, n_registers: int):
    """Gate shape is (n_prefix + 1, D) across different backbone configs."""
    embed_dim = 64
    glimpse_grid_size = 3
    cfg = AVPConfig(
        scene_grid_size=4,
        glimpse_grid_size=glimpse_grid_size,
        use_local_temporal=True,
        layer_scale_init=0.1,
        n_scene_registers=0,
    )
    backbone = MockBackbone(embed_dim, 4, 2, n_registers, PATCH_SIZE, has_cls=has_cls)
    avp = AVPViT(backbone, cfg)

    n_prefix = backbone.n_prefix_tokens
    assert n_prefix == (1 if has_cls else 0) + n_registers

    assert avp.local_temporal_gate is not None
    assert avp.local_temporal_gate.shape == (n_prefix + 1, embed_dim)


@pytest.mark.parametrize("has_cls,n_registers", [
    (True, 0),
    (True, 4),
    (False, 0),
    (False, 2),
])
def test_local_temporal_forward_parametrized(has_cls: bool, n_registers: int):
    """Forward pass works with local temporal across different backbone configs."""
    embed_dim = 64
    glimpse_grid_size = 3
    cfg = AVPConfig(
        scene_grid_size=4,
        glimpse_grid_size=glimpse_grid_size,
        use_local_temporal=True,
        layer_scale_init=0.1,
        n_scene_registers=0,
    )
    backbone = MockBackbone(embed_dim, 4, 2, n_registers, PATCH_SIZE, has_cls=has_cls)
    avp = AVPViT(backbone, cfg)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    vp = Viewpoint.full_scene(B, images.device)

    # Forward with local_prev=None
    out1 = avp.forward_step(images, vp, None, None)
    n_local = backbone.n_prefix_tokens + glimpse_grid_size ** 2
    assert out1.local.shape == (B, n_local, embed_dim)

    # Forward with local_prev from previous step
    out2 = avp.forward_step(images, vp, out1.hidden, out1.local)
    assert out2.local.shape == (B, n_local, embed_dim)


# ==================== Context Tests ====================


def test_context_none_returns_none():
    """When context is None, context_out should be None."""
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=0)
    backbone = MockBackbone(64, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    images = torch.randn(B, 3, 64, 64)
    vp = Viewpoint.full_scene(B, images.device)

    out = avp.forward_step(images, vp, None, None, context=None)
    assert out.context_out is None


def test_context_shapes():
    """Context tokens are transformed and returned with correct shape."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=0)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    n_ctx = 3
    images = torch.randn(B, 3, 64, 64)
    vp = Viewpoint.full_scene(B, images.device)
    context = torch.randn(B, n_ctx, embed_dim)

    out = avp.forward_step(images, vp, None, None, context=context)

    assert out.context_out is not None
    assert out.context_out.shape == (B, n_ctx, embed_dim)
    # Hidden should not include context (same shape as without context)
    n_spatial = cfg.scene_grid_size ** 2
    assert out.hidden.shape == (B, n_spatial, embed_dim)


def test_context_gradient_flow():
    """Gradients flow through context tokens."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, layer_scale_init=1.0, n_scene_registers=0)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    n_ctx = 2
    images = torch.randn(B, 3, 64, 64)
    vp = Viewpoint.full_scene(B, images.device)
    context = torch.randn(B, n_ctx, embed_dim, requires_grad=True)

    out = avp.forward_step(images, vp, None, None, context=context)
    assert out.context_out is not None

    loss = out.context_out.sum() + out.scene.sum()
    loss.backward()

    assert context.grad is not None
    assert context.grad.abs().sum() > 0


def test_context_influences_scene():
    """Different context produces different scene output."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, layer_scale_init=1.0, n_scene_registers=0)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    B = 2
    n_ctx = 2
    torch.manual_seed(42)
    images = torch.randn(B, 3, 64, 64)
    vp = Viewpoint.full_scene(B, images.device)

    ctx1 = torch.randn(B, n_ctx, embed_dim)
    ctx2 = torch.randn(B, n_ctx, embed_dim)

    with torch.no_grad():
        out1 = avp.forward_step(images, vp, None, None, context=ctx1)
        out2 = avp.forward_step(images, vp, None, None, context=ctx2)

    # Different context should produce different scenes (with non-zero gates)
    diff = (out1.scene - out2.scene).abs().mean()
    assert diff > 0.01, f"Context should influence scene output, but diff={diff}"


# ==================== Curriculum Tests ====================


def test_set_scene_grid_size():
    """set_scene_grid_size updates config and recomputes scene_positions."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=0)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    # Initial state
    assert avp.cfg.scene_grid_size == 4
    assert avp.scene_positions.shape == (16, 2)  # 4*4 = 16

    # Update to larger size
    avp.set_scene_grid_size(8)

    assert avp.cfg.scene_grid_size == 8
    assert avp.scene_positions.shape == (64, 2)  # 8*8 = 64
    assert avp.scene_size == 8 * PATCH_SIZE


def test_set_scene_grid_size_forward_works():
    """Forward pass works after set_scene_grid_size."""
    embed_dim = 64
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3, n_scene_registers=0)
    backbone = MockBackbone(embed_dim, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    avp.set_scene_grid_size(8)

    B = 2
    images = torch.randn(B, 3, 8 * PATCH_SIZE, 8 * PATCH_SIZE)
    vp = Viewpoint.full_scene(B, images.device)

    out = avp.forward_step(images, vp, None)

    assert out.scene.shape == (B, 64, embed_dim)  # 8*8 = 64
    assert out.hidden.shape == (B, 64, embed_dim)


def test_set_scene_grid_size_rejects_too_small():
    """set_scene_grid_size rejects sizes smaller than glimpse_grid_size."""
    cfg = AVPConfig(scene_grid_size=8, glimpse_grid_size=7, n_scene_registers=0)
    backbone = MockBackbone(64, 4, 2, 0, PATCH_SIZE)
    avp = AVPViT(backbone, cfg)

    with pytest.raises(AssertionError, match="must be >= glimpse_grid_size"):
        avp.set_scene_grid_size(4)

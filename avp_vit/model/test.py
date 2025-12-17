from typing import override

import torch
from torch import Tensor, nn

from avp_vit.backbone import ViTBackbone
from avp_vit.glimpse import Viewpoint
from avp_vit.model import AVPConfig, AVPViT, LossOutputs, StepOutput
from avp_vit.rope import make_rope_periods


class MockBackbone(ViTBackbone, nn.Module):
    """Minimal backbone for unit testing AVPViT."""

    _rope_periods: Tensor

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        n_blocks: int = 2,
        patch_size: int = 16,
    ) -> None:
        nn.Module.__init__(self)
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._n_blocks = n_blocks
        self._patch_size = patch_size
        self.register_buffer(
            "_rope_periods",
            make_rope_periods(embed_dim // num_heads, dtype=torch.float32),
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
        return 1  # CLS only

    @property
    @override
    def n_register_tokens(self) -> int:
        return 0

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
        B, _, H, W = images.shape
        h, w = H // self._patch_size, W // self._patch_size
        tokens = torch.randn(B, self.n_prefix_tokens + h * w, self._embed_dim, device=images.device)
        return tokens, h, w


def test_forward_shapes():
    cfg = AVPConfig(glimpse_grid_size=3, n_scene_registers=0)
    avp = AVPViT(MockBackbone(), cfg, teacher_dim=64)

    B, G = 2, 4
    images = torch.randn(B, 3, 64, 64)
    hidden = avp.init_hidden(B, G)

    scene, final_hidden = avp(images, [Viewpoint.full_scene(B, images.device)], hidden)

    assert scene.shape == (B, G * G, 64)
    # hidden = [cls | registers | spatial], here n_cls=1, n_reg=0
    assert final_hidden.shape == (B, avp.n_cls + G * G, 64)


def test_registers_in_hidden():
    n_reg = 8
    cfg = AVPConfig(glimpse_grid_size=3, n_scene_registers=n_reg)
    avp = AVPViT(MockBackbone(), cfg, teacher_dim=64)

    assert avp.scene_registers is not None
    assert avp.scene_registers.shape == (1, n_reg, 64)

    B, G = 2, 4
    hidden = avp.init_hidden(B, G)
    # hidden = [cls | registers | spatial]
    assert hidden.shape == (B, avp.n_cls + n_reg + G * G, 64)

    images = torch.randn(B, 3, 64, 64)
    out = avp.forward_step(images, Viewpoint.full_scene(B, images.device), hidden)
    assert out.hidden.shape == (B, avp.n_cls + n_reg + G * G, 64)
    assert out.scene.shape == (B, G * G, 64)  # Scene excludes cls and registers


def test_registers_disabled_when_zero():
    cfg = AVPConfig(n_scene_registers=0)
    avp = AVPViT(MockBackbone(), cfg, teacher_dim=64)
    assert avp.scene_registers is None


def test_recurrence_ln_weight_init():
    """LayerNorm weights should be 1/sqrt(D) when use_recurrence_ln=True."""
    import math
    cfg = AVPConfig(n_scene_registers=8, use_recurrence_ln=True)
    avp = AVPViT(MockBackbone(), cfg, teacher_dim=64)
    expected = 1.0 / math.sqrt(64)
    assert isinstance(avp.cls_ln, torch.nn.LayerNorm)
    assert isinstance(avp.reg_ln, torch.nn.LayerNorm)
    assert isinstance(avp.spatial_ln, torch.nn.LayerNorm)
    assert torch.allclose(avp.cls_ln.weight, torch.full((64,), expected))
    assert torch.allclose(avp.reg_ln.weight, torch.full((64,), expected))
    assert torch.allclose(avp.spatial_ln.weight, torch.full((64,), expected))


def test_recurrence_ln_disabled():
    """With use_recurrence_ln=False, modules should be Identity."""
    cfg = AVPConfig(n_scene_registers=8, use_recurrence_ln=False)
    avp = AVPViT(MockBackbone(), cfg, teacher_dim=64)
    assert isinstance(avp.cls_ln, torch.nn.Identity)
    assert isinstance(avp.reg_ln, torch.nn.Identity)
    assert isinstance(avp.spatial_ln, torch.nn.Identity)


def test_get_spatial():
    n_reg = 8
    cfg = AVPConfig(n_scene_registers=n_reg)
    avp = AVPViT(MockBackbone(), cfg, teacher_dim=64)

    B, G = 2, 4
    # hidden = [cls | registers | spatial]
    hidden = torch.randn(B, avp.n_cls + n_reg + G * G, 64)
    spatial = avp.get_spatial(hidden)
    assert spatial.shape == (B, G * G, 64)
    assert torch.equal(spatial, hidden[:, avp.n_prefix:])


def test_context_flow():
    cfg = AVPConfig(glimpse_grid_size=3, n_scene_registers=0, layer_scale_init=1.0)
    avp = AVPViT(MockBackbone(), cfg, teacher_dim=64)

    B, G, n_ctx = 2, 4, 3
    images = torch.randn(B, 3, 64, 64)
    context = torch.randn(B, n_ctx, 64, requires_grad=True)
    hidden = avp.init_hidden(B, G)

    out = avp.forward_step(images, Viewpoint.full_scene(B, images.device), hidden, context=context)

    assert out.context_out is not None
    assert out.context_out.shape == (B, n_ctx, 64)

    loss = out.context_out.sum() + out.scene.sum()
    loss.backward()
    assert context.grad is not None


def test_different_grid_sizes():
    cfg = AVPConfig(glimpse_grid_size=3, n_scene_registers=4)
    avp = AVPViT(MockBackbone(), cfg, teacher_dim=64)
    B = 2

    for G in [4, 8]:
        hidden = avp.init_hidden(B, G)
        images = torch.randn(B, 3, G * 16, G * 16)
        out = avp.forward_step(images, Viewpoint.full_scene(B, images.device), hidden)
        assert out.scene.shape == (B, G * G, 64)
        # hidden = [cls | registers | spatial]
        assert out.hidden.shape == (B, avp.n_cls + 4 + G * G, 64)


def test_forward_loss():
    cfg = AVPConfig(glimpse_grid_size=3, n_scene_registers=0)
    avp = AVPViT(MockBackbone(), cfg, teacher_dim=64)

    B, G, D = 2, 4, 64
    images = torch.randn(B, 3, 64, 64)
    target = torch.randn(B, G * G, D)
    cls_target = torch.randn(B, D)
    hidden = avp.init_hidden(B, G)
    viewpoints = [Viewpoint.full_scene(B, images.device), Viewpoint.quadrant(B, images.device, 0, 0)]

    losses, final_hidden = avp.forward_loss(images, viewpoints, target, hidden, cls_target=cls_target)

    assert isinstance(losses, LossOutputs)
    assert losses.scene.shape == ()
    assert losses.scene.item() >= 0
    # use_cls_loss=True by default, so cls should be present
    assert losses.cls is not None
    assert losses.cls.shape == ()
    assert losses.cls.item() >= 0
    # hidden = [cls | registers | spatial]
    assert final_hidden.shape == (B, avp.n_cls + G * G, D)


def test_step_output_type():
    cfg = AVPConfig(glimpse_grid_size=3, n_scene_registers=0)
    avp = AVPViT(MockBackbone(), cfg, teacher_dim=64)

    B, G = 2, 4
    images = torch.randn(B, 3, 64, 64)
    hidden = avp.init_hidden(B, G)

    out = avp.forward_step(images, Viewpoint.full_scene(B, images.device), hidden)
    assert isinstance(out, StepOutput)

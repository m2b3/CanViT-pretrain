"""Tests for CanViT."""

import torch
from torch import Tensor, nn

from canvit import CanViT, CanViTConfig
from canvit.backbone import ViTBackbone
from canvit.rope import make_rope_periods


class MockBackbone(ViTBackbone, nn.Module):
    """Minimal backbone for testing."""

    def __init__(self, dim: int = 64, heads: int = 8, blocks: int = 6, patch_px: int = 16):
        nn.Module.__init__(self)
        self._dim = dim
        self._heads = heads
        self._blocks = blocks
        self._patch_px = patch_px
        self._periods = make_rope_periods(dim // heads, torch.float32)
        self.block_modules = nn.ModuleList([nn.Identity() for _ in range(blocks)])
        self.patch_embed = nn.Linear(3 * patch_px * patch_px, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    @property
    def embed_dim(self) -> int:
        return self._dim

    @property
    def num_heads(self) -> int:
        return self._heads

    @property
    def n_prefix_tokens(self) -> int:
        return 1

    @property
    def n_blocks(self) -> int:
        return self._blocks

    @property
    def patch_size_px(self) -> int:
        return self._patch_px

    @property
    def rope_periods(self) -> Tensor:
        return self._periods

    @property
    def rope_dtype(self) -> torch.dtype:
        return torch.float32

    def forward_block(self, idx: int, x: Tensor, rope: tuple[Tensor, Tensor] | None) -> Tensor:
        return self.block_modules[idx](x)

    def prepare_tokens(self, images: Tensor) -> tuple[Tensor, int, int]:
        B, C, H, W = images.shape
        P = self._patch_px
        gh, gw = H // P, W // P
        patches = images.unfold(2, P, P).unfold(3, P, P)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, gh * gw, C * P * P)
        patch_tokens = self.patch_embed(patches)
        cls = self.cls_token.expand(B, -1, -1)
        return torch.cat([cls, patch_tokens], dim=1), gh, gw


def test_canvit_init_canvas():
    backbone = MockBackbone(dim=64)
    cfg = CanViTConfig(n_canvas_registers=8, canvas_num_heads=8)
    model = CanViT(backbone, cfg)

    B, G = 2, 4
    canvas = model.init_canvas(batch_size=B, canvas_grid_size=G)
    assert canvas.shape == (B, model.n_prefix + G * G, model.canvas_dim)


def test_canvit_forward():
    backbone = MockBackbone(dim=64, heads=8, blocks=6, patch_px=16)
    cfg = CanViTConfig(n_canvas_registers=8, adapter_stride=2, layer_scale_init=1e-3, canvas_num_heads=8)
    model = CanViT(backbone, cfg)

    B, D = 2, 64
    canvas_grid = 4
    glimpse_grid = 4
    glimpse_size = glimpse_grid * 16

    glimpse = torch.randn(B, 3, glimpse_size, glimpse_size)
    canvas = model.init_canvas(B, canvas_grid)

    local_pos = torch.randn(B, glimpse_grid ** 2, 2)
    canvas_pos = torch.randn(B, canvas_grid ** 2, 2)

    local_out, canvas_out = model(glimpse, canvas, local_pos, canvas_pos)
    assert local_out.shape[0] == B
    assert local_out.shape[2] == D
    assert canvas_out.shape == canvas.shape


def test_canvit_n_adapters():
    backbone = MockBackbone(blocks=12)
    cfg = CanViTConfig(adapter_stride=2, canvas_num_heads=8)
    model = CanViT(backbone, cfg)
    assert model.n_adapters == 5  # (12-1) // 2 = 5

    cfg3 = CanViTConfig(adapter_stride=3, canvas_num_heads=8)
    model3 = CanViT(backbone, cfg3)
    assert model3.n_adapters == 3  # (12-1) // 3 = 3


def test_canvit_gradients_flow():
    backbone = MockBackbone(dim=64, heads=8, blocks=4, patch_px=16)
    cfg = CanViTConfig(n_canvas_registers=4, adapter_stride=2, canvas_num_heads=8)
    model = CanViT(backbone, cfg)

    B = 2
    canvas_grid = 4
    glimpse_grid = 4
    glimpse_size = glimpse_grid * 16

    glimpse = torch.randn(B, 3, glimpse_size, glimpse_size, requires_grad=True)
    canvas = model.init_canvas(B, canvas_grid).detach().requires_grad_(True)

    local_pos = torch.randn(B, glimpse_grid ** 2, 2)
    canvas_pos = torch.randn(B, canvas_grid ** 2, 2)

    local_out, canvas_out = model(glimpse, canvas, local_pos, canvas_pos)
    loss = local_out.sum() + canvas_out.sum()
    loss.backward()

    assert glimpse.grad is not None
    assert canvas.grad is not None
    assert glimpse.grad.abs().sum() > 0
    assert canvas.grad.abs().sum() > 0

"""Unified AVP tests against all backbones."""

import pytest
import torch
from dinov3.models.vision_transformer import vit_small as dinov3_vit_small
from ijepa.models.vision_transformer import vit_small as ijepa_vit_small

from avp_vit import AVPConfig, AVPViT
from avp_vit.backbone import ViTBackbone
from avp_vit.backbone.dinov3 import DINOv3Backbone
from avp_vit.backbone.ijepa import IJEPABackbone
from avp_vit.glimpse import Viewpoint


@pytest.fixture(params=["dinov3", "ijepa"])
def backbone(request: pytest.FixtureRequest) -> ViTBackbone:
    torch.manual_seed(42)
    if request.param == "dinov3":
        native = dinov3_vit_small(img_size=112, patch_size=16)
        native.init_weights()
        return DINOv3Backbone(native)
    else:
        native = ijepa_vit_small(img_size=[112], patch_size=16)
        return IJEPABackbone(native, rope_dtype=torch.float32)


def test_avp_forward_shapes(backbone: ViTBackbone) -> None:
    """AVP forward produces correct output shapes."""
    cfg = AVPConfig(scene_grid_size=8, glimpse_grid_size=7, n_scene_registers=0)
    avp = AVPViT(backbone, cfg, teacher_dim=backbone.embed_dim)

    B = 2
    images = torch.randn(B, 3, 128, 128)
    viewpoints = [Viewpoint.full_scene(B, images.device)]

    scene, hidden = avp(images, viewpoints)

    assert scene.shape == (B, 64, backbone.embed_dim)
    assert hidden.shape == (B, 64, backbone.embed_dim)


def test_hidden_unchanged_at_init(backbone: ViTBackbone) -> None:
    """With γ=0 and no temporal gating, hidden should equal normalized spatial_init."""
    cfg = AVPConfig(
        scene_grid_size=8,
        glimpse_grid_size=7,
        layer_scale_init=0.0,
        n_scene_registers=0,
    )
    avp = AVPViT(backbone, cfg, teacher_dim=backbone.embed_dim)

    B = 2
    images = torch.randn(B, 3, 128, 128)
    viewpoints = [Viewpoint.full_scene(B, images.device)]

    _, hidden = avp(images, viewpoints)

    expected = avp._get_base_hidden(B)
    assert torch.allclose(hidden, expected, atol=1e-5)


def test_multi_viewpoint_forward(backbone: ViTBackbone) -> None:
    """Forward with multiple viewpoints processes all."""
    cfg = AVPConfig(scene_grid_size=8, glimpse_grid_size=7, n_scene_registers=0)
    avp = AVPViT(backbone, cfg, teacher_dim=backbone.embed_dim)

    B = 2
    images = torch.randn(B, 3, 128, 128)
    viewpoints = [
        Viewpoint.full_scene(B, images.device),
        Viewpoint.quadrant(B, images.device, 0, 0),
        Viewpoint.quadrant(B, images.device, 1, 1),
    ]

    scene, hidden = avp(images, viewpoints)

    assert scene.shape == (B, 64, backbone.embed_dim)
    assert hidden.shape == (B, 64, backbone.embed_dim)


def test_forward_loss(backbone: ViTBackbone) -> None:
    """forward_loss computes averaged MSE correctly."""
    cfg = AVPConfig(scene_grid_size=8, glimpse_grid_size=7, n_scene_registers=0)
    avp = AVPViT(backbone, cfg, teacher_dim=backbone.embed_dim)

    B = 2
    images = torch.randn(B, 3, 128, 128)
    viewpoints = [
        Viewpoint.full_scene(B, images.device),
        Viewpoint.quadrant(B, images.device, 0, 0),
    ]
    target = torch.randn(B, 64, backbone.embed_dim)

    avg_loss, final_hidden = avp.forward_loss(images, viewpoints, target)

    assert avg_loss.shape == ()
    assert avg_loss.requires_grad
    assert final_hidden.shape == (B, 64, backbone.embed_dim)

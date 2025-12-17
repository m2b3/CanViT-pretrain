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
    scene_grid_size = 8
    cfg = AVPConfig(glimpse_grid_size=7, n_scene_registers=0)
    avp = AVPViT(backbone, cfg, teacher_dim=backbone.embed_dim)

    B = 2
    images = torch.randn(B, 3, 128, 128)
    viewpoints = [Viewpoint.full_scene(B, images.device)]
    hidden = avp.init_hidden(B, scene_grid_size)

    scene, final_hidden = avp(images, viewpoints, hidden)

    assert scene.shape == (B, 64, backbone.embed_dim)
    # hidden = [cls | registers | spatial]
    assert final_hidden.shape == (B, avp.n_cls + 64, backbone.embed_dim)


def test_hidden_is_normalized_at_init(backbone: ViTBackbone) -> None:
    """With layer_scale=0, hidden_out equals LN(hidden_in)."""
    scene_grid_size = 8
    cfg = AVPConfig(
        glimpse_grid_size=7,
        layer_scale_init=0.0,
        n_scene_registers=0,
    )
    avp = AVPViT(backbone, cfg, teacher_dim=backbone.embed_dim)

    B = 2
    images = torch.randn(B, 3, 128, 128)
    viewpoints = [Viewpoint.full_scene(B, images.device)]
    hidden = avp.init_hidden(B, scene_grid_size)

    _, final_hidden = avp(images, viewpoints, hidden)

    expected = avp._normalize_hidden(hidden)
    assert torch.allclose(final_hidden, expected, atol=1e-5)


def test_multi_viewpoint_forward(backbone: ViTBackbone) -> None:
    """Forward with multiple viewpoints processes all."""
    scene_grid_size = 8
    cfg = AVPConfig(glimpse_grid_size=7, n_scene_registers=0)
    avp = AVPViT(backbone, cfg, teacher_dim=backbone.embed_dim)

    B = 2
    images = torch.randn(B, 3, 128, 128)
    viewpoints = [
        Viewpoint.full_scene(B, images.device),
        Viewpoint.quadrant(B, images.device, 0, 0),
        Viewpoint.quadrant(B, images.device, 1, 1),
    ]
    hidden = avp.init_hidden(B, scene_grid_size)

    scene, final_hidden = avp(images, viewpoints, hidden)

    assert scene.shape == (B, 64, backbone.embed_dim)
    # hidden = [cls | registers | spatial]
    assert final_hidden.shape == (B, avp.n_cls + 64, backbone.embed_dim)


def test_forward_loss(backbone: ViTBackbone) -> None:
    """forward_loss computes averaged MSE correctly."""
    scene_grid_size = 8
    cfg = AVPConfig(glimpse_grid_size=7, n_scene_registers=0, use_cls_loss=False)
    avp = AVPViT(backbone, cfg, teacher_dim=backbone.embed_dim)

    B = 2
    images = torch.randn(B, 3, 128, 128)
    viewpoints = [
        Viewpoint.full_scene(B, images.device),
        Viewpoint.quadrant(B, images.device, 0, 0),
    ]
    target = torch.randn(B, 64, backbone.embed_dim)
    hidden = avp.init_hidden(B, scene_grid_size)

    losses, final_hidden = avp.forward_loss(images, viewpoints, target, hidden)

    assert losses.scene.shape == ()
    assert losses.scene.requires_grad
    # use_local_loss=False by default, use_cls_loss=False in config
    assert losses.local is None
    assert losses.cls is None
    # hidden = [cls | registers | spatial], CLS always present
    assert final_hidden.shape == (B, avp.n_cls + 64, backbone.embed_dim)

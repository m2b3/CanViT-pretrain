"""Model creation and loading utilities."""

import copy
import logging

import torch
from dinov3.hub.backbones import dinov3_vits16

from avp_vit import AVPViT
from avp_vit.backbone.dinov3 import DINOv3Backbone

from .config import Config

log = logging.getLogger(__name__)


def load_teacher(cfg: Config) -> DINOv3Backbone:
    """Load frozen DINOv3 teacher backbone."""
    log.info(f"Loading teacher from {cfg.teacher_ckpt}")
    model = dinov3_vits16(weights=str(cfg.teacher_ckpt), pretrained=True)
    backbone = DINOv3Backbone(model.eval().to(cfg.device))
    for p in backbone.parameters():
        p.requires_grad = False
    log.info(f"Teacher loaded: {backbone.n_blocks} blocks, embed_dim={backbone.embed_dim}")
    return backbone


def create_avp(teacher: DINOv3Backbone, cfg: Config) -> AVPViT:
    """Create AVP model with copied backbone."""
    patch_size = teacher.patch_size
    scene_px = cfg.avp.scene_grid_size * patch_size
    glimpse_px = cfg.avp.glimpse_grid_size * patch_size
    log.info(
        f"Creating AVP: scene={cfg.avp.scene_grid_size}x{cfg.avp.scene_grid_size} ({scene_px}px), "
        f"glimpse={cfg.avp.glimpse_grid_size}x{cfg.avp.glimpse_grid_size} ({glimpse_px}px)"
    )
    backbone_copy = copy.deepcopy(teacher)
    for p in backbone_copy.parameters():
        p.requires_grad = not cfg.freeze_inner_backbone
    avp = AVPViT(backbone_copy, cfg.avp).to(cfg.device)
    log.info(f"AVP created: inner backbone frozen={cfg.freeze_inner_backbone}")
    return avp


def compile_teacher(teacher: DINOv3Backbone) -> None:
    """Wrap teacher DINOv3 blocks with torch.compile(dynamic=True)."""
    n_blocks = teacher.n_blocks
    log.info(f"Compiling teacher: {n_blocks} self-attention blocks")

    blocks = teacher._backbone.blocks
    for i in range(n_blocks):
        blocks[i] = torch.compile(blocks[i], dynamic=True)  # type: ignore[assignment]

    log.info("Teacher compilation complete")


def compile_avp(avp: AVPViT) -> None:
    """Wrap AVP DINOv3 blocks and cross-attention with torch.compile(dynamic=True)."""
    n_blocks = avp.backbone.n_blocks
    n_adapters = avp.n_adapters
    log.info(
        f"Compiling AVP: {n_blocks} backbone blocks + {n_adapters} read/write attention pairs"
    )

    assert isinstance(avp.backbone, DINOv3Backbone)
    blocks = avp.backbone._backbone.blocks
    for i in range(n_blocks):
        blocks[i] = torch.compile(blocks[i], dynamic=True)  # type: ignore[assignment]

    for i in range(n_adapters):
        avp.read_attn[i] = torch.compile(avp.read_attn[i], dynamic=True)  # type: ignore[assignment]
        avp.write_attn[i] = torch.compile(avp.write_attn[i], dynamic=True)  # type: ignore[assignment]

    log.info("AVP compilation complete")

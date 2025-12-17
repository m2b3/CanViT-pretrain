"""Model creation and loading utilities."""

import logging
from collections.abc import Callable

import torch
from dinov3.hub.backbones import (
    dinov3_vitb16,  # pyright: ignore[reportAttributeAccessIssue]
    dinov3_vitl16,  # pyright: ignore[reportAttributeAccessIssue]
    dinov3_vitl16plus,  # pyright: ignore[reportAttributeAccessIssue]
    dinov3_vits16,
    dinov3_vits16plus,  # pyright: ignore[reportAttributeAccessIssue]
)
from dinov3.models.vision_transformer import DinoVisionTransformer

from avp_vit import AVPViT
from avp_vit.backbone.dinov3 import DINOv3Backbone

from .config import Config

log = logging.getLogger(__name__)

# Registry: model slug -> factory function
MODEL_REGISTRY: dict[str, Callable[..., DinoVisionTransformer]] = {
    "dinov3_vits16": dinov3_vits16,
    "dinov3_vits16plus": dinov3_vits16plus,
    "dinov3_vitb16": dinov3_vitb16,
    "dinov3_vitl16": dinov3_vitl16,
    "dinov3_vitl16plus": dinov3_vitl16plus,
}


def _load_dinov3(
    model_slug: str,
    checkpoint: str | None,
    device: torch.device,
) -> DinoVisionTransformer:
    """Load a DINOv3 model by slug, optionally with pretrained weights."""
    if model_slug not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model: {model_slug!r}. Available: {available}")

    factory = MODEL_REGISTRY[model_slug]

    if checkpoint is None:
        log.info(f"Creating {model_slug} with random initialization")
        model = factory(pretrained=False)
    else:
        log.info(f"Loading {model_slug} from {checkpoint}")
        model = factory(pretrained=True, weights=checkpoint)

    return model.to(device)


def load_teacher(cfg: Config) -> DINOv3Backbone:
    """Load frozen DINOv3 teacher backbone."""
    model = _load_dinov3(cfg.teacher_model, str(cfg.teacher_ckpt), cfg.device)
    backbone = DINOv3Backbone(model.eval())
    for p in backbone.parameters():
        p.requires_grad = False
    log.info(
        f"Teacher ready: {cfg.teacher_model}, "
        f"{backbone.n_blocks} blocks, embed_dim={backbone.embed_dim}"
    )
    return backbone


def load_student_backbone(cfg: Config) -> DINOv3Backbone:
    """Load student DINOv3 backbone (pretrained or random init)."""
    ckpt = str(cfg.student_ckpt) if cfg.student_ckpt is not None else None
    model = _load_dinov3(cfg.student_model, ckpt, cfg.device)
    backbone = DINOv3Backbone(model)
    log.info(
        f"Student backbone ready: {cfg.student_model}, "
        f"{backbone.n_blocks} blocks, embed_dim={backbone.embed_dim}, "
        f"pretrained={cfg.student_ckpt is not None}"
    )
    return backbone


def create_avp(
    student_backbone: DINOv3Backbone,
    teacher_dim: int,
    cfg: Config,
) -> AVPViT:
    """Create AVP model wrapping student backbone, projecting to teacher_dim."""
    patch_size = student_backbone.patch_size
    glimpse_px = cfg.avp.glimpse_grid_size * patch_size

    for p in student_backbone.parameters():
        p.requires_grad = not cfg.freeze_student_backbone

    avp = AVPViT(student_backbone, cfg.avp, teacher_dim).to(cfg.device)

    log.info(
        f"AVP created: grid_sizes={cfg.grid_sizes}, "
        f"glimpse={cfg.avp.glimpse_grid_size}x{cfg.avp.glimpse_grid_size} ({glimpse_px}px), "
        f"student_dim={student_backbone.embed_dim} -> teacher_dim={teacher_dim}, "
        f"freeze_student_backbone={cfg.freeze_student_backbone}"
    )
    return avp


def compile_teacher(teacher: DINOv3Backbone) -> None:
    """Compile teacher DINOv3 blocks in-place."""
    n_blocks = teacher.n_blocks
    log.info(f"Compiling teacher: {n_blocks} self-attention blocks")

    blocks = teacher._backbone.blocks
    for i in range(n_blocks):
        blocks[i].compile(dynamic=True)

    log.info("Teacher compilation complete")


def compile_avp(avp: AVPViT) -> None:
    """Compile AVP DINOv3 blocks and cross-attention in-place."""
    n_blocks = avp.backbone.n_blocks
    n_adapters = avp.n_adapters
    log.info(
        f"Compiling AVP: {n_blocks} backbone blocks + {n_adapters} read/write attention pairs"
    )

    assert isinstance(avp.backbone, DINOv3Backbone)
    blocks = avp.backbone._backbone.blocks
    for i in range(n_blocks):
        blocks[i].compile(dynamic=True)

    for i in range(n_adapters):
        avp.read_attn[i].compile(dynamic=True)
        avp.write_attn[i].compile(dynamic=True)

    log.info("AVP compilation complete")

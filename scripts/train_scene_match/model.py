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

from avp_vit import ActiveCanViT
from canvit.backbone.dinov3 import DINOv3Backbone

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

    # Disable RoPE coordinate rescaling (training-time augmentation that adds noise)
    model.rope_embed.rescale_coords = None

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


def create_model(
    student_backbone: DINOv3Backbone,
    teacher_dim: int,
    cfg: Config,
) -> ActiveCanViT:
    """Create ActiveCanViT wrapping student backbone, projecting to teacher_dim."""
    patch_size = student_backbone.patch_size_px
    glimpse_px = cfg.model.glimpse_grid_size * patch_size

    for p in student_backbone.parameters():
        p.requires_grad = not cfg.freeze_student_backbone

    model = ActiveCanViT(student_backbone, cfg.model, teacher_dim).to(cfg.device)

    log.info(
        f"Model created: grid_size={cfg.grid_size}, "
        f"glimpse={cfg.model.glimpse_grid_size}x{cfg.model.glimpse_grid_size} ({glimpse_px}px), "
        f"student_dim={student_backbone.embed_dim} -> teacher_dim={teacher_dim}, "
        f"freeze_student_backbone={cfg.freeze_student_backbone}"
    )
    return model


def compile_teacher(teacher: DINOv3Backbone) -> None:
    """Compile teacher DINOv3 blocks in-place."""
    teacher.compile()


def compile_model(model: ActiveCanViT) -> None:
    """Compile ActiveCanViT in-place."""
    model.compile()

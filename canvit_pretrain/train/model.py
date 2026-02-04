"""Model creation and loading utilities."""

import logging
from typing import NamedTuple

from canvit import create_backbone
from canvit.backbone.dinov3 import DINOv3Backbone
from canvit.policy import PolicyConfig, PolicyHead
from canvit_utils.teacher import load_teacher as _load_teacher

from canvit_pretrain import CanViTForPretraining

from .config import Config

log = logging.getLogger(__name__)


class ModelBundle(NamedTuple):
    """Model with derived runtime parameters."""

    model: CanViTForPretraining
    glimpse_size_px: int


def load_teacher(cfg: Config) -> DINOv3Backbone:
    """Load frozen DINOv3 teacher backbone."""
    return _load_teacher(cfg.teacher_model, cfg.device, weights=str(cfg.teacher_ckpt))


def load_student_backbone(cfg: Config) -> DINOv3Backbone:
    """Load student DINOv3 backbone (pretrained or random init)."""
    weights = str(cfg.student_ckpt) if cfg.student_ckpt else None
    backbone = create_backbone(
        cfg.student_model, pretrained=weights is not None, weights=weights
    )
    log.info(
        f"Student backbone ready: {cfg.student_model}, pretrained={weights is not None}"
    )
    return backbone.to(cfg.device)


def create_model(
    student_backbone: DINOv3Backbone,
    teacher_dim: int,
    cfg: Config,
) -> ModelBundle:
    """Create CanViTForPretraining wrapping student backbone."""
    cfg.model.teacher_dim = teacher_dim

    policy = None
    if cfg.enable_policy:
        policy_cfg = PolicyConfig()
        policy = PolicyHead(embed_dim=student_backbone.embed_dim, cfg=policy_cfg)
        log.info(f"Policy head created: embed_dim={student_backbone.embed_dim}")

    model = CanViTForPretraining(backbone=student_backbone, cfg=cfg.model, policy=policy).to(
        cfg.device
    )
    glimpse_size_px = cfg.glimpse_grid_size * student_backbone.patch_size_px

    log.info(
        f"Model created: canvas={cfg.grid_size}x{cfg.grid_size}, "
        f"glimpse={cfg.glimpse_grid_size}x{cfg.glimpse_grid_size} ({glimpse_size_px}px), "
        f"student_dim={student_backbone.embed_dim} -> teacher_dim={teacher_dim}, "
    )
    return ModelBundle(model, glimpse_size_px)


def compile_teacher(teacher: DINOv3Backbone) -> None:
    """Compile teacher DINOv3 blocks in-place."""
    teacher.compile()


def compile_model(model: CanViTForPretraining) -> None:
    """Compile CanViTForPretraining in-place."""
    model.compile()

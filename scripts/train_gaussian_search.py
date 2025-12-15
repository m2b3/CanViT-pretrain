"""Train AVP policy to find target-colored gaussian among distractors.

Visual search task demonstrating end-to-end differentiable policy learning.

Task:
- Canvas with N colored gaussian blobs (different colors)
- A target color is given as query
- Policy must center viewpoint on the blob matching target color

Key elements:
- Policy network: (hidden state, target color) → viewpoint
- Same image + different query → different expected behavior
- Reparameterization trick for differentiable sampling
- Reward: -distance(viewpoint_center, target_blob_center)
"""

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path

import comet_ml
import torch
import torch.nn as nn
from dinov3.hub.backbones import dinov3_vits16
from torch import Tensor
from tqdm import tqdm
from ytch.device import get_sensible_device
from ytch.model import count_parameters

from avp_vit import AVPConfig, AVPViT
from avp_vit.backbone.dinov3 import DINOv3Backbone
from avp_vit.glimpse import Viewpoint, extract_glimpse
from avp_vit.train import warmup_cosine_scheduler
from avp_vit.train_gaussians import (
    generate_multi_blob_batch,
    log_figure,
    plot_policy_scatter,
    plot_scale_distribution,
    plot_scene_pca,
    plot_trajectory_with_glimpses,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ============================================================================
# Policy network: context token → viewpoint
# ============================================================================


class ViewpointPolicy(nn.Module):
    """Decode viewpoint from context token + spatial hidden mean.

    Architecture: cat(ctx_norm, spatial_mean_norm) → MLP → heads

    Bounding: center first, then scale constrained by center.
    - centers = tanh(logits) * (1 - min_scale)
    - max_valid_scale = 1 - max(|x|, |y|)
    - scale = sigmoid(logit) * (max_valid_scale - min_scale) + min_scale
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_hidden: int,
        min_scale: float,
        max_scale: float,
        noise_std: float,
        center_head_init_scale: float,
        scale_head_init_scale: float,
        fixed_scale: float | None,
    ) -> None:
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.noise_std = noise_std
        self.fixed_scale = fixed_scale

        # Context embedding: target_colors [B, 3] → [B, D]
        self.context_embed = nn.Linear(3, embed_dim)

        # Learnable inits for first viewpoint (before any scene info)
        self.ctx_init = nn.Parameter(torch.randn(1, embed_dim) / (embed_dim**0.5))
        self.spatial_init = nn.Parameter(torch.randn(1, embed_dim) / (embed_dim**0.5))

        # Norms for ctx and spatial mean
        self.ctx_norm = nn.LayerNorm(embed_dim)
        self.spatial_norm = nn.LayerNorm(embed_dim)

        # MLP takes concatenated ctx + spatial_mean
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU(),
        )
        self.center_head = nn.Linear(mlp_hidden, 2)
        self.scale_head = nn.Linear(mlp_hidden, 1)

        self._init_weights(center_head_init_scale, scale_head_init_scale)

    def _init_weights(self, center_init: float, scale_init: float) -> None:
        nn.init.uniform_(self.center_head.weight, -center_init, center_init)
        nn.init.zeros_(self.center_head.bias)
        nn.init.uniform_(self.scale_head.weight, -scale_init, scale_init)
        nn.init.zeros_(self.scale_head.bias)

    def embed_context(self, target_colors: Tensor) -> Tensor:
        """Embed target colors to context token [B, 1, D]."""
        color_norm = (target_colors - 0.5) / 0.4  # normalize to ~N(0,1)
        return self.context_embed(color_norm).unsqueeze(1)  # [B, 1, D]

    def forward(
        self, ctx: Tensor, spatial: Tensor, deterministic: bool = False
    ) -> tuple[Viewpoint, dict[str, Tensor]]:
        """Decode viewpoint from context token + spatial hidden mean.

        Args:
            ctx: Context token [B, D] or [B, 1, D]
            spatial: Spatial hidden tokens [B, N, D] or mean [B, D]
            deterministic: If True, no noise added
        """
        if ctx.ndim == 3:
            ctx = ctx.squeeze(1)  # [B, 1, D] → [B, D]
        if spatial.ndim == 3:
            spatial = spatial.mean(dim=1)  # [B, N, D] → [B, D]

        x = torch.cat([self.ctx_norm(ctx), self.spatial_norm(spatial)], dim=-1)
        x = self.mlp(x)

        # Heads
        center_logits = self.center_head(x)  # [B, 2]
        scale_logit = self.scale_head(x).squeeze(-1)  # [B]

        # Add noise to logits, then bound
        if deterministic:
            noisy_center = center_logits
            noisy_scale_logit = scale_logit
        else:
            noisy_center = (
                center_logits + torch.randn_like(center_logits) * self.noise_std
            )
            noisy_scale_logit = (
                scale_logit + torch.randn_like(scale_logit) * self.noise_std
            )

        # Center: tanh bounds to valid range (independent of scale)
        max_center_offset = 1 - self.min_scale
        centers = torch.tanh(noisy_center) * max_center_offset

        # Scale: constrained by center position
        max_valid_scale = 1 - torch.max(torch.abs(centers), dim=-1).values  # [B]

        if self.fixed_scale is not None:
            scale = torch.minimum(
                torch.full_like(scale_logit, self.fixed_scale),
                max_valid_scale,
            )
        else:
            scale = (
                torch.sigmoid(noisy_scale_logit) * (max_valid_scale - self.min_scale)
                + self.min_scale
            )

        stats = {
            "center_logits_y": center_logits[:, 0],
            "center_logits_x": center_logits[:, 1],
            "scale_logit": scale_logit,
            "center_y": centers[:, 0],
            "center_x": centers[:, 1],
            "scale": scale,
            "max_valid_scale": max_valid_scale,
        }

        viewpoint = Viewpoint(name="policy", centers=centers, scales=scale)
        return viewpoint, stats


# ============================================================================
# Policy-specific utilities
# ============================================================================


def compute_policy_grad_norms(policy: ViewpointPolicy) -> dict[str, float]:
    """Compute gradient norms for policy components."""

    def module_grad_norm(module: nn.Module) -> float:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if not grads:
            return 0.0
        return torch.stack([g.norm(2) for g in grads]).norm(2).item()

    return {
        "grad_policy_total": module_grad_norm(policy),
        "grad_context_embed": module_grad_norm(policy.context_embed),
        "grad_ctx_norm": module_grad_norm(policy.ctx_norm),
        "grad_spatial_norm": module_grad_norm(policy.spatial_norm),
        "grad_mlp": module_grad_norm(policy.mlp),
        "grad_center_head": module_grad_norm(policy.center_head),
        "grad_scale_head": module_grad_norm(policy.scale_head),
    }


def log_grad_breakdown(norms: dict[str, float], step: int) -> None:
    """Log detailed grad breakdown to stdout."""
    log.info(f"Step {step} grad breakdown:")
    for name, val in norms.items():
        log.info(f"  {name}: {val:.6f}")


def summarize_policy_stats(stats: dict[str, Tensor]) -> dict[str, float]:
    """Summarize policy stats into key metrics.

    batch_spread_logits: How varied are pre-tanh outputs across the batch?
        Low = policy outputs similar values for all inputs (not learning to differentiate)
        High = policy outputs different values per input (learning)
    """
    return {
        "batch_spread_logits": (
            stats["center_logits_y"].std() + stats["center_logits_x"].std()
        ).item()
        / 2,
        "batch_spread_centers": (
            stats["center_y"].std() + stats["center_x"].std()
        ).item()
        / 2,
        "scale_mean": stats["scale"].mean().item(),
        "scale_std": stats["scale"].std().item(),
    }


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class Config:
    # Paths
    teacher_ckpt: Path = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    avp_ckpt: Path | None = None  # Pretrained AVP (e.g., from train_gaussian_recon)
    # Model
    # scene_size = scene_grid_size * patch_size (14), glimpse similarly
    # min_scale = glimpse_grid_size / scene_grid_size
    avp: AVPConfig = field(
        default_factory=lambda: AVPConfig(
            scene_grid_size=16,  # 16*14=224px scene
            glimpse_grid_size=4,  # 4*14=56px glimpse, min_scale=0.25
            gate_init=1e-4,
            use_output_proj=True,
            use_scene_registers=True,
            gradient_checkpointing=False,
        )
    )
    policy_mlp_hidden: int = 256
    policy_noise_std: float = 0.1
    policy_center_head_init_scale: float = 0.01
    policy_scale_head_init_scale: float = 0.01
    policy_fixed_scale: float | None = None
    # Training
    n_steps_per_episode: int = 4
    n_steps: int = 10000
    batch_size: int = 64
    ref_lr: float = 1e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 5000
    grad_clip: float = 1.0
    adam_beta1: float = 0.85
    adam_beta2: float = 0.995
    # Task
    n_blobs: int = 2
    blob_margin: float = 0.3
    blob_sigma_min: float = 0.08
    blob_sigma_max: float = 0.12
    # Logging
    log_every: int = 20
    val_every: int = 100
    # Compilation
    compile: bool = False
    # Runtime
    device: torch.device = field(default_factory=get_sensible_device)

    @property
    def min_viewpoint_scale(self) -> float:
        return self.avp.glimpse_grid_size / self.avp.scene_grid_size

    @property
    def max_viewpoint_scale(self) -> float:
        return 1.0


# ============================================================================
# Model loading
# ============================================================================


def load_backbone(cfg: Config) -> DINOv3Backbone:
    model = dinov3_vits16(weights=str(cfg.teacher_ckpt), pretrained=True)
    backbone = DINOv3Backbone(model.eval().to(cfg.device))
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def create_avp(backbone: DINOv3Backbone, cfg: Config) -> AVPViT:
    backbone_copy = copy.deepcopy(backbone)
    for p in backbone_copy.parameters():
        p.requires_grad = False
    return AVPViT(backbone_copy, cfg.avp).to(cfg.device)


def compile_model(avp: AVPViT) -> None:
    assert isinstance(avp.backbone, DINOv3Backbone)
    blocks = avp.backbone._backbone.blocks
    for i in range(avp.backbone.n_blocks):
        blocks[i] = torch.compile(blocks[i])  # type: ignore[assignment]
    for i in range(avp.backbone.n_blocks):
        avp.read_attn[i] = torch.compile(avp.read_attn[i])  # type: ignore[assignment]
        avp.write_attn[i] = torch.compile(avp.write_attn[i])  # type: ignore[assignment]


# ============================================================================
# Training
# ============================================================================


def compute_distance_loss(viewpoint: Viewpoint, target_centers: Tensor) -> Tensor:
    """Mean L2 distance between viewpoint centers and target centers."""
    return torch.norm(viewpoint.centers - target_centers, dim=-1).mean()


def evaluate_policy(
    exp: comet_ml.Experiment,
    policy: ViewpointPolicy,
    avp: AVPViT,
    cfg: Config,
    step: int,
    prefix: str = "eval",
) -> float:
    """Evaluate policy with deterministic inference and log visualizations.

    Returns final distance.
    """
    image_size = avp.scene_size
    glimpse_size = avp.cfg.glimpse_grid_size * avp.backbone.patch_size

    with torch.inference_mode():
        images, target_colors, target_centers, _ = generate_multi_blob_batch(
            cfg.batch_size,
            image_size,
            cfg.n_blobs,
            cfg.device,
            margin=cfg.blob_margin,
            sigma_range=(cfg.blob_sigma_min, cfg.blob_sigma_max),
        )

        B = images.shape[0]
        hidden = avp._init_hidden(B, None)
        ctx_in = policy.embed_context(target_colors)  # [B, 1, D] fresh, passed to AVP
        # First step: learnable inits (before any scene info)
        ctx_for_policy = policy.ctx_init.expand(B, -1)
        spatial_for_policy = policy.spatial_init.expand(B, -1)

        # Run episode with deterministic policy - collect viewpoints and glimpses
        viewpoints: list[Viewpoint] = []
        glimpses: list[Tensor] = []
        scales_det: list[Tensor] = []
        dists_t: list[Tensor] = []
        # Normalize init hidden for fair PCA comparison (model normalizes at each step)
        # Use get_spatial to exclude registers
        hiddens_for_viz: list[Tensor] = [avp.get_spatial(avp.scene_input_norm(hidden.clone()))]

        for t in range(cfg.n_steps_per_episode):
            vp, stats = policy(ctx_for_policy, spatial_for_policy, deterministic=True)
            viewpoints.append(vp)
            scales_det.append(stats["scale"])
            dists_t.append(torch.norm(vp.centers - target_centers, dim=-1).mean())

            # Extract glimpse for viz
            glimpse = extract_glimpse(images, vp, glimpse_size)
            glimpses.append(glimpse)

            out = avp.forward_step(images, vp, hidden, None, ctx_in)  # fresh ctx to AVP
            hidden = out.hidden
            ctx_for_policy = out.context_out  # transformed ctx for policy
            spatial_for_policy = avp.get_spatial(hidden)  # spatial mean for policy
            hiddens_for_viz.append(avp.get_spatial(hidden.clone()))

        final_vp = viewpoints[-1]
        final_dist = dists_t[-1].item()

        # Also run stochastic to compare scales (using final context_out)
        vp_noised, stats_noised = policy(ctx_for_policy, spatial_for_policy, deterministic=False)
        scales_noised = stats_noised["scale"]

        # Log per-timestep distance
        for t, d in enumerate(dists_t):
            exp.log_metric(f"{prefix}/dist_t{t}", d.item(), step=step)
        exp.log_metric(f"{prefix}/dist", final_dist, step=step)

        # Scatter plot of final predictions (batch overview)
        fig = plot_policy_scatter(
            final_vp.centers, target_centers, target_colors, f"{prefix} - Step {step}"
        )
        log_figure(exp, fig, f"{prefix}/scatter", step)

        # Trajectory with glimpses (detailed view of one sample)
        fig = plot_trajectory_with_glimpses(
            images, target_colors, target_centers, viewpoints, glimpses, sample_idx=0
        )
        log_figure(exp, fig, f"{prefix}/trajectory", step)

        # Scale distribution
        fig = plot_scale_distribution(
            torch.cat(scales_det),  # All timesteps deterministic
            scales_noised,  # Single step with noise
            title=f"Scale Distribution - Step {step}",
        )
        log_figure(exp, fig, f"{prefix}/scales", step)

        # Scene hidden PCA across timesteps
        fig = plot_scene_pca(hiddens_for_viz, cfg.avp.scene_grid_size, sample_idx=0)
        log_figure(exp, fig, f"{prefix}/scene_pca", step)

        return final_dist


def train(cfg: Config) -> None:
    """Train policy to find target-colored gaussian blobs."""
    log.info(f"Device: {cfg.device}")
    log.info(f"Config: {cfg}")

    exp = comet_ml.Experiment(
        project_name="avp-gaussian-search", auto_metric_logging=False
    )
    exp.log_parameters(
        {
            k: str(v) if isinstance(v, (torch.device, Path)) else v
            for k, v in cfg.__dict__.items()
        }
    )

    log.info("Loading backbone...")
    backbone = load_backbone(cfg)
    log.info(f"Backbone params: {count_parameters(backbone):,}")

    log.info("Creating AVP model...")
    avp = create_avp(backbone, cfg)
    if cfg.avp_ckpt is not None:
        log.info(f"Loading pretrained AVP from {cfg.avp_ckpt}")
        avp.load_state_dict(torch.load(cfg.avp_ckpt, map_location=cfg.device, weights_only=True))
    if cfg.compile:
        compile_model(avp)

    log.info("Creating policy...")
    policy = ViewpointPolicy(
        embed_dim=backbone.embed_dim,
        mlp_hidden=cfg.policy_mlp_hidden,
        min_scale=cfg.min_viewpoint_scale,
        max_scale=cfg.max_viewpoint_scale,
        noise_std=cfg.policy_noise_std,
        center_head_init_scale=cfg.policy_center_head_init_scale,
        scale_head_init_scale=cfg.policy_scale_head_init_scale,
        fixed_scale=cfg.policy_fixed_scale,
    ).to(cfg.device)
    log.info(f"Policy params: {count_parameters(policy):,}")

    peak_lr = cfg.ref_lr * cfg.batch_size
    all_params = list(avp.parameters()) + list(policy.parameters())
    optimizer = torch.optim.AdamW(
        all_params,
        lr=peak_lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
    )
    scheduler = warmup_cosine_scheduler(optimizer, cfg.n_steps, cfg.warmup_steps)
    log.info(
        f"Optimizer: peak_lr={peak_lr:.2e}, warmup={cfg.warmup_steps}, betas=({cfg.adam_beta1}, {cfg.adam_beta2})"
    )

    exp.log_parameters({"policy_params": count_parameters(policy), "peak_lr": peak_lr})

    # =========== INITIAL EVAL (step 0, before training) ===========
    log.info("Initial evaluation...")
    init_dist = evaluate_policy(exp, policy, avp, cfg, step=0, prefix="eval")
    log.info(f"Init dist: {init_dist:.4f}")

    ema_loss = torch.tensor(0.0, device=cfg.device)
    alpha = 2 / (cfg.log_every + 1)
    image_size = avp.scene_size
    glimpse_size = avp.cfg.glimpse_grid_size * avp.backbone.patch_size

    log.info("Starting training...")
    pbar = tqdm(range(cfg.n_steps), desc="Training", unit="step")

    for step in pbar:
        # Generate batch
        images, target_colors, target_centers, _ = generate_multi_blob_batch(
            cfg.batch_size,
            image_size,
            cfg.n_blobs,
            cfg.device,
            margin=cfg.blob_margin,
            sigma_range=(cfg.blob_sigma_min, cfg.blob_sigma_max),
        )

        # Run episode (stochastic policy for exploration)
        # Loss is averaged across ALL timesteps
        # Collect viewpoints/glimpses for trajectory viz at val_every
        B = images.shape[0]
        hidden = avp._init_hidden(B, None)
        ctx_in = policy.embed_context(target_colors)  # [B, 1, D] fresh, passed to AVP
        # First step: learnable inits (before any scene info)
        ctx_for_policy = policy.ctx_init.expand(B, -1)
        spatial_for_policy = policy.spatial_init.expand(B, -1)
        first_stats = None
        train_viewpoints: list[Viewpoint] = []
        train_glimpses: list[Tensor] = []

        timestep_losses = []
        for t in range(cfg.n_steps_per_episode):
            vp, stats = policy(ctx_for_policy, spatial_for_policy, deterministic=False)
            if t == 0:
                first_stats = stats
            train_viewpoints.append(vp)
            train_glimpses.append(extract_glimpse(images, vp, glimpse_size))
            timestep_losses.append(compute_distance_loss(vp, target_centers))
            out = avp.forward_step(images, vp, hidden, None, ctx_in)  # fresh ctx to AVP
            hidden = out.hidden
            ctx_for_policy = out.context_out  # transformed ctx for policy
            spatial_for_policy = avp.get_spatial(hidden)  # spatial mean for policy

        loss = torch.stack(timestep_losses).mean()  # average over timesteps

        optimizer.zero_grad()
        loss.backward()

        # Compute grad norms BEFORE clipping (only at log steps to avoid overhead)
        grad_norms = None
        if step % cfg.log_every == 0:
            grad_norms = compute_policy_grad_norms(policy)

        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        # EMA (keep as tensor, no .item() in hot loop)
        ema_loss = (
            loss.detach()
            if step == 0
            else alpha * loss.detach() + (1 - alpha) * ema_loss
        )

        if step % cfg.log_every == 0:
            assert grad_norms is not None
            assert first_stats is not None
            policy_stats = summarize_policy_stats(first_stats)

            # Log detailed grad breakdown on first step
            if step == 0:
                log_grad_breakdown(grad_norms, step)

            exp.log_metrics(
                {
                    "loss": ema_loss.item(),
                    "grad_norm": grad_norm.item(),
                    **grad_norms,
                    "lr": scheduler.get_last_lr()[0],
                    "batch_spread_logits": policy_stats["batch_spread_logits"],
                    "scale_mean": policy_stats["scale_mean"],
                    "scale_std": policy_stats["scale_std"],
                },
                step=step,
            )

            pbar.set_postfix_str(f"loss={ema_loss.item():.3f}")

        # Training trajectory viz (stochastic, shows noise effect)
        if step > 0 and step % cfg.val_every == 0:
            with torch.inference_mode():
                fig = plot_trajectory_with_glimpses(
                    images,
                    target_colors,
                    target_centers,
                    train_viewpoints,
                    train_glimpses,
                    sample_idx=0,
                )
                log_figure(exp, fig, "train/trajectory", step)

            # Eval with visualization (deterministic)
            evaluate_policy(exp, policy, avp, cfg, step=step, prefix="eval")

    # Final eval
    final_dist = evaluate_policy(exp, policy, avp, cfg, step=cfg.n_steps, prefix="eval")
    log.info(f"Training complete. Final dist: {final_dist:.4f}")
    exp.end()


def main() -> None:
    import tyro

    torch.set_float32_matmul_precision("high")
    cfg = tyro.cli(Config)
    train(cfg)


if __name__ == "__main__":
    main()

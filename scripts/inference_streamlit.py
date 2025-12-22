#!/usr/bin/env python3
"""Interactive Streamlit app for model inference with click-based viewpoints."""

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from canvit.backbone.dinov3 import DINOv3Backbone
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
from torch import Tensor
from torchvision import transforms

from dinov3_probes import DINOv3LinearClassificationHead
from torchvision.models.resnet import ResNet50_Weights

from avp_vit import ActiveCanViT, gram_mse
from avp_vit.checkpoint import _get_backbone_factory
from avp_vit.checkpoint import load as load_ckpt
from avp_vit.checkpoint import load_model
from avp_vit.train.data import imagenet_normalize
from avp_vit.train.norm import PositionAwareNorm
from avp_vit.train.probe import load_probe
from avp_vit.train.viewpoint import Viewpoint
from avp_vit.train.viz import fit_pca, imagenet_denormalize, pca_rgb, timestep_colors
from ytch.device import sync_device
IMAGENET_LABELS = ResNet50_Weights.DEFAULT.meta["categories"]

DISPLAY_PX = 512

st.set_page_config(page_title="AVP-ViT", layout="wide")


@dataclass
class StepResult:
    """Result from one inference step."""

    hidden: np.ndarray  # [G², D] pre-projection features
    projected: np.ndarray  # [G², teacher_dim] scene output
    scene_cos_sim: float | None
    cls_cos_sim: float | None
    gram_loss: float | None  # L2-normalized cosine gram
    corr_gram_loss: float | None  # Z-scored correlation gram
    glimpse: np.ndarray | None  # [H, W, 3] normalized to [0, 1]
    model_step_ms: float  # Model forward step latency
    top5_classes: list[int] | None  # Top-5 predicted class indices
    top5_probs: list[float] | None  # Top-5 probabilities


def correlation_gram(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Spatial Gram matrix measuring correlation structure between positions.

    Standard Gram: L2-normalizes per position → cosine similarity (direction alignment).
    Correlation Gram: z-scores per feature → correlation between positions (deviation patterns).

    The z-scoring matches PCA preprocessing: positions with similar patterns of
    deviation from the mean (across features) will have high correlation.
    Normalized by D so values are comparable to standard gram's [-1, 1] range.
    """
    # x: [B, N, D] where N=positions, D=features
    B, N, D = x.shape
    # Z-score per feature (across positions)
    x_centered = x - x.mean(dim=1, keepdim=True)
    x_std = x_centered.std(dim=1, keepdim=True)
    x_zscore = x_centered / (x_std + eps)
    # Gram: [B, N, N] - normalize by D so entries are O(1) not O(D)
    return torch.bmm(x_zscore, x_zscore.transpose(1, 2)) / D


def correlation_gram_mse(pred: Tensor, target: Tensor) -> Tensor:
    """MSE between correlation Gram matrices."""
    return F.mse_loss(correlation_gram(pred), correlation_gram(target))


@dataclass
class LatencyRecord:
    """Latency samples for a specific configuration."""

    samples: list[float]  # Individual measurements in ms

    def add(self, ms: float) -> None:
        self.samples.append(ms)

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples)) if self.samples else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.samples)) if len(self.samples) > 1 else 0.0


def upscale_nearest(img: np.ndarray, target_size: int) -> Image.Image:
    """Upscale with nearest neighbor to preserve blocky pixels."""
    pil = Image.fromarray(
        (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
    )
    return pil.resize((target_size, target_size), Image.Resampling.NEAREST)


@st.cache_resource
def cached_load_model(checkpoint_path: str, device: str) -> ActiveCanViT:
    return load_model(Path(checkpoint_path), torch.device(device))


@st.cache_resource
def cached_load_teacher(checkpoint_path: str, device: str) -> DINOv3Backbone:
    ckpt = load_ckpt(Path(checkpoint_path), "cpu")
    factory = _get_backbone_factory(ckpt["backbone"])
    raw_teacher = factory(pretrained=True)
    return DINOv3Backbone(raw_teacher.to(device).eval())  # type: ignore[arg-type]


@st.cache_resource
def cached_load_normalizers(
    checkpoint_path: str, device: str
) -> tuple[PositionAwareNorm | None, PositionAwareNorm | None]:
    """Load scene and cls normalizers from checkpoint if available."""
    ckpt = load_ckpt(Path(checkpoint_path), "cpu")

    scene_norm = None
    scene_state = ckpt.get("scene_norm_state")
    if scene_state is not None:
        n_tokens, embed_dim = scene_state["mean"].shape
        grid_size = int(n_tokens**0.5)
        scene_norm = PositionAwareNorm(n_tokens, embed_dim, grid_size)
        scene_norm.load_state_dict(scene_state)
        scene_norm.eval()
        scene_norm = scene_norm.to(device)

    cls_norm = None
    cls_state = ckpt.get("cls_norm_state")
    if cls_state is not None:
        n_tokens, embed_dim = cls_state["mean"].shape
        cls_norm = PositionAwareNorm(n_tokens, embed_dim, grid_size=1)
        cls_norm.load_state_dict(cls_state)
        cls_norm.eval()
        cls_norm = cls_norm.to(device)

    return scene_norm, cls_norm


@st.cache_resource
def cached_load_probe(
    checkpoint_path: str, device: str
) -> DINOv3LinearClassificationHead | None:
    """Load classification probe from HF Hub if backbone is supported."""
    ckpt = load_ckpt(Path(checkpoint_path), "cpu")
    return load_probe(ckpt["backbone"], torch.device(device))


def load_and_preprocess(
    pil_img: Image.Image, size: int, device: torch.device
) -> Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            imagenet_normalize(),
        ]
    )
    tensor = transform(pil_img)
    assert isinstance(tensor, Tensor)
    return tensor.unsqueeze(0).to(device)


def pixel_to_viewpoint(
    cx_px: float,
    cy_px: float,
    scale: float,
    H: int,
    W: int,
    device: torch.device,
    name: str,
) -> Viewpoint:
    """Convert pixel click to viewpoint, clamping to stay in bounds."""
    cx = (cx_px / W) * 2 - 1
    cy = (cy_px / H) * 2 - 1
    max_offset = 1.0 - scale
    cx = max(-max_offset, min(max_offset, cx))
    cy = max(-max_offset, min(max_offset, cy))
    # Viewpoint uses (row, col) = (y, x) order
    return Viewpoint(
        name=name,
        centers=torch.tensor([[cy, cx]], device=device),
        scales=torch.tensor([scale], device=device),
    )


def draw_boxes(
    img_pil: Image.Image, viewpoints: list[Viewpoint], H: int, W: int
) -> Image.Image:
    """Draw viewpoint boxes on image with timestep colors."""
    img = img_pil.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    colors = timestep_colors(len(viewpoints)) if viewpoints else []
    for i, vp in enumerate(viewpoints):
        box = vp.to_pixel_box(0, H, W)
        r, g, b = [int(c * 255) for c in colors[i][:3]]
        draw.rectangle(
            [box.left, box.top, box.left + box.width, box.top + box.height],
            outline=(r, g, b, 255),
            width=2,
        )
    return img


def compute_teacher_targets(
    teacher: DINOv3Backbone,
    image: Tensor,
    patch_size: int,
    native_grid: int,
    glimpse_grid: int,
    scene_norm: PositionAwareNorm | None,
    cls_norm: PositionAwareNorm | None,
) -> tuple[Tensor, Tensor, Tensor | None, float, float]:
    """Compute teacher features at native and glimpse resolutions.

    Returns:
        scene_target: [native_grid², D] normalized features for loss
        teacher_at_glimpse: [glimpse_grid², D] raw features for display
        cls_target: [1, D] normalized CLS token
        native_ms: latency for native resolution
        glimpse_ms: latency for glimpse resolution
    """
    # Native resolution (for loss comparison)
    native_px = native_grid * patch_size
    img_native = F.interpolate(
        image, size=(native_px, native_px), mode="bilinear", align_corners=False
    )
    sync_device(image.device)
    t0 = time.perf_counter()
    raw_feats = teacher.forward_norm_features(img_native)
    sync_device(image.device)
    native_ms = (time.perf_counter() - t0) * 1000

    teacher_feats = raw_feats.patches  # [1, native_grid², D]
    if scene_norm is not None:
        teacher_feats = scene_norm(teacher_feats)
    scene_target = teacher_feats[0]

    cls_target = None
    if cls_norm is not None:
        cls_target = cls_norm(raw_feats.cls.unsqueeze(1)).squeeze(1)

    # Glimpse resolution (for display comparison)
    glimpse_px = glimpse_grid * patch_size
    img_glimpse = F.interpolate(
        image, size=(glimpse_px, glimpse_px), mode="bilinear", align_corners=False
    )
    sync_device(image.device)
    t0 = time.perf_counter()
    glimpse_feats = teacher.forward_norm_features(img_glimpse)
    sync_device(image.device)
    glimpse_ms = (time.perf_counter() - t0) * 1000

    teacher_at_glimpse = glimpse_feats.patches[0]  # [glimpse_grid², D]

    return scene_target, teacher_at_glimpse, cls_target, native_ms, glimpse_ms


def run_inference_step(
    model: ActiveCanViT,
    image: Tensor,
    canvas: Tensor,
    vp: Viewpoint,
    glimpse_size_px: int,
    canvas_grid: int,
    teacher_grid: int,
    scene_target: Tensor | None,
    cls_target: Tensor | None,
    apply_hidden_ln: bool,
    probe: DINOv3LinearClassificationHead | None,
    cls_norm: PositionAwareNorm | None,
) -> tuple[Tensor, StepResult]:
    """Run one inference step and compute metrics.

    Metrics are computed at teacher_grid resolution (same as training).

    Returns:
        new_canvas: Updated canvas tensor
        result: StepResult with features and metrics
    """
    sync_device(image.device)
    t0 = time.perf_counter()
    out = model.forward_step(
        image=image,
        canvas=canvas,
        viewpoint=vp,
        glimpse_size_px=glimpse_size_px,
    )
    sync_device(image.device)
    model_step_ms = (time.perf_counter() - t0) * 1000

    spatial = model.get_spatial(out.canvas)[0]
    assert spatial.shape == (canvas_grid * canvas_grid, spatial.shape[-1])

    scene = model.compute_scene(out.canvas)
    assert scene.shape == (1, canvas_grid * canvas_grid, scene.shape[-1])

    hidden = spatial
    if apply_hidden_ln:
        scene_ln = model.scene_head[0]
        assert isinstance(scene_ln, torch.nn.LayerNorm)
        hidden = scene_ln(spatial)

    scene_cos_sim = None
    gram_loss = None
    corr_gram_loss = None
    if scene_target is not None:
        # Downsample scene prediction to teacher resolution for fair comparison
        D = scene.shape[-1]
        scene_spatial = scene.view(1, canvas_grid, canvas_grid, D).permute(0, 3, 1, 2)
        scene_at_teacher = F.interpolate(
            scene_spatial, size=(teacher_grid, teacher_grid), mode="bilinear", align_corners=False
        )
        scene_pred = scene_at_teacher[0].permute(1, 2, 0).reshape(-1, D)  # [teacher_grid², D]

        scene_cos_sim = (
            F.cosine_similarity(scene_pred, scene_target, dim=-1).mean().item()
        )
        pred_batch = scene_pred.unsqueeze(0)
        target_batch = scene_target.unsqueeze(0)
        gram_loss = gram_mse(pred_batch, target_batch).item()
        corr_gram_loss = correlation_gram_mse(pred_batch, target_batch).item()

    cls_cos_sim = None
    cls_pred = model.compute_cls(out.canvas)
    if cls_target is not None:
        cls_cos_sim = F.cosine_similarity(cls_pred, cls_target, dim=-1).mean().item()

    top5_classes: list[int] | None = None
    top5_probs: list[float] | None = None
    if probe is not None and cls_norm is not None:
        cls_raw = cls_norm.denormalize(cls_pred)
        logits = probe(cls_raw)
        probs = F.softmax(logits, dim=-1)
        top5_p, top5_c = probs[0].topk(5)
        top5_classes = top5_c.tolist()
        top5_probs = top5_p.tolist()

    glimpse_np = imagenet_denormalize(out.glimpse[0].cpu()).numpy()

    return out.canvas, StepResult(
        hidden=hidden.cpu().numpy(),
        projected=scene[0].cpu().numpy(),
        scene_cos_sim=scene_cos_sim,
        cls_cos_sim=cls_cos_sim,
        gram_loss=gram_loss,
        corr_gram_loss=corr_gram_loss,
        glimpse=glimpse_np,
        model_step_ms=model_step_ms,
        top5_classes=top5_classes,
        top5_probs=top5_probs,
    )


def reset_state() -> None:
    keys_to_delete = [
        k
        for k in st.session_state.keys()
        if k in ("viewpoints", "results", "pending_click", "canvas", "_state_key")
    ]
    for key in keys_to_delete:
        del st.session_state[key]


def state_key(
    checkpoint: str, device: str, canvas_grid: int, glimpse_grid: int, image_id: str
) -> str:
    """Generate a key that changes when any critical config changes."""
    return f"{checkpoint}:{device}:{canvas_grid}:{glimpse_grid}:{image_id}"


# Sidebar
with st.sidebar:
    st.title("Config")
    checkpoint_path = st.text_input("Checkpoint", value="reference.pt")
    device_name = st.selectbox("Device", ["mps", "cuda", "cpu"], index=0)
    assert isinstance(device_name, str)
    scale = st.slider("Viewpoint scale", 0.05, 1.0, 0.25, 0.05)
    glimpse_grid = st.slider("Glimpse grid", 4, 16, 8, 1)
    canvas_grid = st.slider("Canvas grid", 16, 256, 128, 16)
    pc_offset = st.slider("PC offset", 0, 9, 0, 1, help="Show PC (n+1)-(n+3)")
    hidden_ln = st.checkbox("Hidden LN", value=False)
    if st.button("Reset"):
        reset_state()
        st.rerun()

uploaded = st.file_uploader(
    "Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
)

if not Path(checkpoint_path).exists():
    st.error(f"Checkpoint not found: {checkpoint_path}")
    st.stop()

device = torch.device(device_name)
model = cached_load_model(checkpoint_path, device_name)
patch_size = model.backbone.patch_size_px
glimpse_size_px = glimpse_grid * patch_size
img_size = canvas_grid * patch_size

teacher: DINOv3Backbone | None = None
scene_norm: PositionAwareNorm | None = None
cls_norm: PositionAwareNorm | None = None
probe: DINOv3LinearClassificationHead | None = None
try:
    teacher = cached_load_teacher(checkpoint_path, device_name)
    scene_norm, cls_norm = cached_load_normalizers(checkpoint_path, device_name)
    probe = cached_load_probe(checkpoint_path, device_name)
    if scene_norm:
        st.sidebar.caption(f"Normalizer: {scene_norm.grid_size}² grid")
    if probe:
        st.sidebar.caption("Probe: IN1k classification")
except Exception as e:
    st.warning(f"Teacher: {e}")

if uploaded is None:
    st.info("Upload an image.")
    st.stop()

# Compute state key from all critical parameters
current_key = state_key(checkpoint_path, device_name, canvas_grid, glimpse_grid, uploaded.file_id)

# Reset state if any critical config changed (but keep latency history)
if st.session_state.get("_state_key") != current_key:
    reset_state()
    st.session_state._state_key = current_key
    st.session_state.viewpoints = []
    st.session_state.results = []
    st.session_state.pending_click = None
    st.session_state.canvas = model.init_canvas(batch_size=1, canvas_grid_size=canvas_grid)

# Persistent latency storage: dict[config_label, LatencyRecord]
if "latency_records" not in st.session_state:
    st.session_state.latency_records = {}

latency_records: dict[str, LatencyRecord] = st.session_state.latency_records
model_key = f"Model {glimpse_grid}→{canvas_grid}"
if model_key not in latency_records:
    latency_records[model_key] = LatencyRecord(samples=[])

pil_img = Image.open(uploaded).convert("RGB")
image = load_and_preprocess(pil_img, img_size, device)
img_np = imagenet_denormalize(image[0].cpu()).numpy()
H, W = img_np.shape[:2]
img_pil = Image.fromarray((img_np * 255).astype(np.uint8))

# Compute teacher targets at native and glimpse resolutions
scene_target: Tensor | None = None
teacher_at_glimpse: Tensor | None = None
cls_target: Tensor | None = None
teacher_grid: int = scene_norm.grid_size if scene_norm else 16

# Teacher latency keys
teacher_native_key = f"Teacher {teacher_grid}²"
teacher_glimpse_key = f"Teacher {glimpse_grid}²"
if teacher_native_key not in latency_records:
    latency_records[teacher_native_key] = LatencyRecord(samples=[])
if teacher_glimpse_key not in latency_records:
    latency_records[teacher_glimpse_key] = LatencyRecord(samples=[])

if teacher is not None:
    with torch.no_grad():
        scene_target, teacher_at_glimpse, cls_target, native_ms, glimpse_ms = compute_teacher_targets(
            teacher, image, patch_size, teacher_grid, glimpse_grid, scene_norm, cls_norm
        )
        # Store teacher latencies (accumulate samples)
        latency_records[teacher_native_key].add(native_ms)
        latency_records[teacher_glimpse_key].add(glimpse_ms)

# Display columns
viewpoints: list[Viewpoint] = st.session_state.viewpoints
results: list[StepResult] = st.session_state.results
n = len(viewpoints)

c1, c2, c3, c4, c5, c6 = st.columns(6)
col_width = DISPLAY_PX * 2 // 3  # Smaller for 6 columns

with c1:
    st.markdown("**Input** (click)")
    display_img = draw_boxes(img_pil, viewpoints, H, W)
    display_scale = col_width / W
    display_resized = display_img.resize((col_width, col_width), Image.Resampling.LANCZOS)
    coords = streamlit_image_coordinates(display_resized, key="img")

    if coords is not None:
        cx_display, cy_display = coords["x"], coords["y"]
        cx = cx_display / display_scale
        cy = cy_display / display_scale
        click_key = (cx_display, cy_display, scale)

        if st.session_state.pending_click != click_key:
            st.session_state.pending_click = click_key
            vp = pixel_to_viewpoint(cx, cy, scale, H, W, device, f"t{n}")
            st.session_state.viewpoints.append(vp)

            with torch.no_grad():
                new_canvas, result = run_inference_step(
                    model=model,
                    image=image,
                    canvas=st.session_state.canvas,
                    vp=vp,
                    glimpse_size_px=glimpse_size_px,
                    canvas_grid=canvas_grid,
                    teacher_grid=teacher_grid,
                    scene_target=scene_target,
                    cls_target=cls_target,
                    apply_hidden_ln=hidden_ln,
                    probe=probe,
                    cls_norm=cls_norm,
                )
                st.session_state.canvas = new_canvas
                st.session_state.results.append(result)
                latency_records[model_key].add(result.model_step_ms)

            st.rerun()

pc_label = f"PC{pc_offset+1}-{pc_offset+3}" if pc_offset > 0 else ""

with c2:
    if scene_target is not None:
        st.markdown(f"**Teacher {teacher_grid}²** {pc_label}")
        pca_teacher = fit_pca(scene_target.cpu().numpy())
        teacher_rgb = pca_rgb(
            pca_teacher, scene_target.cpu().numpy(), teacher_grid, teacher_grid, pc_offset=pc_offset
        )
        st.image(upscale_nearest(teacher_rgb, col_width), width=col_width)
    else:
        st.markdown("**Teacher** (not loaded)")

with c3:
    if teacher_at_glimpse is not None:
        st.markdown(f"**Teacher {glimpse_grid}²**")
        pca_glimpse = fit_pca(teacher_at_glimpse.cpu().numpy())
        glimpse_rgb = pca_rgb(
            pca_glimpse, teacher_at_glimpse.cpu().numpy(), glimpse_grid, glimpse_grid, pc_offset=pc_offset
        )
        st.image(upscale_nearest(glimpse_rgb, col_width), width=col_width)
    else:
        st.markdown(f"**Teacher {glimpse_grid}²** (not loaded)")

with c4:
    if teacher_at_glimpse is not None and scene_target is not None:
        st.markdown(f"**{glimpse_grid}²→{teacher_grid}²**")
        # Upsample glimpse features to native grid, then PCA with native basis
        D = teacher_at_glimpse.shape[-1]
        glimpse_spatial = teacher_at_glimpse.view(glimpse_grid, glimpse_grid, D).permute(2, 0, 1).unsqueeze(0)
        upsampled = F.interpolate(
            glimpse_spatial, size=(teacher_grid, teacher_grid), mode="bilinear", align_corners=False
        )
        upsampled_flat = upsampled[0].permute(1, 2, 0).reshape(-1, D).cpu().numpy()
        pca_native = fit_pca(scene_target.cpu().numpy())
        upsampled_rgb = pca_rgb(pca_native, upsampled_flat, teacher_grid, teacher_grid, normalize=True, pc_offset=pc_offset)
        st.image(upscale_nearest(upsampled_rgb, col_width), width=col_width)
    else:
        st.markdown(f"**{glimpse_grid}²→{teacher_grid}²** (not loaded)")

with c5:
    st.markdown(f"**Hidden {canvas_grid}²**" + (f" T{n - 1}" if n > 0 else ""))
    if n > 0:
        hidden = results[-1].hidden
        pca_hidden = fit_pca(hidden)
        hidden_rgb = pca_rgb(pca_hidden, hidden, canvas_grid, canvas_grid, pc_offset=pc_offset)
        st.image(upscale_nearest(hidden_rgb, col_width), width=col_width)

with c6:
    st.markdown(f"**Proj {canvas_grid}²**" + (f" T{n - 1}" if n > 0 else ""))
    if n > 0 and scene_target is not None:
        pca_teacher = fit_pca(scene_target.cpu().numpy())
        proj = results[-1].projected
        proj_rgb = pca_rgb(pca_teacher, proj, canvas_grid, canvas_grid, normalize=True, pc_offset=pc_offset)
        st.image(upscale_nearest(proj_rgb, col_width), width=col_width)
        scene_cos = results[-1].scene_cos_sim
        cls_cos = results[-1].cls_cos_sim
        if scene_cos is not None:
            caption = f"scene={scene_cos:.4f}"
            if cls_cos is not None:
                caption += f" cls={cls_cos:.4f}"
            st.caption(caption)

# Top-5 predictions
if n > 0 and results[-1].top5_classes is not None:
    st.markdown("---")
    st.markdown("**Top-5 IN1k Predictions**")
    top5_c = results[-1].top5_classes
    top5_p = results[-1].top5_probs
    assert top5_c is not None and top5_p is not None
    for rank, (cls_idx, prob) in enumerate(zip(top5_c, top5_p, strict=True), 1):
        label = IMAGENET_LABELS[cls_idx]
        st.markdown(f"{rank}. **{label}** ({100*prob:.1f}%)")

# Metrics plot
if n > 0:
    scene_sims = [r.scene_cos_sim for r in results if r.scene_cos_sim is not None]
    cls_sims = [r.cls_cos_sim for r in results if r.cls_cos_sim is not None]
    gram_losses = [r.gram_loss for r in results if r.gram_loss is not None]
    corr_gram_losses = [r.corr_gram_loss for r in results if r.corr_gram_loss is not None]

    if scene_sims or cls_sims or gram_losses or corr_gram_losses:
        st.markdown("---")
        fig, ax1 = plt.subplots(figsize=(8, 3))

        # Left axis: cosine similarity
        if scene_sims:
            ax1.plot(range(len(scene_sims)), scene_sims, "o-", label="scene cos", markersize=4, color="C0")
        if cls_sims:
            ax1.plot(range(len(cls_sims)), cls_sims, "s-", label="cls cos", markersize=4, color="C1")
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Cosine similarity")
        ax1.set_xlim(-0.5, n - 0.5)

        # Right axis: gram losses (different scale)
        ax2 = None
        if gram_losses or corr_gram_losses:
            ax2 = ax1.twinx()
            if gram_losses:
                ax2.plot(range(len(gram_losses)), gram_losses, "^-", label="gram (cos)", markersize=4, color="C2")
            if corr_gram_losses:
                ax2.plot(range(len(corr_gram_losses)), corr_gram_losses, "v-", label="gram (corr)", markersize=4, color="C3")
            ax2.set_ylabel("Gram loss")
            ax2.tick_params(axis="y")

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels() if ax2 else ([], [])
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

# Timeline
if n > 0 and scene_target is not None:
    st.markdown("**Timeline** (glimpse → proj)")
    pca_teacher = fit_pca(scene_target.cpu().numpy())
    cols = st.columns(min(n, 10))
    for t in range(min(n, 10)):
        with cols[t]:
            glimpse = results[t].glimpse
            if glimpse is not None:
                st.image((glimpse * 255).astype(np.uint8), width=80)
            proj = results[t].projected
            proj_rgb = pca_rgb(pca_teacher, proj, canvas_grid, canvas_grid, normalize=True, pc_offset=pc_offset)
            st.image(upscale_nearest(proj_rgb, 80), width=80)
            scene_cos = results[t].scene_cos_sim
            st.caption(f"T{t}: {scene_cos:.3f}" if scene_cos else f"T{t}")

# Glimpse log
if n > 0:
    with st.expander(f"Glimpse log ({n})"):
        colors = timestep_colors(n)
        for i, vp in enumerate(viewpoints):
            box = vp.to_pixel_box(0, H, W)
            r, g, b = [int(c * 255) for c in colors[i][:3]]
            scene_cos = results[i].scene_cos_sim
            scene_str = f"{scene_cos:.3f}" if scene_cos else "?"
            st.markdown(
                f"T{i} ({box.center_x:.0f},{box.center_y:.0f}) s={vp.scales[0].item():.2f} "
                f"scene={scene_str} <span style='color:rgb({r},{g},{b})'>●</span>",
                unsafe_allow_html=True,
            )

# Latency distribution plot
records_with_data = {k: v for k, v in latency_records.items() if v.samples}
if records_with_data:
    st.markdown("---")
    col_left, col_right = st.columns([3, 1])
    with col_right:
        if st.button("Clear latency data"):
            st.session_state.latency_records = {}
            st.rerun()

    with col_left:
        st.markdown("**Latency distribution** (ms)")

    # Prepare data for violin plot
    all_data = []
    labels = []
    for label, record in sorted(records_with_data.items()):
        all_data.append(record.samples)
        labels.append(f"{label}\n(n={len(record.samples)})")

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4))
    positions = np.arange(len(labels))

    # Violin plot for distribution shape
    parts = ax.violinplot(all_data, positions=positions, showmeans=False, showmedians=False, showextrema=False)
    if "bodies" in parts:
        for i, pc in enumerate(parts["bodies"]):  # type: ignore[arg-type]
            pc.set_facecolor(f"C{i % 10}")
            pc.set_alpha(0.3)

    # Box plot for quartiles
    bp = ax.boxplot(all_data, positions=positions, widths=0.15, patch_artist=True,
                    showfliers=False, medianprops={"color": "black", "linewidth": 1.5})
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(f"C{i % 10}")
        patch.set_alpha(0.7)

    # Strip plot (jittered points)
    for i, data in enumerate(all_data):
        jitter = np.random.normal(0, 0.04, len(data))
        ax.scatter(positions[i] + jitter, data, alpha=0.5, s=15, c=f"C{i % 10}", zorder=3)

    # Mean markers
    means = [np.mean(d) for d in all_data]
    ax.scatter(positions, means, marker="D", s=40, c="white", edgecolors="black", zorder=4, label="mean")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Latency (ms)")
    ax.grid(True, alpha=0.3, axis="y")

    # Add mean values as text
    for i, mean in enumerate(means):
        ax.text(float(i), ax.get_ylim()[1], f"{mean:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

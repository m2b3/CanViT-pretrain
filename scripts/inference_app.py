#!/usr/bin/env python3
"""Streamlit app for AVP-ViT inference. Click image to add viewpoints."""

import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
from canvit.backbone.dinov3 import DINOv3Backbone
from canvit.policy import PolicyHead
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from streamlit_image_coordinates import streamlit_image_coordinates
from torch import Tensor
from torchvision import transforms
from torchvision.models.resnet import ResNet50_Weights

from avp_vit.checkpoint import CheckpointData, load as load_ckpt, load_model
from avp_vit.train.config import Config as TrainConfig
from avp_vit.train.data import imagenet_normalize
from avp_vit.train.norm import PositionAwareNorm
from avp_vit.train.probe import load_probe
from avp_vit.train.viewpoint import Viewpoint
from avp_vit.train.viz import fit_pca, imagenet_denormalize, pca_rgb, timestep_colors
from canvit.hub import create_backbone
from dinov3_probes import DINOv3LinearClassificationHead
from ytch.device import sync_device

LABELS: list[str] = ResNet50_Weights.DEFAULT.meta["categories"]
COL_W = 300

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
st.set_page_config(page_title="AVP-ViT", layout="wide")


# --- Data ---


@dataclass
class Resources:
    """Cached model and teacher resources."""
    model: Any  # ActiveCanViT - type not exported
    teacher: DINOv3Backbone | None
    scene_norm: PositionAwareNorm | None
    cls_norm: PositionAwareNorm | None
    probe: DINOv3LinearClassificationHead | None
    patch_size: int
    backbone: str
    step: int


@dataclass
class Config:
    """Sidebar configuration."""
    ckpt_path: str
    device: str
    scale: float
    glimpse_grid: int
    canvas_grid: int
    l2_norm: bool
    show_full: bool
    show_glimpse: bool
    show_up: bool
    normalize: bool
    pc_offset: int
    projected_basis: str  # "teacher" | "own"
    up_basis: str  # "teacher" | "own"
    rerun_teacher: bool


@dataclass
class TeacherOut:
    """Teacher features at one resolution."""
    scene: Tensor | None  # [N, D]
    cls: Tensor | None  # [D]
    top5: list[tuple[str, float]]
    ms: float


@dataclass
class StepResult:
    """One model inference step."""
    hidden: np.ndarray  # [N, D]
    projected: np.ndarray  # [N, D]
    glimpse: np.ndarray  # [H, W, 3]
    scene_cos: float | None = None
    cls_cos: float | None = None
    ms: float = 0.0
    top5: list[tuple[str, float]] = field(default_factory=list)
    policy: Viewpoint | None = None


@dataclass
class ImageCtx:
    """Per-image context: everything derived from the uploaded image."""
    image: Tensor  # [1, 3, H, W] normalized
    pil: Image.Image  # display size
    H: int
    W: int
    teacher_full: TeacherOut
    teacher_glimpse: TeacherOut | None
    pca_full: PCA | None
    pca_glimpse: PCA | None
    teacher_grid: int


@dataclass
class SeqState:
    """Sequence state from session."""
    viewpoints: list[Viewpoint]
    results: list[StepResult]
    latency: dict[str, list[float]]


# --- Loading ---


@st.cache_resource
def load_resources(ckpt_path: str, device_str: str) -> Resources:
    log.info(f"Loading: {ckpt_path} on {device_str}")
    device = torch.device(device_str)
    model = load_model(Path(ckpt_path), device)
    ckpt = load_ckpt(Path(ckpt_path), "cpu")

    teacher: DINOv3Backbone | None = None
    try:
        teacher = create_backbone(ckpt["backbone"], pretrained=True).to(device).eval()
        assert isinstance(teacher, DINOv3Backbone)
    except Exception as e:
        log.warning(f"Teacher failed: {e}")

    return Resources(
        model=model,
        teacher=teacher,
        scene_norm=_load_norm(ckpt, "scene_norm_state", device),
        cls_norm=_load_norm(ckpt, "cls_norm_state", device, grid=1),
        probe=load_probe(ckpt["backbone"], device),
        patch_size=model.backbone.patch_size_px,
        backbone=ckpt.get("backbone", "?"),
        step=int(ckpt.get("step") or -1),
    )


def _load_norm(ckpt: CheckpointData, key: str, device: torch.device, grid: int | None = None) -> PositionAwareNorm | None:
    if (s := ckpt.get(key)) is None:
        return None
    n, d = s["mean"].shape
    norm = PositionAwareNorm(n, d, grid if grid else int(n**0.5))
    norm.load_state_dict(s)
    return norm.eval().to(device)


# --- Inference ---


def get_teacher(res: Resources, image: Tensor, grid: int, *, normalize: bool, interp: bool = False) -> TeacherOut:
    """Get teacher features. If grid != norm grid and interp=True, interpolate norm stats."""
    if res.teacher is None:
        return TeacherOut(None, None, [], 0.0)

    px = grid * res.patch_size
    img = F.interpolate(image, (px, px), mode="bilinear", align_corners=False)

    sync_device(image.device)
    t0 = time.perf_counter()
    feats = res.teacher.forward_norm_features(img)
    sync_device(image.device)
    ms = (time.perf_counter() - t0) * 1000

    scene = feats.patches  # [1, N, D]
    if normalize and res.scene_norm is not None:
        if grid == res.scene_norm.grid_size:
            scene = res.scene_norm(scene)
        elif interp:
            scene = _interp_norm(scene[0], res.scene_norm, grid).unsqueeze(0)

    cls = res.cls_norm(feats.cls.unsqueeze(1)).squeeze(1) if normalize and res.cls_norm else None
    top5 = _top5(feats.cls, res.probe)
    return TeacherOut(scene[0], cls, top5, ms)


def _interp_norm(x: Tensor, norm: PositionAwareNorm, tgt: int) -> Tensor:
    """Apply norm with bilinearly interpolated stats (interpolate variance, not std)."""
    src, D = norm.grid_size, norm.mean.shape[-1]
    dev = x.device
    # [N,D] -> [1,D,H,W] -> interp -> [N,D]
    def reshape_interp(t: Tensor) -> Tensor:
        t2d = t.view(src, src, D).permute(2, 0, 1).unsqueeze(0).to(dev)
        t_up = F.interpolate(t2d, (tgt, tgt), mode="bilinear", align_corners=False)
        return t_up[0].permute(1, 2, 0).reshape(-1, D)
    mean, var = reshape_interp(norm.mean), reshape_interp(norm.var)
    return (x - mean) / (var + norm.eps).sqrt()


def _top5(cls: Tensor, probe: DINOv3LinearClassificationHead | None) -> list[tuple[str, float]]:
    if probe is None:
        return []
    p, c = F.softmax(probe(cls), dim=-1)[0].topk(5)
    return [(LABELS[i], prob) for i, prob in zip(c.tolist(), p.tolist())]


def run_step(
    res: Resources, image: Tensor, canvas: Tensor, cls: Tensor, vp: Viewpoint,
    *, glimpse_px: int, canvas_grid: int, teacher: TeacherOut, teacher_grid: int, l2: bool,
) -> tuple[Tensor, Tensor, StepResult]:
    """Run one model step, compute metrics."""
    sync_device(image.device)
    t0 = time.perf_counter()
    out = res.model.forward_step(image=image, canvas=canvas, cls=cls, viewpoint=vp, glimpse_size_px=glimpse_px)
    sync_device(image.device)
    ms = (time.perf_counter() - t0) * 1000

    spatial = res.model.get_spatial(out.canvas)[0]
    if l2:
        spatial = F.normalize(spatial, p=2, dim=-1)
    scene = res.model.predict_teacher_scene(out.canvas)

    # Cosine sims (downsample model to teacher grid)
    scene_cos = cls_cos = None
    if teacher.scene is not None:
        D = scene.shape[-1]
        s2d = scene.view(1, canvas_grid, canvas_grid, D).permute(0, 3, 1, 2)
        s_down = F.interpolate(s2d, (teacher_grid, teacher_grid), mode="bilinear", align_corners=False)
        scene_cos = F.cosine_similarity(s_down[0].permute(1, 2, 0).reshape(-1, D), teacher.scene, dim=-1).mean().item()
    cls_pred = res.model.predict_teacher_cls(out.cls, out.canvas)
    if teacher.cls is not None:
        cls_cos = F.cosine_similarity(cls_pred, teacher.cls, dim=-1).mean().item()

    top5 = _top5(res.cls_norm.denormalize(cls_pred), res.probe) if res.probe and res.cls_norm else []

    policy = None
    if isinstance(res.model.policy, PolicyHead) and out.vpe is not None:
        pol = res.model.policy(out.vpe)
        policy = Viewpoint(name="policy", centers=pol.position, scales=pol.scale)

    return out.canvas, out.cls, StepResult(
        hidden=spatial.cpu().numpy(), projected=scene[0].cpu().numpy(),
        glimpse=imagenet_denormalize(out.glimpse[0].cpu()).numpy(),
        scene_cos=scene_cos, cls_cos=cls_cos, ms=ms, top5=top5, policy=policy,
    )


# --- Helpers ---


def upscale(arr: np.ndarray, size: int) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST)


def upsample(latents: np.ndarray, src: int, dst: int) -> np.ndarray:
    D = latents.shape[-1]
    t = torch.from_numpy(latents).float().view(src, src, D).permute(2, 0, 1).unsqueeze(0)
    return F.interpolate(t, (dst, dst), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).reshape(-1, D).numpy()


def pca_img(features: np.ndarray, grid: int, pca: PCA, pc_off: int, *, norm: bool) -> Image.Image:
    return upscale(pca_rgb(pca=pca, features=features, H=grid, W=grid, normalize=norm, pc_offset=pc_off), COL_W)


def click_to_vp(cx: float, cy: float, scale: float, H: int, W: int, dev: torch.device, name: str) -> Viewpoint:
    cx_n = float(np.clip((cx / W) * 2 - 1, -(1 - scale), 1 - scale))
    cy_n = float(np.clip((cy / H) * 2 - 1, -(1 - scale), 1 - scale))
    return Viewpoint(name=name, centers=torch.tensor([[cy_n, cx_n]], device=dev), scales=torch.tensor([scale], device=dev))


def draw_boxes(img: Image.Image, vps: list[Viewpoint], H: int, W: int, policy: Viewpoint | None) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    colors = timestep_colors(len(vps)) if vps else []
    for i, vp in enumerate(vps):
        box = vp.to_pixel_box(0, H, W)
        r, g, b = (int(c * 255) for c in colors[i][:3])
        draw.rectangle([box.left, box.top, box.left + box.width, box.top + box.height], outline=(r, g, b, 255), width=2)
        cx, cy = int(box.center_x), int(box.center_y)
        draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(r, g, b, 200))
        draw.text((cx, cy), str(i), fill=(255, 255, 255), anchor="mm")
    if policy:
        box = policy.to_pixel_box(0, H, W)
        draw.rectangle([box.left, box.top, box.left + box.width, box.top + box.height], outline=(255, 200, 0, 200), width=2)
        draw.text((int(box.center_x), int(box.center_y)), "→", fill=(255, 200, 0), anchor="mm")
    return out


# --- UI ---


def sidebar() -> Config:
    with st.sidebar:
        st.title("AVP-ViT")
        ckpt = st.text_input("Checkpoint", value="reference.pt")
        dev = st.selectbox("Device", ["mps", "cuda", "cpu"], index=0)
        assert isinstance(dev, str)

        st.markdown("---")
        defaults = TrainConfig()
        scale = st.slider("Scale", 0.05, 1.0, 0.25, 0.05)
        glimpse = st.slider("Glimpse grid", 2, 16, defaults.glimpse_grid_size, 1)
        canvas = st.slider("Canvas grid", 8, 256, defaults.grid_size, 8)
        l2 = st.checkbox("L2 norm hidden", value=False)

        st.markdown("---")
        st.markdown("**Teacher**")
        show_full = st.checkbox("Full res", value=True)
        show_glimpse = st.checkbox("Glimpse res", value=True)
        show_up = st.checkbox("Upsampled", value=True)
        normalize = st.checkbox("Normalize", value=True, help="Full: exact stats. Glimpse: interpolated.")

        st.markdown("---")
        st.markdown("**PCA**")
        pc_off = st.slider("PC offset", 0, 9, 0, 1)
        proj_basis = st.radio("Projected", ["teacher", "own"], horizontal=True)
        up_basis = st.radio("Upsampled", ["teacher", "own"], horizontal=True)

        st.markdown("---")
        c1, c2 = st.columns(2)
        if c1.button("Clear"):
            for k in ["viewpoints", "results", "canvas", "cls", "last_click", "_cfg"]:
                st.session_state.pop(k, None)
            st.rerun()
        if c2.button("Undo"):
            vps, res = st.session_state.get("viewpoints", []), st.session_state.get("results", [])
            if vps and res:
                vps.pop()
                res.pop()
                st.session_state.last_click = None
                st.rerun()

        c3, c4 = st.columns(2)
        rerun = c3.button("Re-teacher")
        if c4.button("Clear lat"):
            st.session_state.pop("latency", None)
            st.rerun()

    return Config(
        ckpt_path=ckpt, device=dev, scale=scale, glimpse_grid=glimpse, canvas_grid=canvas,
        l2_norm=l2, show_full=show_full, show_glimpse=show_glimpse, show_up=show_up,
        normalize=normalize, pc_offset=pc_off, projected_basis=proj_basis, up_basis=up_basis,
        rerun_teacher=rerun,
    )


def init_seq(res: Resources, cfg: Config, file_key: str) -> SeqState:
    """Get or init sequence state from session."""
    if "latency" not in st.session_state:
        st.session_state.latency = {}

    cfg_key = f"{cfg.ckpt_path}:{cfg.device}:{cfg.canvas_grid}:{cfg.glimpse_grid}:{file_key}"
    if st.session_state.get("_cfg") != cfg_key:
        log.info("Config changed, reset")
        st.session_state._cfg = cfg_key
        st.session_state.viewpoints = []
        st.session_state.results = []
        st.session_state.canvas = res.model.init_canvas(batch_size=1, canvas_grid_size=cfg.canvas_grid)
        st.session_state.cls = res.model.init_cls(batch_size=1)
        st.session_state.last_click = None

    return SeqState(st.session_state.viewpoints, st.session_state.results, st.session_state.latency)


def make_image_ctx(res: Resources, cfg: Config, raw: bytes) -> ImageCtx:
    """Build per-image context: transform, get teacher features, fit PCA."""
    assert res.scene_norm is not None, "Checkpoint must have scene_norm_state"
    device = torch.device(cfg.device)
    img_size = cfg.canvas_grid * res.patch_size
    teacher_grid = res.scene_norm.grid_size

    transform = transforms.Compose([
        transforms.Resize(img_size, transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        imagenet_normalize(),
    ])
    pil_orig = Image.open(io.BytesIO(raw)).convert("RGB")
    img_t = transform(pil_orig)
    assert isinstance(img_t, Tensor)
    image = img_t.unsqueeze(0).to(device)

    img_np = imagenet_denormalize(image[0].cpu()).numpy()
    H, W = img_np.shape[:2]
    pil = Image.fromarray((img_np * 255).astype(np.uint8))

    with torch.no_grad():
        t_full = get_teacher(res, image, teacher_grid, normalize=cfg.normalize)
        t_glimpse = get_teacher(res, image, cfg.glimpse_grid, normalize=cfg.normalize, interp=True) if cfg.show_glimpse else None

    scene_np = t_full.scene.cpu().numpy() if t_full.scene is not None else None
    glimpse_np = t_glimpse.scene.cpu().numpy() if t_glimpse and t_glimpse.scene is not None else None

    return ImageCtx(
        image=image, pil=pil, H=H, W=W,
        teacher_full=t_full, teacher_glimpse=t_glimpse,
        pca_full=fit_pca(scene_np) if scene_np is not None else None,
        pca_glimpse=fit_pca(glimpse_np) if glimpse_np is not None else None,
        teacher_grid=teacher_grid,
    )


def do_step(res: Resources, cfg: Config, ctx: ImageCtx, seq: SeqState, vp: Viewpoint) -> None:
    """Execute model step, update session state, rerun."""
    seq.viewpoints.append(vp)
    glimpse_px = cfg.glimpse_grid * res.patch_size
    with torch.no_grad():
        new_canvas, new_cls, result = run_step(
            res, ctx.image, st.session_state.canvas, st.session_state.cls, vp,
            glimpse_px=glimpse_px, canvas_grid=cfg.canvas_grid,
            teacher=ctx.teacher_full, teacher_grid=ctx.teacher_grid, l2=cfg.l2_norm,
        )
    st.session_state.canvas = new_canvas
    st.session_state.cls = new_cls
    seq.results.append(result)
    seq.latency.setdefault(f"Model {cfg.glimpse_grid}→{cfg.canvas_grid}", []).append(result.ms)
    st.rerun()


# --- Rendering ---


def render_top5(top5: list[tuple[str, float]], label: str) -> None:
    if not top5:
        return
    st.markdown(f"**{label}**")
    cols = st.columns(5)
    for i, (name, prob) in enumerate(top5):
        cols[i].metric(name, f"{100*prob:.1f}%")


def render_plots(seq: SeqState, teacher_grid: int) -> None:
    if not seq.results:
        return
    st.markdown("---")
    c1, c2 = st.columns(2)

    # Cosine similarity
    with c1:
        scene = [r.scene_cos for r in seq.results if r.scene_cos is not None]
        cls = [r.cls_cos for r in seq.results if r.cls_cos is not None]
        if scene or cls:
            fig = go.Figure()
            mx = max(len(scene), len(cls), 1)
            fig.add_trace(go.Scatter(x=[0, mx-1], y=[1, 1], mode="lines", line=dict(dash="dash", color="gray")))
            if scene:
                fig.add_trace(go.Scatter(y=scene, mode="lines+markers", name="Scene"))
            if cls:
                fig.add_trace(go.Scatter(y=cls, mode="lines+markers", name="CLS"))
            fig.update_layout(title=f"Cos sim (vs {teacher_grid}²)", height=250, margin=dict(l=20,r=20,t=40,b=40), yaxis_range=[0,1.05])
            st.plotly_chart(fig, use_container_width=True)

    # Latency
    with c2:
        if seq.latency:
            fig = go.Figure()
            colors = ["rgba(100,100,100,0.6)", "rgba(66,133,244,0.6)", "rgba(244,66,66,0.6)", "rgba(66,244,66,0.6)"]
            for i, (k, v) in enumerate(sorted(seq.latency.items())):
                fig.add_trace(go.Box(y=v, name=f"{k} (n={len(v)})", marker_color=colors[i % 4], boxpoints="all", jitter=0.3))
            fig.update_layout(title="Latency", yaxis_title="ms", height=250, margin=dict(l=20,r=20,t=40,b=40), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


def render_timeline(seq: SeqState, cfg: Config, pca_teacher: PCA | None, teacher_grid: int) -> None:
    if not seq.results:
        return
    st.markdown("---")
    n = min(len(seq.results), 8)
    start = len(seq.results) - n
    st.markdown(f"**Timeline** ({teacher_grid}² basis)")
    cols = st.columns(n)
    for i, col in enumerate(cols):
        t = start + i
        r = seq.results[t]
        with col:
            st.image(Image.fromarray((np.clip(r.glimpse, 0, 1) * 255).astype(np.uint8)).resize((256, 256), Image.Resampling.LANCZOS), width=80)
            pca_h = fit_pca(r.hidden)
            if pca_h:
                st.image(pca_img(r.hidden, cfg.canvas_grid, pca_h, cfg.pc_offset, norm=False), width=80)
            if pca_teacher:
                st.image(pca_img(r.projected, cfg.canvas_grid, pca_teacher, cfg.pc_offset, norm=True), width=80)
            cos = f" {r.scene_cos:.2f}" if r.scene_cos else ""
            st.caption(f"T{t} s={seq.viewpoints[t].scales[0].item():.2f}{cos}")


# --- Main ---


def main() -> None:
    cfg = sidebar()

    # Upload
    up = st.file_uploader("Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    if up:
        st.session_state.file = up.read()
        st.session_state.fname = up.name

    if not Path(cfg.ckpt_path).exists():
        st.error(f"Not found: {cfg.ckpt_path}")
        return
    if "file" not in st.session_state:
        st.info("Upload image")
        return

    res = load_resources(cfg.ckpt_path, cfg.device)
    seq = init_seq(res, cfg, st.session_state.get("fname", ""))
    ctx = make_image_ctx(res, cfg, st.session_state.file)
    device = torch.device(cfg.device)
    tg = ctx.teacher_grid

    # Record teacher latency (Re-teacher adds samples for variability measurement)
    for key, t in [(f"Teacher {tg}²", ctx.teacher_full), (f"Teacher {cfg.glimpse_grid}²", ctx.teacher_glimpse)]:
        if t and t.ms > 0 and (not seq.latency.get(key) or cfg.rerun_teacher):
            seq.latency.setdefault(key, []).append(t.ms)

    # Get numpy arrays for teacher features
    full_np = ctx.teacher_full.scene.cpu().numpy() if ctx.teacher_full.scene is not None else None
    glimpse_np = ctx.teacher_glimpse.scene.cpu().numpy() if ctx.teacher_glimpse and ctx.teacher_glimpse.scene is not None else None

    # --- Columns ---
    # Count: input + hidden + projected + conditionals
    n_cols = 3 + cfg.show_full + cfg.show_glimpse + cfg.show_up + (cfg.show_glimpse and cfg.show_up)
    cols = iter(st.columns(n_cols))

    # Input
    with next(cols):
        st.markdown("**Input** (click)")
        last_pol = seq.results[-1].policy if seq.results else None
        disp = draw_boxes(ctx.pil, seq.viewpoints, ctx.H, ctx.W, last_pol).resize((COL_W, COL_W), Image.Resampling.LANCZOS)
        coords = streamlit_image_coordinates(disp, key="img")
        if coords and (click := (coords["x"], coords["y"])) != st.session_state.get("last_click"):
            st.session_state.last_click = click
            cx, cy = click[0] * ctx.W / COL_W, click[1] * ctx.H / COL_W
            vp = click_to_vp(cx, cy, cfg.scale, ctx.H, ctx.W, device, f"t{len(seq.viewpoints)}")
            do_step(res, cfg, ctx, seq, vp)
        if last_pol and st.button("Policy →"):
            vp = Viewpoint(name=f"pol_t{len(seq.viewpoints)}", centers=last_pol.centers.clone(), scales=last_pol.scales.clone())
            do_step(res, cfg, ctx, seq, vp)

    # Teacher full
    if cfg.show_full and full_np is not None and ctx.pca_full:
        with next(cols):
            st.markdown(f"**Teacher {tg}²**" + ("" if cfg.normalize else " (raw)"))
            st.image(pca_img(full_np, tg, ctx.pca_full, cfg.pc_offset, norm=False), width=COL_W)

    # Teacher glimpse
    if cfg.show_glimpse and glimpse_np is not None and ctx.pca_glimpse:
        with next(cols):
            st.markdown(f"**Teacher {cfg.glimpse_grid}²**" + (" (interp)" if cfg.normalize else " (raw)"))
            st.image(pca_img(glimpse_np, cfg.glimpse_grid, ctx.pca_glimpse, cfg.pc_offset, norm=False), width=COL_W)

    # Hidden
    with next(cols):
        suf = " (L2)" if cfg.l2_norm else ""
        st.markdown(f"**Hidden {cfg.canvas_grid}²**{suf}" + (f" T{len(seq.results)-1}" if seq.results else ""))
        if seq.results:
            h = seq.results[-1].hidden
            pca_h = fit_pca(h)
            if pca_h:
                st.image(pca_img(h, cfg.canvas_grid, pca_h, cfg.pc_offset, norm=False), width=COL_W)

    # Projected
    with next(cols):
        basis_lbl = " (own)" if cfg.projected_basis == "own" else f" ({tg}²)"
        st.markdown(f"**Projected {cfg.canvas_grid}²**{basis_lbl}" + (f" T{len(seq.results)-1}" if seq.results else ""))
        if seq.results:
            p = seq.results[-1].projected
            pca = fit_pca(p) if cfg.projected_basis == "own" else ctx.pca_full
            if pca:
                st.image(pca_img(p, cfg.canvas_grid, pca, cfg.pc_offset, norm=(cfg.projected_basis == "teacher")), width=COL_W)
                if seq.results[-1].scene_cos is not None:
                    st.caption(f"cos vs {tg}² = {seq.results[-1].scene_cos:.4f}")

    # Teacher↑ full
    if cfg.show_up and full_np is not None:
        with next(cols):
            basis_lbl = " (own)" if cfg.up_basis == "own" else f" ({tg}²)"
            st.markdown(f"**Teacher↑ {tg}²→{cfg.canvas_grid}²**{basis_lbl}")
            up = upsample(full_np, tg, cfg.canvas_grid)
            pca = fit_pca(up) if cfg.up_basis == "own" else ctx.pca_full
            if pca:
                st.image(pca_img(up, cfg.canvas_grid, pca, cfg.pc_offset, norm=(cfg.up_basis == "teacher")), width=COL_W)

    # Teacher↑ glimpse
    if cfg.show_glimpse and cfg.show_up and glimpse_np is not None:
        with next(cols):
            basis_lbl = " (own)" if cfg.up_basis == "own" else f" ({cfg.glimpse_grid}²)"
            st.markdown(f"**Teacher↑ {cfg.glimpse_grid}²→{cfg.canvas_grid}²**{basis_lbl}")
            up = upsample(glimpse_np, cfg.glimpse_grid, cfg.canvas_grid)
            pca = fit_pca(up) if cfg.up_basis == "own" else ctx.pca_glimpse
            if pca:
                st.image(pca_img(up, cfg.canvas_grid, pca, cfg.pc_offset, norm=(cfg.up_basis == "teacher")), width=COL_W)

    # --- Bottom ---
    st.markdown("---")
    render_top5(seq.results[-1].top5 if seq.results else [], "Model")
    render_top5(ctx.teacher_full.top5, f"Teacher {tg}²")
    if ctx.teacher_glimpse:
        render_top5(ctx.teacher_glimpse.top5, f"Teacher {cfg.glimpse_grid}²")

    render_plots(seq, tg)
    render_timeline(seq, cfg, ctx.pca_full, tg)

    # Debug (collapsed)
    with st.expander(f"Debug ({len(seq.viewpoints)} vps)"):
        lat = ", ".join(f"{k}:{np.mean(v):.1f}ms" for k, v in sorted(seq.latency.items()))
        st.code(f"backbone:{res.backbone} step:{res.step} patch:{res.patch_size}px\nteacher:{'✓' if res.teacher else '✗'} norm:{res.scene_norm.grid_size if res.scene_norm else '✗'}² probe:{'✓' if res.probe else '✗'} policy:{'✓' if res.model.policy else '✗'}\n{lat or 'no latency'}")


if __name__ == "__main__":
    main()

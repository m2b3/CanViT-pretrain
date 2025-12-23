#!/usr/bin/env python3
"""Clean Streamlit app for AVP-ViT inference with click-based viewpoints.

User flow:
1. Upload image → state initialized
2. Click image → add viewpoint, run inference
3. Change scale → affects next viewpoint only
4. Change canvas/glimpse/checkpoint → resets state
5. Reset viewpoints → clears viewpoints but keeps image
"""

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
from dinov3.models.vision_transformer import DinoVisionTransformer
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
from torch import Tensor
from torchvision import transforms
from torchvision.models.resnet import ResNet50_Weights

from avp_vit.checkpoint import _get_backbone_factory
from avp_vit.checkpoint import load as load_ckpt
from avp_vit.checkpoint import load_model
from avp_vit.train.data import imagenet_normalize
from avp_vit.train.norm import PositionAwareNorm
from avp_vit.train.probe import load_probe
from avp_vit.train.viewpoint import Viewpoint
from avp_vit.train.viz import fit_pca, imagenet_denormalize, pca_rgb, timestep_colors
from dinov3_probes import DINOv3LinearClassificationHead
from ytch.device import sync_device

IMAGENET_LABELS = ResNet50_Weights.DEFAULT.meta["categories"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

st.set_page_config(page_title="AVP-ViT", layout="wide")


# --- Data ---


@dataclass
class StepResult:
    hidden: np.ndarray
    projected: np.ndarray
    glimpse: np.ndarray
    scene_cos: float | None = None
    cls_cos: float | None = None
    step_ms: float = 0.0
    top5: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class Resources:
    model: Any  # ActiveCanViT - using Any to avoid complex type deps
    teacher: DINOv3Backbone | None
    scene_norm: PositionAwareNorm | None
    cls_norm: PositionAwareNorm | None
    probe: DINOv3LinearClassificationHead | None
    ckpt: Any
    patch_size: int
    backbone: str
    ckpt_step: int


# --- Loading ---


@st.cache_resource
def load_resources(ckpt_path: str, device_name: str) -> Resources:
    log.info(f"Loading resources: ckpt={ckpt_path}, device={device_name}")
    device = torch.device(device_name)
    model = load_model(Path(ckpt_path), device)
    ckpt = load_ckpt(Path(ckpt_path), "cpu")

    teacher = None
    try:
        factory = _get_backbone_factory(ckpt["backbone"])
        raw = factory(pretrained=True).to(device).eval()
        assert isinstance(raw, DinoVisionTransformer)
        teacher = DINOv3Backbone(raw)
        log.info(f"Teacher loaded: {ckpt['backbone']}")
    except Exception as e:
        log.warning(f"Failed to load teacher: {e}")

    scene_norm = None
    if (s := ckpt.get("scene_norm_state")) is not None:
        n, d = s["mean"].shape
        scene_norm = PositionAwareNorm(n, d, int(n**0.5))
        scene_norm.load_state_dict(s)
        scene_norm.eval().to(device)
        log.info(f"Scene norm loaded: {scene_norm.grid_size}² positions, dim={d}")

    cls_norm = None
    if (s := ckpt.get("cls_norm_state")) is not None:
        n, d = s["mean"].shape
        cls_norm = PositionAwareNorm(n, d, 1)
        cls_norm.load_state_dict(s)
        cls_norm.eval().to(device)
        log.info(f"CLS norm loaded: dim={d}")

    probe = load_probe(ckpt["backbone"], device)
    log.info(f"Probe: {'loaded' if probe else 'not available'}")

    return Resources(
        model=model,
        teacher=teacher,
        scene_norm=scene_norm,
        cls_norm=cls_norm,
        probe=probe,
        ckpt=ckpt,
        patch_size=model.backbone.patch_size_px,
        backbone=ckpt.get("backbone", "?"),
        ckpt_step=int(ckpt.get("step") or -1),
    )


# --- Core ---


@dataclass
class TeacherResult:
    scene: Tensor | None
    cls: Tensor | None
    top5: list[tuple[str, float]]
    ms: float


def get_teacher_target(res: Resources, image: Tensor, teacher_grid: int) -> TeacherResult:
    if res.teacher is None:
        return TeacherResult(None, None, [], 0.0)

    px = teacher_grid * res.patch_size
    img = F.interpolate(image, (px, px), mode="bilinear", align_corners=False)

    sync_device(image.device)
    t0 = time.perf_counter()
    feats = res.teacher.forward_norm_features(img)
    sync_device(image.device)
    ms = (time.perf_counter() - t0) * 1000

    scene = feats.patches
    if res.scene_norm is not None:
        scene = res.scene_norm(scene)

    cls_target = None
    if res.cls_norm is not None:
        cls_target = res.cls_norm(feats.cls.unsqueeze(1)).squeeze(1)

    # Teacher top-5 predictions
    top5: list[tuple[str, float]] = []
    if res.probe is not None:
        logits = res.probe(feats.cls)
        probs = F.softmax(logits, dim=-1)
        p, c = probs[0].topk(5)
        top5 = [(IMAGENET_LABELS[i], prob) for i, prob in zip(c.tolist(), p.tolist())]

    return TeacherResult(scene[0] if scene is not None else None, cls_target, top5, ms)


def run_step(
    res: Resources,
    image: Tensor,
    canvas: Tensor,
    cls: Tensor,
    vp: Viewpoint,
    glimpse_px: int,
    canvas_grid: int,
    teacher_target: Tensor | None,
    cls_target: Tensor | None,
    teacher_grid: int,
    l2_norm: bool,
) -> tuple[Tensor, Tensor, StepResult]:
    # Shape assertions
    assert image.shape[0] == 1, f"batch size must be 1, got {image.shape[0]}"
    assert canvas.shape[0] == 1, f"canvas batch size must be 1, got {canvas.shape[0]}"
    assert cls.shape[0] == 1, f"cls batch size must be 1, got {cls.shape[0]}"

    sync_device(image.device)
    t0 = time.perf_counter()
    out = res.model.forward_step(
        image=image, canvas=canvas, cls=cls, viewpoint=vp, glimpse_size_px=glimpse_px
    )
    sync_device(image.device)
    step_ms = (time.perf_counter() - t0) * 1000

    spatial = res.model.get_spatial(out.canvas)[0]
    assert spatial.shape[0] == canvas_grid * canvas_grid, f"spatial tokens {spatial.shape[0]} != {canvas_grid}²"

    if l2_norm:
        spatial = F.normalize(spatial, p=2, dim=-1)

    scene = res.model.predict_teacher_scene(out.canvas)
    assert scene.shape[1] == canvas_grid * canvas_grid, f"scene tokens {scene.shape[1]} != {canvas_grid}²"

    # Scene cosine similarity
    scene_cos = None
    if teacher_target is not None:
        D = scene.shape[-1]
        scene_2d = scene.view(1, canvas_grid, canvas_grid, D).permute(0, 3, 1, 2)
        scene_down = F.interpolate(scene_2d, (teacher_grid, teacher_grid), mode="bilinear")
        scene_flat = scene_down[0].permute(1, 2, 0).reshape(-1, D)
        scene_cos = F.cosine_similarity(scene_flat, teacher_target, dim=-1).mean().item()

    # CLS cosine similarity
    cls_cos = None
    cls_pred = res.model.predict_teacher_cls(out.cls, out.canvas)
    if cls_target is not None:
        cls_cos = F.cosine_similarity(cls_pred, cls_target, dim=-1).mean().item()

    # Top-5 classification
    top5 = []
    if res.probe is not None and res.cls_norm is not None:
        logits = res.probe(res.cls_norm.denormalize(cls_pred))
        probs = F.softmax(logits, dim=-1)
        p, c = probs[0].topk(5)
        top5 = [(IMAGENET_LABELS[i], prob) for i, prob in zip(c.tolist(), p.tolist())]

    glimpse_np = imagenet_denormalize(out.glimpse[0].cpu()).numpy()
    assert glimpse_np.shape[0] == glimpse_px and glimpse_np.shape[1] == glimpse_px, f"glimpse shape {glimpse_np.shape[:2]} != {glimpse_px}²"

    return out.canvas, out.cls, StepResult(
        hidden=spatial.cpu().numpy(),
        projected=scene[0].cpu().numpy(),
        glimpse=glimpse_np,
        scene_cos=scene_cos,
        cls_cos=cls_cos,
        step_ms=step_ms,
        top5=top5,
    )


# --- Helpers ---


def upscale(arr: np.ndarray, size: int) -> Image.Image:
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    return img.resize((size, size), Image.Resampling.NEAREST)


def to_viewpoint(cx: float, cy: float, scale: float, H: int, W: int, device: torch.device, name: str) -> Viewpoint:
    cx_n = float(np.clip((cx / W) * 2 - 1, -(1 - scale), 1 - scale))
    cy_n = float(np.clip((cy / H) * 2 - 1, -(1 - scale), 1 - scale))
    return Viewpoint(
        name=name,
        centers=torch.tensor([[cy_n, cx_n]], device=device, dtype=torch.float32),
        scales=torch.tensor([scale], device=device, dtype=torch.float32),
    )


def draw_boxes(img: Image.Image, viewpoints: list[Viewpoint], H: int, W: int) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    colors = timestep_colors(len(viewpoints)) if viewpoints else []
    for i, vp in enumerate(viewpoints):
        box = vp.to_pixel_box(0, H, W)
        r, g, b = (int(c * 255) for c in colors[i][:3])
        draw.rectangle([box.left, box.top, box.left + box.width, box.top + box.height], outline=(r, g, b, 255), width=2)
        # Draw timestep number at center
        cx, cy = int(box.center_x), int(box.center_y)
        draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(r, g, b, 200))
        draw.text((cx, cy), str(i), fill=(255, 255, 255, 255), anchor="mm")
    return out


# --- UI ---


def main() -> None:
    # Sidebar
    with st.sidebar:
        st.title("AVP-ViT")
        ckpt_path = st.text_input("Checkpoint", value="reference.pt")
        device_name = st.selectbox("Device", ["mps", "cuda", "cpu"], index=0)
        assert isinstance(device_name, str)

        st.markdown("---")
        scale = st.slider("Viewpoint scale", 0.05, 1.0, 0.25, 0.05)
        glimpse_grid = st.slider("Glimpse grid", 2, 16, 8, 1)
        canvas_grid = st.slider("Canvas grid", 8, 256, 32, 8)
        l2_norm = st.checkbox("L2 normalize hidden", value=False)

        st.markdown("---")
        col_clear, col_undo = st.columns(2)
        with col_clear:
            if st.button("Clear viewpoints", help="Resets canvas and all viewpoints"):
                st.session_state.pop("viewpoints", None)
                st.session_state.pop("results", None)
                st.session_state.pop("canvas", None)
                st.session_state.pop("cls", None)
                st.session_state.pop("last_click", None)
                st.session_state.pop("_config", None)  # Force reinit
                log.info("Cleared all viewpoints")
                st.rerun()
        with col_undo:
            if st.button("Undo last", help="Removes last viewpoint from display (canvas state preserved)"):
                vps = st.session_state.get("viewpoints", [])
                if vps:
                    removed = vps.pop()
                    st.session_state.results.pop()
                    st.session_state.last_click = None
                    log.info(f"Undid viewpoint {removed.name}, {len(vps)} remaining")
                    st.rerun()

        col_teacher, col_latency = st.columns(2)
        with col_teacher:
            rerun_teacher = st.button("Rerun teacher", help="Sample teacher latency again")
        with col_latency:
            if st.button("Clear latency", help="Clear all latency measurements"):
                st.session_state.pop("latency_data", None)
                log.info("Cleared latency data")
                st.rerun()

    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    # Store uploaded file in session state to survive reruns
    if uploaded is not None:
        st.session_state.uploaded_file = uploaded.read()
        st.session_state.uploaded_name = uploaded.name
        uploaded.seek(0)  # Reset for later use

    if not Path(ckpt_path).exists():
        st.error(f"Checkpoint not found: {ckpt_path}")
        return

    if "uploaded_file" not in st.session_state:
        st.info("Upload an image to begin.")
        return

    res = load_resources(ckpt_path, device_name)
    device = torch.device(device_name)
    glimpse_px = glimpse_grid * res.patch_size
    img_size = canvas_grid * res.patch_size
    teacher_grid = res.scene_norm.grid_size if res.scene_norm else 16

    # Persistent latency data (survives config changes)
    if "latency_data" not in st.session_state:
        st.session_state.latency_data = {}  # dict[str, list[float]]

    # Use stored filename for config key (stable across reruns)
    file_key = st.session_state.get("uploaded_name", "none")

    # Config change detection (excludes scale, l2_norm)
    config_key = f"{ckpt_path}:{device_name}:{canvas_grid}:{glimpse_grid}:{file_key}"
    old_config = st.session_state.get("_config")
    n_vps = len(st.session_state.get("viewpoints", []))
    if old_config != config_key:
        log.warning(f"CONFIG RESET! old='{old_config}' new='{config_key}'")
        st.session_state._config = config_key
        st.session_state.viewpoints = []
        st.session_state.results = []
        st.session_state.canvas = res.model.init_canvas(batch_size=1, canvas_grid_size=canvas_grid)
        st.session_state.cls = res.model.init_cls(batch_size=1)
        st.session_state.last_click = None
    else:
        log.info(f"Render: vps={n_vps}")

    # Load image from session state (stable across reruns)
    transform = transforms.Compose([
        transforms.Resize(img_size, transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        imagenet_normalize(),
    ])
    pil = Image.open(io.BytesIO(st.session_state.uploaded_file)).convert("RGB")
    orig_size = pil.size
    img_t = transform(pil)
    assert isinstance(img_t, Tensor)
    image = img_t.unsqueeze(0).to(device)
    img_np = imagenet_denormalize(image[0].cpu()).numpy()
    H, W = img_np.shape[:2]
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    log.debug(f"Image: {orig_size} → {H}x{W} ({img_size}px)")

    # Teacher target
    latency_data: dict[str, list[float]] = st.session_state.latency_data
    teacher_key = f"Teacher {teacher_grid}²"
    model_key = f"Model {glimpse_grid}→{canvas_grid}"

    with torch.no_grad():
        teacher = get_teacher_target(res, image, teacher_grid)
        if teacher.ms > 0:
            if teacher_key not in latency_data:
                latency_data[teacher_key] = []
            # First run for this config or rerun button pressed
            if not latency_data[teacher_key] or rerun_teacher:
                latency_data[teacher_key].append(teacher.ms)

    # Cache expensive computations
    scene_target_np = teacher.scene.cpu().numpy() if teacher.scene is not None else None
    pca_teacher = fit_pca(scene_target_np) if scene_target_np is not None else None

    viewpoints: list[Viewpoint] = st.session_state.viewpoints
    results: list[StepResult] = st.session_state.results
    n = len(viewpoints)

    # Main layout
    c1, c2, c3, c4 = st.columns(4)
    col_w = 384

    with c1:
        st.markdown("**Input** (click to add)")
        display = draw_boxes(img_pil, viewpoints, H, W).resize((col_w, col_w), Image.Resampling.LANCZOS)
        coords = streamlit_image_coordinates(display, key="img")

        if coords is not None:
            click = (coords["x"], coords["y"])
            if click != st.session_state.last_click:
                st.session_state.last_click = click
                cx, cy = click[0] * W / col_w, click[1] * H / col_w
                vp = to_viewpoint(cx, cy, scale, H, W, device, f"t{n}")
                viewpoints.append(vp)

                with torch.no_grad():
                    sync_device(device)
                    new_canvas, new_cls, result = run_step(
                        res, image, st.session_state.canvas, st.session_state.cls,
                        vp, glimpse_px, canvas_grid, teacher.scene, teacher.cls,
                        teacher_grid, l2_norm,
                    )
                    sync_device(device)
                    st.session_state.canvas = new_canvas
                    st.session_state.cls = new_cls
                    results.append(result)
                    # Add to persistent latency data
                    if model_key not in latency_data:
                        latency_data[model_key] = []
                    latency_data[model_key].append(result.step_ms)
                    cos_str = f"{result.scene_cos:.4f}" if result.scene_cos else "?"
                    log.info(f"T{n}: pos=({cx:.1f},{cy:.1f}) scale={scale:.2f} cos={cos_str} time={result.step_ms:.1f}ms")
                st.rerun()

    with c2:
        st.markdown(f"**Teacher {teacher_grid}²**")
        if scene_target_np is not None and pca_teacher is not None:
            st.image(upscale(pca_rgb(pca_teacher, scene_target_np, teacher_grid, teacher_grid), col_w), width=col_w)

    with c3:
        lbl = f"**Hidden {canvas_grid}²**" + (" (L2)" if l2_norm else "") + (f" T{n-1}" if n else "")
        st.markdown(lbl)
        if n > 0:
            h = results[-1].hidden
            st.image(upscale(pca_rgb(fit_pca(h), h, canvas_grid, canvas_grid), col_w), width=col_w)

    with c4:
        st.markdown(f"**Projected {canvas_grid}²**" + (f" T{n-1}" if n else ""))
        if n > 0 and pca_teacher is not None:
            p = results[-1].projected
            st.image(upscale(pca_rgb(pca_teacher, p, canvas_grid, canvas_grid, normalize=True), col_w), width=col_w)
            if results[-1].scene_cos is not None:
                st.caption(f"cos = {results[-1].scene_cos:.4f}")

    # Top-5 predictions (Model vs Teacher)
    has_model_preds = n > 0 and results[-1].top5
    has_teacher_preds = bool(teacher.top5)
    if has_model_preds or has_teacher_preds:
        st.markdown("---")
        if has_model_preds:
            st.markdown("**Model** (AVP-ViT)")
            cols = st.columns(5)
            for i, (label, prob) in enumerate(results[-1].top5):
                with cols[i]:
                    st.metric(label, f"{100*prob:.1f}%")
        if has_teacher_preds:
            st.markdown("**Teacher** (DINOv3)")
            cols = st.columns(5)
            for i, (label, prob) in enumerate(teacher.top5):
                with cols[i]:
                    st.metric(label, f"{100*prob:.1f}%")

    # Metrics plots
    if n > 0:
        st.markdown("---")
        scene_sims = [r.scene_cos for r in results if r.scene_cos is not None]
        cls_sims = [r.cls_cos for r in results if r.cls_cos is not None]

        col_sim, col_lat = st.columns(2)

        with col_sim:
            if scene_sims or cls_sims:
                fig = go.Figure()
                # Reference line at cos=1 (perfect alignment)
                max_t = max(len(scene_sims) if scene_sims else 1, len(cls_sims) if cls_sims else 1)
                fig.add_trace(go.Scatter(x=[0, max_t - 1], y=[1.0, 1.0], mode="lines", name="Target", line={"dash": "dash", "color": "gray", "width": 1}))
                if scene_sims:
                    fig.add_trace(go.Scatter(x=list(range(len(scene_sims))), y=scene_sims, mode="lines+markers", name="Scene"))
                if cls_sims:
                    fig.add_trace(go.Scatter(x=list(range(len(cls_sims))), y=cls_sims, mode="lines+markers", name="CLS"))
                fig.update_layout(title="Cosine Similarity", xaxis_title="T", yaxis_title="cos", height=250, margin=dict(l=20, r=20, t=40, b=40), yaxis_range=[0, 1.05])
                st.plotly_chart(fig, width="stretch")

        with col_lat:
            if latency_data:
                fig = go.Figure()
                colors = ["rgba(100,100,100,0.6)", "rgba(66,133,244,0.6)", "rgba(244,66,66,0.6)",
                          "rgba(66,244,66,0.6)", "rgba(244,166,66,0.6)", "rgba(166,66,244,0.6)"]
                for i, (label, times) in enumerate(sorted(latency_data.items())):
                    fig.add_trace(go.Box(
                        y=times, name=f"{label} (n={len(times)})",
                        marker_color=colors[i % len(colors)], boxpoints="all", jitter=0.3
                    ))
                fig.update_layout(
                    title="Latency Distribution",
                    yaxis_title="ms", height=250, margin=dict(l=20, r=20, t=40, b=40), showlegend=False
                )
                st.plotly_chart(fig, width="stretch")

    # Timeline (glimpse → hidden → projected)
    n_results = len(results)
    if n_results > 0:
        st.markdown("---")
        n_show = min(n_results, 8)
        start_t = n_results - n_show  # Show last N
        if n_results > 8:
            st.markdown(f"**Timeline** (showing T{start_t}-T{n_results - 1} of {n_results})")
        else:
            st.markdown("**Timeline** (glimpse → hidden → projected)")
        cols = st.columns(n_show)
        for i in range(n_show):
            t = start_t + i
            with cols[i]:
                # Glimpse (upscale for better expand view)
                glimpse_pil = Image.fromarray((np.clip(results[t].glimpse, 0, 1) * 255).astype(np.uint8))
                glimpse_big = glimpse_pil.resize((256, 256), Image.Resampling.LANCZOS)
                st.image(glimpse_big, width=80)
                # Hidden (own PCA) - upscale to 256 for expand, display at 80
                h = results[t].hidden
                pca_h = fit_pca(h)
                st.image(upscale(pca_rgb(pca_h, h, canvas_grid, canvas_grid), 256), width=80)
                # Projected (teacher PCA)
                if pca_teacher is not None:
                    st.image(upscale(pca_rgb(pca_teacher, results[t].projected, canvas_grid, canvas_grid, normalize=True), 256), width=80)
                # Caption with scale and cos
                vp_scale = viewpoints[t].scales[0].item() if t < len(viewpoints) else 0
                cos = results[t].scene_cos
                caption = f"T{t} s={vp_scale:.2f}"
                if cos is not None:
                    caption += f" cos={cos:.2f}"
                st.caption(caption)

    # Debug
    st.markdown("---")
    with st.expander(f"Debug ({n} viewpoints)"):
        latency_summary = ", ".join(f"{k}: {np.mean(v):.1f}ms (n={len(v)})" for k, v in sorted(latency_data.items()))
        st.code(f"""backbone: {res.backbone}, step: {res.ckpt_step}
patch: {res.patch_size}px, img: {img_size}px, glimpse: {glimpse_px}px
teacher: {'✓' if res.teacher else '✗'}, scene_norm: {res.scene_norm.grid_size if res.scene_norm else '✗'}², probe: {'✓' if res.probe else '✗'}
canvas: {tuple(st.session_state.canvas.shape)}, cls: {tuple(st.session_state.cls.shape)}
latency: {latency_summary or 'none'}""")

        if viewpoints:
            st.markdown("**Viewpoints**")
            vp_lines = []
            for i, vp in enumerate(viewpoints):
                box = vp.to_pixel_box(0, H, W)
                cy_n, cx_n = vp.centers[0].tolist()  # normalized (y,x) in [-1,1]
                s = vp.scales[0].item()
                scene_cos = results[i].scene_cos if i < len(results) else None
                cls_cos = results[i].cls_cos if i < len(results) else None
                cos_str = f"scene={scene_cos:.3f}" if scene_cos else "scene=?"
                if cls_cos:
                    cos_str += f" cls={cls_cos:.3f}"
                vp_lines.append(f"T{i}: norm=({cy_n:+.2f},{cx_n:+.2f}) scale={s:.2f} → px=({box.center_x:.0f},{box.center_y:.0f}) {cos_str}")
            st.code("\n".join(vp_lines))


if __name__ == "__main__":
    main()

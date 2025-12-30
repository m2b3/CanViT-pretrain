"""Main Streamlit application. UI only - no torch code here."""

import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from avp_vit.train.viz import fit_pca

from inference_app.config import Config, render_sidebar
from inference_app.gpu_worker import get_worker
from inference_app.rendering import (
    COL_W,
    click_to_viewpoint,
    cos_sim_heatmap,
    draw_token_box,
    draw_viewpoint_boxes,
    glimpse_to_pil,
    pca_img,
    pos_to_token,
    sim_to_img,
    upsample_features,
)
from inference_app.types import ImageContext, SequenceState, Viewpoint

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
st.set_page_config(page_title="AVP-ViT", layout="wide")


def get_state() -> SequenceState:
    """Get or create sequence state from session."""
    if "seq" not in st.session_state:
        st.session_state.seq = SequenceState(viewpoints=[], results=[])
    return st.session_state.seq


def clear_state() -> None:
    """Clear all session state."""
    keys = ["seq", "image_ctx", "image_bytes", "last_click", "sim_positions", "_config_key"]
    keys += [k for k in list(st.session_state.keys()) if k.startswith("_sim_seen_")]
    for k in keys:
        st.session_state.pop(k, None)


def render_top5(top5: list[tuple[str, float]], label: str) -> None:
    """Render top-5 predictions."""
    if not top5:
        return
    st.markdown(f"**{label}**")
    cols = st.columns(5)
    for i, (name, prob) in enumerate(top5):
        cols[i].metric(name, f"{100*prob:.1f}%")


def render_plots(seq: SequenceState, latency: dict[str, list[float]]) -> None:
    """Render cosine similarity and latency plots."""
    if not seq.results:
        return

    st.markdown("---")
    c1, c2 = st.columns(2)

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
            fig.update_layout(title="Cos sim", height=250, margin=dict(l=20,r=20,t=40,b=40), yaxis_range=[0,1.05])
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if latency:
            fig = go.Figure()
            colors = ["rgba(100,100,100,0.6)", "rgba(66,133,244,0.6)", "rgba(244,66,66,0.6)"]
            for i, (k, v) in enumerate(sorted(latency.items())):
                fig.add_trace(go.Box(y=v, name=f"{k} (n={len(v)})", marker_color=colors[i % 3], boxpoints="all", jitter=0.3))
            fig.update_layout(title="Latency", yaxis_title="ms", height=250, margin=dict(l=20,r=20,t=40,b=40), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


def render_timeline(seq: SequenceState, cfg: Config, pca_teacher: object | None) -> None:
    """Render timeline of glimpses."""
    if not seq.results:
        return

    st.markdown("---")
    n = min(len(seq.results), 8)
    start = len(seq.results) - n
    st.markdown("**Timeline**")
    cols = st.columns(n)

    for i, col in enumerate(cols):
        t = start + i
        r = seq.results[t]
        vp = seq.viewpoints[t]
        with col:
            st.image(glimpse_to_pil(r.glimpse, 256), width=80)
            pca_h = fit_pca(r.hidden)
            if pca_h:
                st.image(pca_img(r.hidden, cfg.canvas_grid, pca_h, cfg.pc_offset, norm=False), width=80)
            cos = f" {r.scene_cos:.2f}" if r.scene_cos else ""
            st.caption(f"T{t} s={vp.scale:.2f}{cos}")


def main() -> None:
    """Main app entry point."""
    cfg, clear_requested, undo_requested = render_sidebar()

    # Handle clear/undo before anything else
    if clear_requested:
        clear_state()
        get_worker().reset_canvas(cfg.canvas_grid)
        st.rerun()

    if undo_requested:
        seq = get_state()
        if seq.viewpoints and seq.results:
            seq.viewpoints.pop()
            seq.results.pop()
            st.session_state.pop("last_click", None)
            st.rerun()

    # File upload
    up = st.file_uploader("Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    if up:
        st.session_state.image_bytes = up.read()
        st.session_state.image_name = up.name

    if not Path(cfg.ckpt_path).exists():
        st.error(f"Not found: {cfg.ckpt_path}")
        return

    if "image_bytes" not in st.session_state:
        st.info("Upload image")
        return

    # Initialize GPU worker
    worker = get_worker()
    backbone, step_count = worker.load(cfg.ckpt_path, cfg.device)

    # Check if config changed - need to reset image context
    config_key = f"{cfg.key}:{st.session_state.get('image_name', '')}"
    if st.session_state.get("_config_key") != config_key:
        # Config changed - reload image
        ctx = worker.set_image(
            st.session_state.image_bytes,
            cfg.canvas_grid, cfg.glimpse_grid, cfg.normalize
        )
        st.session_state.image_ctx = ctx
        st.session_state._config_key = config_key
        # Reset sequence
        st.session_state.seq = SequenceState(viewpoints=[], results=[])
        st.session_state.pop("last_click", None)

    ctx: ImageContext = st.session_state.image_ctx
    seq = get_state()
    tg = ctx.teacher_full.grid

    # Latency tracking
    if "latency" not in st.session_state:
        st.session_state.latency = {}
    latency = st.session_state.latency

    # Record teacher latency
    for key, t in [(f"Teacher {tg}²", ctx.teacher_full), (f"Teacher {cfg.glimpse_grid}²", ctx.teacher_glimpse)]:
        if t and t.ms > 0 and key not in latency:
            latency[key] = [t.ms]

    # Feature arrays
    full_np = ctx.teacher_full.scene
    glimpse_np = ctx.teacher_glimpse.scene if ctx.teacher_glimpse else None

    # --- Columns ---
    n_cols = 3 + cfg.show_full + cfg.show_glimpse + cfg.show_up + (cfg.show_glimpse and cfg.show_up)
    cols = iter(st.columns(n_cols))

    # For similarity visualization
    feat_data: list[tuple[str, np.ndarray | None, int]] = []  # (label, features, grid)
    new_sim_pos: tuple[float, float] | None = None

    def check_click(key: str, features: np.ndarray | None, grid: int) -> None:
        nonlocal new_sim_pos
        coords = streamlit_image_coordinates(st.session_state[f"_pca_img_{key}"], key=key)
        if coords:
            pos = (coords["x"] / COL_W, coords["y"] / COL_W)
            seen_key = f"_sim_seen_{key}"
            if pos != st.session_state.get(seen_key):
                st.session_state[seen_key] = pos
                if pos not in st.session_state.get("sim_positions", []):
                    new_sim_pos = pos
        feat_data.append((key, features, grid))

    # Input column
    with next(cols):
        st.markdown("**Input** (click)")
        last_result = seq.results[-1] if seq.results else None
        policy = None
        if last_result and last_result.policy_center and last_result.policy_scale:
            policy = (*last_result.policy_center, last_result.policy_scale)
        disp = draw_viewpoint_boxes(ctx.pil, seq.viewpoints, ctx.H, ctx.W, policy)
        disp = disp.resize((COL_W, COL_W), Image.Resampling.LANCZOS)

        coords = streamlit_image_coordinates(disp, key="img")
        if coords and (click := (coords["x"], coords["y"])) != st.session_state.get("last_click"):
            st.session_state.last_click = click
            cx, cy = click[0] * ctx.W / COL_W, click[1] * ctx.H / COL_W
            vp = click_to_viewpoint(cx, cy, cfg.scale, ctx.H, ctx.W, f"t{len(seq.viewpoints)}")

            result = worker.step(vp, cfg.glimpse_grid, cfg.canvas_grid, cfg.l2_norm)
            seq.viewpoints.append(vp)
            seq.results.append(result)
            key = f"Model {cfg.glimpse_grid}→{cfg.canvas_grid}"
            latency.setdefault(key, []).append(result.ms)
            latency.setdefault(f"{key} fwd", []).append(result.ms_forward)
            latency.setdefault(f"{key} post", []).append(result.ms_post)
            st.rerun()

        if policy and st.button("Policy →"):
            vp = Viewpoint(cy=policy[0], cx=policy[1], scale=policy[2], name=f"pol_t{len(seq.viewpoints)}")
            result = worker.step(vp, cfg.glimpse_grid, cfg.canvas_grid, cfg.l2_norm)
            seq.viewpoints.append(vp)
            seq.results.append(result)
            key = f"Model {cfg.glimpse_grid}→{cfg.canvas_grid}"
            latency.setdefault(key, []).append(result.ms)
            latency.setdefault(f"{key} fwd", []).append(result.ms_forward)
            latency.setdefault(f"{key} post", []).append(result.ms_post)
            st.rerun()

    feat_data.append(("Input", None, 0))

    # Teacher full
    if cfg.show_full and full_np is not None and ctx.pca_full:
        with next(cols):
            st.markdown(f"**Teacher {tg}²**")
            img = pca_img(full_np, tg, ctx.pca_full, cfg.pc_offset, norm=False)
            st.session_state["_pca_img_t_full"] = img
            check_click("t_full", full_np, tg)

    # Teacher glimpse
    if cfg.show_glimpse and glimpse_np is not None and ctx.pca_glimpse:
        with next(cols):
            st.markdown(f"**Teacher {cfg.glimpse_grid}²**")
            img = pca_img(glimpse_np, cfg.glimpse_grid, ctx.pca_glimpse, cfg.pc_offset, norm=False)
            st.session_state["_pca_img_t_glimpse"] = img
            check_click("t_glimpse", glimpse_np, cfg.glimpse_grid)

    # Hidden
    with next(cols):
        suf = " (L2)" if cfg.l2_norm else ""
        st.markdown(f"**Hidden {cfg.canvas_grid}²**{suf}")
        h = seq.results[-1].hidden if seq.results else None
        if h is not None:
            pca_h = fit_pca(h)
            if pca_h:
                img = pca_img(h, cfg.canvas_grid, pca_h, cfg.pc_offset, norm=False)
                st.session_state["_pca_img_hidden"] = img
                check_click("hidden", h, cfg.canvas_grid)
        feat_data.append((f"H{cfg.canvas_grid}²", h, cfg.canvas_grid))

    # Projected
    with next(cols):
        basis_lbl = " (own)" if cfg.projected_basis == "own" else f" ({tg}²)"
        st.markdown(f"**Projected {cfg.canvas_grid}²**{basis_lbl}")
        p = seq.results[-1].projected if seq.results else None
        if p is not None:
            pca = fit_pca(p) if cfg.projected_basis == "own" else ctx.pca_full
            if pca:
                img = pca_img(p, cfg.canvas_grid, pca, cfg.pc_offset, norm=(cfg.projected_basis == "teacher"))
                st.session_state["_pca_img_projected"] = img
                check_click("projected", p, cfg.canvas_grid)
                if seq.results[-1].scene_cos is not None:
                    st.caption(f"cos vs {tg}² = {seq.results[-1].scene_cos:.4f}")
        feat_data.append((f"P{cfg.canvas_grid}²", p, cfg.canvas_grid))

    # Teacher upsampled
    if cfg.show_up and full_np is not None:
        with next(cols):
            st.markdown(f"**Teacher↑ {tg}²→{cfg.canvas_grid}²**")
            up_full = upsample_features(full_np, tg, cfg.canvas_grid)
            pca = fit_pca(up_full) if cfg.up_basis == "own" else ctx.pca_full
            if pca:
                img = pca_img(up_full, cfg.canvas_grid, pca, cfg.pc_offset, norm=(cfg.up_basis == "teacher"))
                st.session_state["_pca_img_t_up"] = img
                check_click("t_up", up_full, cfg.canvas_grid)

    # Similarity anchors
    if new_sim_pos:
        if "sim_positions" not in st.session_state:
            st.session_state.sim_positions = []
        st.session_state.sim_positions.append(new_sim_pos)

    sim_positions = st.session_state.get("sim_positions", [])
    if sim_positions:
        ctrl_cols = st.columns([1, 1, 4])
        with ctrl_cols[0]:
            if st.button("Clear anchors"):
                st.session_state.sim_positions = []
                for k in list(st.session_state.keys()):
                    if k.startswith("_sim_seen_"):
                        del st.session_state[k]
                st.rerun()
        with ctrl_cols[1]:
            if st.button("Undo anchor"):
                st.session_state.sim_positions.pop()
                st.rerun()
        with ctrl_cols[2]:
            pos_strs = [f"({p[0]:.2f},{p[1]:.2f})" for p in sim_positions]
            st.caption(f"Anchors: {' '.join(pos_strs)}")

        # Render similarity heatmaps
        if sim_positions and feat_data:
            valid = [(label, f, g) for label, f, g in feat_data if f is not None]
            if valid:
                # Compute all similarities
                sims = []
                for label, features, grid in valid:
                    tokens = [pos_to_token(nx, ny, grid) for nx, ny in sim_positions]
                    indices = [t[2] for t in tokens]
                    sim = cos_sim_heatmap(features, indices)
                    sims.append((label, sim, grid, [(t[0], t[1]) for t in tokens]))

                vmin = min(s[1].min() for s in sims)
                vmax = max(s[1].max() for s in sims)

                cols = st.columns(len(sims))
                for col, (label, sim, grid, row_cols) in zip(cols, sims):
                    with col:
                        img = sim_to_img(sim, grid, COL_W, vmin, vmax, cfg.sim_sharpness)
                        for row, c in row_cols:
                            img = draw_token_box(img, row, c, grid)
                        st.image(img, width=COL_W)
                        st.caption(f"{label} [{vmin:.2f},{vmax:.2f}]")

    # Bottom section
    st.markdown("---")
    render_top5(seq.results[-1].top5 if seq.results else [], "Model")
    render_top5(ctx.teacher_full.top5, f"Teacher {tg}²")
    if ctx.teacher_glimpse:
        render_top5(ctx.teacher_glimpse.top5, f"Teacher {cfg.glimpse_grid}²")

    render_plots(seq, latency)
    render_timeline(seq, cfg, ctx.pca_full)

    # Debug
    with st.expander(f"Debug ({len(seq.viewpoints)} vps)"):
        info = worker.get_info()
        lat_str = ", ".join(f"{k}:{np.mean(v):.1f}ms" for k, v in sorted(latency.items()))
        st.code(f"backbone:{info['backbone']} step:{info['step']} patch:{info['patch_size']}px\n"
                f"teacher:{'Y' if info['has_teacher'] else 'N'} norm:{'Y' if info['has_norm'] else 'N'} "
                f"probe:{'Y' if info['has_probe'] else 'N'} policy:{'Y' if info['has_policy'] else 'N'}\n"
                f"{lat_str or 'no latency'}")

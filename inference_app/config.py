"""Configuration and sidebar UI."""

from dataclasses import dataclass

import streamlit as st

from avp_vit.train.config import Config as TrainConfig


@dataclass
class Config:
    """App configuration from sidebar."""
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
    sim_sharpness: float
    projected_basis: str  # "teacher" | "own"
    up_basis: str  # "teacher" | "own"

    @property
    def key(self) -> str:
        """Unique key for this config (for cache invalidation)."""
        return f"{self.ckpt_path}:{self.device}:{self.canvas_grid}:{self.glimpse_grid}"


def render_sidebar() -> tuple[Config, bool, bool]:
    """Render sidebar and return (config, clear_requested, undo_requested)."""
    clear_requested = False
    undo_requested = False

    with st.sidebar:
        st.title("AVP-ViT")
        ckpt = st.text_input("Checkpoint", value="reference.pt")
        dev = st.selectbox("Device", ["mps", "cuda", "cpu"], index=0)
        assert isinstance(dev, str)

        st.markdown("---")
        defaults = TrainConfig()
        scale = st.slider("Scale", 0.05, 1.0, 1.0, 0.05)
        glimpse = st.slider("Glimpse grid", 2, 16, defaults.glimpse_grid_size, 1)
        canvas = st.slider("Canvas grid", 8, 256, defaults.grid_size, 8)
        l2 = st.checkbox("L2 norm hidden", value=False)

        st.markdown("---")
        st.markdown("**Teacher**")
        show_full = st.checkbox("Full res", value=True)
        show_glimpse = st.checkbox("Glimpse res", value=True)
        show_up = st.checkbox("Upsampled", value=True)
        normalize = st.checkbox("Normalize", value=True)

        st.markdown("---")
        st.markdown("**PCA / Similarity**")
        pc_off = st.slider("PC offset", 0, 9, 0, 1)
        sim_sharpness = st.slider("Sim sharpness", 0.1, 5.0, 1.0, 0.1)
        proj_basis = st.radio("Projected", ["teacher", "own"], horizontal=True)
        up_basis = st.radio("Upsampled", ["teacher", "own"], horizontal=True)

        st.markdown("---")
        c1, c2 = st.columns(2)
        if c1.button("Clear"):
            clear_requested = True
        if c2.button("Undo"):
            undo_requested = True

    cfg = Config(
        ckpt_path=ckpt, device=dev, scale=scale, glimpse_grid=glimpse, canvas_grid=canvas,
        l2_norm=l2, show_full=show_full, show_glimpse=show_glimpse, show_up=show_up,
        normalize=normalize, pc_offset=pc_off, sim_sharpness=sim_sharpness,
        projected_basis=proj_basis, up_basis=up_basis,
    )
    return cfg, clear_requested, undo_requested

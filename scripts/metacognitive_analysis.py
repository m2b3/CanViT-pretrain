"""Metacognitive accuracy analysis: does the model update where it helps?

Key findings:
- t=0 NEVER hurts (0% negative) - any update from empty canvas helps
- Later timesteps: % negative increases (53% by t=7)
- Efficiency (improvement/delta) drops from ~30 to ~0.4
- Fractional improvement: 60% error removed at t=0, <1% by t=7
- Strong correlation (r~0.8) between update magnitude and improvement
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

from avp_vit.checkpoint import load as load_ckpt, load_model
from avp_vit.train.norm import PositionAwareNorm
from avp_vit.train.viewpoint import Viewpoint
from canvit.viewpoint import sample_at_viewpoint
from canvit import create_backbone


def load_everything(ckpt_path: Path, device: str = "mps"):
    """Load model, teacher, scene_norm from checkpoint."""
    ckpt = load_ckpt(ckpt_path, "cpu")
    model = load_model(ckpt_path, device).eval()
    teacher = create_backbone(ckpt["backbone"], pretrained=True).to(device).eval()

    scene_norm_state = ckpt["scene_norm_state"]
    n, d = scene_norm_state["mean"].shape
    grid_size = int(n**0.5)
    scene_norm = PositionAwareNorm(n_tokens=n, embed_dim=d, grid_size=grid_size)
    scene_norm.load_state_dict(scene_norm_state)
    scene_norm = scene_norm.eval().to(device)

    return model, teacher, scene_norm, grid_size


def get_teacher_features(img_path: str, teacher, scene_norm, grid_size: int, device: str):
    """Load image and get normalized teacher features."""
    img_size = grid_size * 16
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(img_size, transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = teacher.forward_norm_features(img_t)
        teacher_norm = scene_norm(feats.patches)

    return img_t, teacher_norm


def collect_trajectory_data(
    model, img_t, teacher_norm, grid_size: int, device: str,
    n_timesteps: int = 8, n_trials: int = 15, seed: int = 42
):
    """Run trajectories and collect per-token metrics."""
    glimpse_px = 8 * 16
    data = []

    np.random.seed(seed)
    with torch.no_grad():
        for trial in range(n_trials):
            state = model.init_state(batch_size=1, canvas_grid_size=grid_size)

            for t in range(n_timesteps):
                vp = Viewpoint.random(
                    batch_size=1, device=torch.device(device),
                    min_scale=0.2, max_scale=0.5
                )

                recon_before = model.predict_teacher_scene(state.canvas)
                error_before = (recon_before - teacher_norm).pow(2).sum(dim=-1)

                glimpse = sample_at_viewpoint(spatial=img_t, viewpoint=vp, glimpse_size_px=glimpse_px)
                out = model.forward(glimpse=glimpse, state=state, viewpoint=vp)
                state = out.state

                recon_after = model.predict_teacher_scene(state.canvas)
                error_after = (recon_after - teacher_norm).pow(2).sum(dim=-1)

                # Per-token metrics
                delta = (recon_after - recon_before).pow(2).sum(dim=-1).sqrt()[0].cpu().numpy()
                change = (error_before - error_after)[0].cpu().numpy()
                err_before = error_before[0].cpu().numpy()

                for tok in range(1024):
                    data.append({
                        "t": t,
                        "delta": delta[tok],          # reconstruction change magnitude
                        "change": change[tok],        # error improvement (positive = good)
                        "err_before": err_before[tok],
                    })

    return data


def compute_metrics(data: list[dict]):
    """Compute derived metrics from raw data."""
    ts = np.array([d["t"] for d in data])
    deltas = np.array([d["delta"] for d in data])
    changes = np.array([d["change"] for d in data])
    err_befores = np.array([d["err_before"] for d in data])

    # Derived metrics
    efficiency = changes / (deltas + 1e-6)  # improvement per unit delta
    frac_improvement = changes / (err_befores + 1e-6)  # fractional error reduction

    return ts, deltas, changes, efficiency, frac_improvement


def summarize_by_timestep(ts, changes, efficiency, frac_improvement):
    """Print summary stats by timestep."""
    print("Summary by timestep:")
    print(f"{'t':>2} | {'%neg':>6} | {'eff':>6} | {'frac':>6}")
    print("-" * 30)
    for t in sorted(set(ts)):
        mask = ts == t
        neg_frac = (changes[mask] < 0).mean()
        mean_eff = efficiency[mask].mean()
        mean_frac = frac_improvement[mask].mean()
        print(f"{t:>2} | {neg_frac*100:>5.1f}% | {mean_eff:>6.1f} | {mean_frac:>6.3f}")


def plot_analysis(ts, deltas, changes, efficiency, frac_improvement, save_path: Path):
    """Create summary visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Scatter colored by timestep
    ax = axes[0, 0]
    scatter = ax.scatter(deltas, changes, c=ts, cmap="viridis", alpha=0.2, s=3)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Recon delta (L2)")
    ax.set_ylabel("Error change (+ = improvement)")
    ax.set_title(f"Metacognitive accuracy (r={np.corrcoef(deltas, changes)[0,1]:.3f})")
    plt.colorbar(scatter, ax=ax, label="timestep")

    # 2. Log-log (positive only)
    ax = axes[0, 1]
    pos = (deltas > 0.1) & (changes > 0.1)
    scatter = ax.scatter(deltas[pos], changes[pos], c=ts[pos], cmap="viridis", alpha=0.2, s=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Recon delta (log)")
    ax.set_ylabel("Error change (log)")
    ax.set_title("Log-log (positive only)")
    plt.colorbar(scatter, ax=ax, label="timestep")

    # 3. Efficiency by timestep
    ax = axes[1, 0]
    for t in [0, 2, 5, 7]:
        mask = ts == t
        eff_t = efficiency[mask]
        eff_t = eff_t[(eff_t > -50) & (eff_t < 100)]
        ax.hist(eff_t, bins=50, alpha=0.5, density=True, label=f"t={t}")
    ax.axvline(0, color="red", linestyle="--")
    ax.legend()
    ax.set_xlabel("Efficiency (change/delta)")
    ax.set_title("Update efficiency by timestep")

    # 4. % negative by timestep
    ax = axes[1, 1]
    unique_ts = sorted(set(ts))
    neg_fracs = [(changes[ts == t] < 0).mean() * 100 for t in unique_ts]
    ax.bar(unique_ts, neg_fracs, color="steelblue")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("% negative updates")
    ax.set_title("Fraction of harmful updates")
    ax.axhline(50, color="red", linestyle="--", alpha=0.5, label="coin flip")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    device = "mps"
    ckpt_path = Path("train-6793137-step-234624.pt")
    img_path = "test1.jpg"

    print("Loading model...")
    model, teacher, scene_norm, grid_size = load_everything(ckpt_path, device)

    print("Getting teacher features...")
    img_t, teacher_norm = get_teacher_features(img_path, teacher, scene_norm, grid_size, device)

    print("Collecting trajectory data...")
    data = collect_trajectory_data(model, img_t, teacher_norm, grid_size, device)

    print("Computing metrics...")
    ts, deltas, changes, efficiency, frac_improvement = compute_metrics(data)

    print()
    summarize_by_timestep(ts, changes, efficiency, frac_improvement)

    print()
    plot_analysis(ts, deltas, changes, efficiency, frac_improvement, Path("outputs/metacognitive_analysis.png"))

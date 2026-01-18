"""GPU worker - owns all torch state, runs all GPU operations.

Singleton pattern. All public methods are thread-safe.
Interface uses only numpy/PIL/primitives - no torch tensors escape.
"""

import logging
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.models.resnet import ResNet50_Weights

from canvit import RecurrentState
from avp_vit.checkpoint import load as load_ckpt, load_model
from avp_vit.train.transforms import imagenet_normalize
from avp_vit.train.norm import PositionAwareNorm
from avp_vit.train.probe import load_probe
from avp_vit.train.viewpoint import Viewpoint as NamedViewpoint
from canvit.viewpoint import sample_at_viewpoint
from avp_vit.train.viz import fit_pca, imagenet_denormalize
from canvit.backbone.dinov3 import DINOv3Backbone
from canvit import create_backbone
from canvit.policy import PolicyHead
from dinov3_probes import DINOv3LinearClassificationHead
from ytch.device import sync_device

from inference_app.types import ImageContext, StepResult, TeacherFeatures, Viewpoint

log = logging.getLogger(__name__)
LABELS: list[str] = ResNet50_Weights.DEFAULT.meta["categories"]


class GPUWorker:
    """Singleton GPU worker. Owns all torch state."""

    _instance: "GPUWorker | None" = None
    _lock = threading.RLock()

    def __new__(cls) -> "GPUWorker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._model = None
        self._teacher: DINOv3Backbone | None = None
        self._scene_norm: PositionAwareNorm | None = None
        self._cls_norm: PositionAwareNorm | None = None
        self._probe: DINOv3LinearClassificationHead | None = None
        self._device: torch.device | None = None
        self._patch_size: int = 14
        self._backbone: str = ""
        self._step_count: int = -1

        # Current state
        self._state: RecurrentState | None = None
        self._image: Tensor | None = None  # current image tensor

        # Config tracking
        self._ckpt_path: str = ""
        self._device_str: str = ""

    def _sync(self) -> None:
        """Sync GPU operations."""
        if self._device is not None:
            sync_device(self._device)

    def load(self, ckpt_path: str, device_str: str) -> tuple[str, int]:
        """Load model and teacher. Returns (backbone_name, step_count)."""
        with self._lock:
            if self._ckpt_path == ckpt_path and self._device_str == device_str:
                return self._backbone, self._step_count

            log.info(f"Loading: {ckpt_path} on {device_str}")
            self._device = torch.device(device_str)
            self._model = load_model(Path(ckpt_path), self._device)
            ckpt = load_ckpt(Path(ckpt_path), "cpu")

            self._teacher = None
            try:
                self._teacher = create_backbone(ckpt["backbone"], pretrained=True).to(self._device).eval()
                assert isinstance(self._teacher, DINOv3Backbone)
            except Exception as e:
                log.warning(f"Teacher failed: {e}")

            self._scene_norm = self._load_norm(ckpt, "scene_norm_state")
            self._cls_norm = self._load_norm(ckpt, "cls_norm_state", grid=1)
            self._probe = load_probe(ckpt["backbone"], self._device)
            self._patch_size = self._model.backbone.patch_size_px
            self._backbone = ckpt.get("backbone", "?")
            self._step_count = int(ckpt.get("step") or -1)
            self._ckpt_path = ckpt_path
            self._device_str = device_str

            # Reset state
            self._state = None
            self._image = None

            self._sync()
            return self._backbone, self._step_count

    def _load_norm(self, ckpt: dict, key: str, grid: int | None = None) -> PositionAwareNorm | None:
        if (s := ckpt.get(key)) is None:
            return None
        n, d = s["mean"].shape
        norm = PositionAwareNorm(n, d, grid if grid else int(n**0.5))
        norm.load_state_dict(s)
        return norm.eval().to(self._device)

    @property
    def teacher_grid(self) -> int:
        """Grid size the teacher/norm was trained on."""
        return self._scene_norm.grid_size if self._scene_norm else 16

    def set_image(self, image_bytes: bytes, canvas_grid: int, glimpse_grid: int, normalize: bool) -> ImageContext:
        """Set current image, get teacher features, reset canvas."""
        with self._lock:
            assert self._model is not None, "Must call load() first"
            assert self._scene_norm is not None, "Checkpoint must have scene_norm_state"

            img_size = canvas_grid * self._patch_size
            tg = self.teacher_grid

            transform = transforms.Compose([
                transforms.Resize(img_size, transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                imagenet_normalize(),
            ])

            pil_orig = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB")
            img_t = transform(pil_orig)
            assert isinstance(img_t, Tensor)
            self._image = img_t.unsqueeze(0).to(self._device)

            # Reset state
            self._state = self._model.init_state(batch_size=1, canvas_grid_size=canvas_grid)

            # Get display image
            img_np = imagenet_denormalize(self._image[0].detach().cpu()).numpy()
            H, W = img_np.shape[:2]
            pil = Image.fromarray((img_np * 255).astype(np.uint8))

            # Teacher features
            with torch.no_grad():
                t_full = self._get_teacher(tg, normalize=normalize, interp=False)
                t_glimpse = self._get_teacher(glimpse_grid, normalize=normalize, interp=True)

            self._sync()

            return ImageContext(
                pil=pil, H=H, W=W,
                teacher_full=t_full,
                teacher_glimpse=t_glimpse,
                pca_full=fit_pca(t_full.scene) if t_full.scene is not None else None,
                pca_glimpse=fit_pca(t_glimpse.scene) if t_glimpse.scene is not None else None,
            )

    def _get_teacher(self, grid: int, *, normalize: bool, interp: bool) -> TeacherFeatures:
        """Get teacher features at given grid size."""
        if self._teacher is None or self._image is None:
            return TeacherFeatures(None, None, [], 0.0, grid)

        px = grid * self._patch_size
        img = F.interpolate(self._image, (px, px), mode="bilinear", align_corners=False)

        self._sync()
        t0 = time.perf_counter()
        feats = self._teacher.forward_norm_features(img)
        self._sync()
        ms = (time.perf_counter() - t0) * 1000

        scene = feats.patches  # [1, N, D]
        if normalize and self._scene_norm is not None:
            if grid == self._scene_norm.grid_size:
                scene = self._scene_norm(scene)
            elif interp:
                scene = self._interp_norm(scene[0], grid).unsqueeze(0)

        cls_feat = self._cls_norm(feats.cls.unsqueeze(1)).squeeze(1) if normalize and self._cls_norm else feats.cls
        top5 = self._top5(feats.cls)

        return TeacherFeatures(
            scene=scene[0].detach().cpu().numpy(),
            cls_features=cls_feat[0].detach().cpu().numpy() if cls_feat is not None else None,
            top5=top5,
            ms=ms,
            grid=grid,
        )

    def _interp_norm(self, x: Tensor, tgt: int) -> Tensor:
        """Apply norm with bilinearly interpolated stats."""
        assert self._scene_norm is not None
        src, D = self._scene_norm.grid_size, self._scene_norm.mean.shape[-1]
        dev = x.device

        def reshape_interp(t: Tensor) -> Tensor:
            t2d = t.view(src, src, D).permute(2, 0, 1).unsqueeze(0).to(dev)
            t_up = F.interpolate(t2d, (tgt, tgt), mode="bilinear", align_corners=False)
            return t_up[0].permute(1, 2, 0).reshape(-1, D)

        mean, var = reshape_interp(self._scene_norm.mean), reshape_interp(self._scene_norm.var)
        return (x - mean) / (var + self._scene_norm.eps).sqrt()

    def _top5(self, cls: Tensor) -> list[tuple[str, float]]:
        if self._probe is None:
            return []
        p, c = F.softmax(self._probe(cls), dim=-1)[0].topk(5)
        return [(LABELS[i], prob) for i, prob in zip(c.tolist(), p.tolist())]

    def step(self, vp: Viewpoint, glimpse_grid: int, canvas_grid: int, l2_norm: bool) -> StepResult:
        """Run one model step. Returns StepResult with numpy arrays."""
        with self._lock:
            assert self._model is not None and self._state is not None and self._image is not None

            # Convert to torch viewpoint
            named_vp = NamedViewpoint(
                name=vp.name,
                centers=torch.tensor([[vp.cy, vp.cx]], device=self._device),
                scales=torch.tensor([vp.scale], device=self._device),
            )
            glimpse_px = glimpse_grid * self._patch_size

            self._sync()
            t_start = time.perf_counter()
            with torch.no_grad():
                glimpse = sample_at_viewpoint(
                    spatial=self._image, viewpoint=named_vp, glimpse_size_px=glimpse_px
                )
                out = self._model.forward(glimpse=glimpse, state=self._state, viewpoint=named_vp)
            self._sync()
            t_forward = time.perf_counter()

            # Update state
            self._state = out.state

            # Extract features
            spatial = self._model.get_spatial(out.state.canvas)[0]
            if l2_norm:
                spatial = F.normalize(spatial, p=2, dim=-1)
            scene = self._model.predict_teacher_scene(out.state.canvas)

            # Cosine similarities
            scene_cos = cls_cos = None

            # Classification
            cls_pred = self._model.predict_scene_teacher_cls(out.state.recurrent_cls)
            top5 = self._top5(self._cls_norm.denormalize(cls_pred)) if self._probe and self._cls_norm else []

            # Policy
            policy_center = policy_scale = None
            if isinstance(self._model.policy, PolicyHead) and out.vpe is not None:
                pol = self._model.policy(out.vpe)
                cy, cx = pol.position[0].tolist()
                policy_center = (cy, cx)
                policy_scale = pol.scale[0].item()

            # Numpy conversion (includes CPU transfer)
            hidden_np = spatial.detach().cpu().numpy()
            projected_np = scene[0].detach().cpu().numpy()
            glimpse_np = imagenet_denormalize(glimpse[0].detach().cpu()).numpy()

            t_end = time.perf_counter()

            ms_forward = (t_forward - t_start) * 1000
            ms_post = (t_end - t_forward) * 1000
            ms_total = (t_end - t_start) * 1000

            return StepResult(
                hidden=hidden_np,
                projected=projected_np,
                glimpse=glimpse_np,
                scene_cos=scene_cos,
                cls_cos=cls_cos,
                ms=ms_total,
                ms_forward=ms_forward,
                ms_post=ms_post,
                top5=top5,
                policy_center=policy_center,
                policy_scale=policy_scale,
            )

    def reset_canvas(self, canvas_grid: int) -> None:
        """Reset recurrent state."""
        with self._lock:
            if self._model is not None:
                self._state = self._model.init_state(batch_size=1, canvas_grid_size=canvas_grid)
                self._sync()

    def get_info(self) -> dict:
        """Get worker info for debug display."""
        with self._lock:
            return {
                "backbone": self._backbone,
                "step": self._step_count,
                "patch_size": self._patch_size,
                "has_teacher": self._teacher is not None,
                "has_norm": self._scene_norm is not None,
                "has_probe": self._probe is not None,
                "has_policy": self._model is not None and hasattr(self._model, "policy") and self._model.policy is not None,
            }


# Global instance
def get_worker() -> GPUWorker:
    """Get the singleton GPU worker."""
    return GPUWorker()

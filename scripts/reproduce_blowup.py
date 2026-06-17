"""Instrumented GPU replay to localize the IN1k training NaN.

Resumes from a clean checkpoint (full model+optimizer+scheduler state) and
replays training with the deterministic shard data order, capturing — in ONE
run — per-module activation magnitudes (absmax AND L2), per-module gradient
norms, teacher-target data statistics, loss components, and finiteness at every
substage. The weights checkpoint only shows the aftermath; the origin is a
runtime activation/gradient, visible only here.

Faithfulness:
- cfg reconstructed from the checkpoint's `training_config_history` (the actual
  run config): data paths / branch counts / chunk_size / continue_prob match.
- num_workers MUST match the run (shard worker-interleaving sets batch
  composition); default 16.
- Same FlashAttention SDPA + bf16 autocast + backward_pass_autocast=off as loop.py.
- Glimpse viewpoints are RNG (not checkpointed) -> student path is stochastic;
  teacher TARGETS are reproduced exactly. A target-driven trigger recurs at the
  same step; a glimpse-driven one is stochastic (replay many steps / seeds).

Outputs: a per-step JSONL trajectory (--out) + a full per-module dump on the
first non-finite (target / activation / loss / grad).
"""

import argparse
import json
import logging
from collections import deque
from contextlib import nullcontext
from pathlib import Path

# Match loop.py: backward runs outside autocast.
import torch._functorch.config

torch._functorch.config.backward_pass_autocast = "off"  # type: ignore[attr-defined]

import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402

# Match loop.py: force FlashAttention SDPA.
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)

import dacite  # noqa: E402
from canvit_pytorch import create_backbone  # noqa: E402

from canvit_pretrain import CanViTForPretraining, CanViTForPretrainingConfig  # noqa: E402
from canvit_pretrain.checkpoint import load as load_checkpoint  # noqa: E402
from canvit_pretrain.checkpoint import load_state_dict_flexible  # noqa: E402
from canvit_pretrain.train.config import Config  # noqa: E402
from canvit_pretrain.train.data import create_loaders  # noqa: E402
from canvit_pretrain.train.model import load_teacher  # noqa: E402
from canvit_pretrain.train.scheduler import warmup_constant_scheduler  # noqa: E402
from canvit_pretrain.train.step import training_step  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("reproduce_blowup")


def cfg_from_history(hist_entry: dict, device: torch.device) -> Config:
    def p(key: str) -> Path | None:
        v = hist_entry.get(key)
        return Path(v) if v not in (None, "None") else None

    cfg = Config(device=device)
    cfg.dataset = hist_entry["dataset"]
    cfg.feature_base_dir = p("feature_base_dir")
    cfg.feature_image_root = p("feature_image_root")
    cfg.tar_dir = p("tar_dir")
    cfg.val_dir = p("val_dir") or cfg.val_dir
    cfg.train_index_dir = p("train_index_dir")
    cfg.val_index_dir = p("val_index_dir")
    cfg.scene_resolution = int(hist_entry["scene_resolution"])
    cfg.batch_size = int(hist_entry["batch_size"])
    cfg.glimpse_grid_size = int(hist_entry["glimpse_grid_size"])
    cfg.canvas_patch_grid_size = int(hist_entry["canvas_patch_grid_size"])
    cfg.n_full_start_branches = int(hist_entry["n_full_start_branches"])
    cfg.n_random_start_branches = int(hist_entry["n_random_start_branches"])
    cfg.chunk_size = int(hist_entry["chunk_size"])
    cfg.continue_prob = float(hist_entry["continue_prob"])
    cfg.min_viewpoint_scale = float(hist_entry["min_viewpoint_scale"])
    cfg.grad_clip = float(hist_entry["grad_clip"])
    cfg.peak_lr = float(hist_entry["peak_lr"])
    cfg.weight_decay = float(hist_entry["weight_decay"])
    cfg.warmup_steps = int(hist_entry["warmup_steps"])
    cfg.amp = bool(hist_entry["amp"])
    cfg.compile = False  # eager for clean per-module hooks
    return cfg


def _flatten_tensors(x) -> list[Tensor]:
    if isinstance(x, Tensor):
        return [x]
    if isinstance(x, (list, tuple)):
        out: list[Tensor] = []
        for e in x:
            out.extend(_flatten_tensors(e))
        return out
    if hasattr(x, "__dict__"):
        out = []
        for v in vars(x).values():
            out.extend(_flatten_tensors(v))
        return out
    return []


class ActivationWatch:
    """Per-leaf-module forward-output absmax + L2 + finiteness, GPU-resident.

    Tracks the PEAK over all fires within a step (a module fires once per glimpse
    per branch). No .item() until reset/dump.
    """

    def __init__(self, model: nn.Module) -> None:
        self.absmax: dict[str, Tensor] = {}
        self.l2: dict[str, Tensor] = {}
        self.nf_out: dict[str, Tensor] = {}
        self.nf_in: dict[str, Tensor] = {}
        for name, module in model.named_modules():
            if name == "" or any(True for _ in module.children()):
                continue
            module.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def hook(_m, inputs, output):
            outs = [t for t in _flatten_tensors(output) if t.is_floating_point()]
            if not outs:
                return
            ins = [t for t in _flatten_tensors(inputs) if t.is_floating_point()]
            am = torch.stack([t.detach().abs().amax() for t in outs]).amax()
            l2 = torch.stack([t.detach().float().pow(2).sum() for t in outs]).sum().sqrt()
            ob = torch.stack([(~torch.isfinite(t)).any() for t in outs]).any()
            self.absmax[name] = am if name not in self.absmax else torch.maximum(self.absmax[name], am)
            self.l2[name] = l2 if name not in self.l2 else torch.maximum(self.l2[name], l2)
            self.nf_out[name] = ob if name not in self.nf_out else (self.nf_out[name] | ob)
            if ins:
                ib = torch.stack([(~torch.isfinite(t)).any() for t in ins]).any()
                self.nf_in[name] = ib if name not in self.nf_in else (self.nf_in[name] | ib)
        return hook

    def reset(self) -> None:
        self.absmax.clear()
        self.l2.clear()
        self.nf_out.clear()
        self.nf_in.clear()

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {n: {"absmax": float(v.item()), "l2": float(self.l2[n].item())} for n, v in self.absmax.items()}

    def top(self, k: int, by: str = "absmax") -> list[tuple[str, float, float]]:
        rows = [(n, float(self.absmax[n].item()), float(self.l2[n].item())) for n in self.absmax]
        return sorted(rows, key=lambda r: r[1] if by == "absmax" else r[2], reverse=True)[:k]

    def origin_modules(self) -> list[str]:
        out = []
        for name, bad in self.nf_out.items():
            if bool(bad.item()):
                ib = self.nf_in.get(name)
                was = bool(ib.item()) if ib is not None else False
                out.append(f"{name}{'  [input ALSO nonfinite -> propagated]' if was else '  [input FINITE -> ORIGIN]'}")
        return out


def grad_stats(model: nn.Module) -> dict:
    """Per-top-module grad norm + absmax + finiteness; totals. GPU-resident reductions.

    No big cat: accumulate per-module sum-of-squares and absmax on GPU, sync once
    per module. A module's grad-norm is non-finite iff any grad element is, so
    finiteness is read off the (single) synced norm value.
    """
    sumsq: dict[str, Tensor] = {}
    amax: dict[str, Tensor] = {}
    for name, prm in model.named_parameters():
        if prm.grad is None:
            continue
        mod = name.split(".")[0]
        g = prm.grad.detach()
        ss = g.float().pow(2).sum()
        am = g.abs().amax()
        sumsq[mod] = ss if mod not in sumsq else sumsq[mod] + ss
        amax[mod] = am if mod not in amax else torch.maximum(amax[mod], am)
    per_module = {}
    total_sq = 0.0
    nonfinite_mods = []
    for mod in sumsq:
        nrm = float(sumsq[mod].sqrt().item())
        am = float(amax[mod].item())
        finite = nrm == nrm and nrm != float("inf") and am == am and am != float("inf")
        per_module[mod] = {"norm": nrm, "absmax": am, "finite": finite}
        if not finite:
            nonfinite_mods.append(mod)
        else:
            total_sq += nrm * nrm
    return {"per_module": per_module, "total_norm": total_sq ** 0.5, "nonfinite_modules": nonfinite_mods}


def data_stats(raw_patches: Tensor, raw_cls: Tensor, scene_t: Tensor, cls_t: Tensor) -> dict:
    # per-sample max over patches to find an outlier image in the batch
    per_sample = raw_patches.detach().abs().amax(dim=(1, 2))  # [B]
    worst = int(per_sample.argmax().item())
    return {
        "raw_patches": {"absmax": float(raw_patches.abs().max().item()), "l2": float(raw_patches.float().norm().item()),
                        "finite": bool(torch.isfinite(raw_patches).all().item())},
        "raw_cls": {"absmax": float(raw_cls.abs().max().item()), "finite": bool(torch.isfinite(raw_cls).all().item())},
        "norm_scene": {"absmax": float(scene_t.abs().max().item()), "l2": float(scene_t.float().norm().item()),
                       "finite": bool(torch.isfinite(scene_t).all().item())},
        "norm_cls": {"absmax": float(cls_t.abs().max().item()), "finite": bool(torch.isfinite(cls_t).all().item())},
        "worst_sample_in_batch": worst,
        "worst_sample_absmax": float(per_sample[worst].item()),
    }


def loss_components(metrics) -> dict:
    out = {"total_loss": float(metrics.total_loss.item()), "n_glimpses": metrics.n_glimpses}
    for tag, m in [("full", metrics.full_start), ("random", metrics.random_start)]:
        if m is None:
            continue
        out[f"{tag}_loss"] = float(m.loss.item())
        out[f"{tag}_scene_patches"] = float(m.scene_patches_loss.item())
        out[f"{tag}_scene_cls"] = float(m.scene_cls_loss.item())
        out[f"{tag}_scene_cos_norm"] = float(m.scene_cos_norm.item())
        out[f"{tag}_cls_cos_norm"] = float(m.cls_cos_norm.item())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--n-steps", type=int, default=5200)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("reproduce_blowup_trajectory.jsonl"))
    ap.add_argument("--anomaly", action="store_true", help="set_detect_anomaly (slow; pinpoints backward NaN)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    if args.anomaly:
        torch.autograd.set_detect_anomaly(True)

    ckpt = load_checkpoint(args.ckpt, device)
    assert ckpt["step"] is not None
    hist = ckpt["training_config_history"]
    assert hist, "checkpoint has no training_config_history"
    cfg = cfg_from_history(hist[sorted(hist)[-1]], device)
    log.info(f"Resuming {args.ckpt.name} step={ckpt['step']} dataset={cfg.dataset} seed={args.seed}")
    log.info(f"feature_base_dir={cfg.feature_base_dir} image_root={cfg.feature_image_root}")

    model_cfg = dacite.from_dict(CanViTForPretrainingConfig, ckpt["model_config"])
    model = CanViTForPretraining(
        backbone=create_backbone(ckpt["backbone_name"]), cfg=model_cfg,
        backbone_name=ckpt["backbone_name"], canvas_patch_grid_sizes=ckpt["canvas_patch_grid_sizes"],
    ).to(device)
    load_state_dict_flexible(model, ckpt["state_dict"])
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.peak_lr, weight_decay=cfg.weight_decay)
    scheduler = warmup_constant_scheduler(optimizer, cfg.warmup_steps, cfg.peak_lr, start_lr=cfg.start_lr)
    assert ckpt["optimizer_state"] and ckpt["scheduler_state"]
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_step = scheduler.last_epoch

    teacher = load_teacher(cfg)
    G = cfg.canvas_patch_grid_size
    glimpse_size_px = cfg.glimpse_grid_size * teacher.model.config.patch_size
    cfg.num_workers = args.num_workers
    train_loader, _ = create_loaders(cfg, start_step=start_step)
    cls_norm, scene_norm = model.standardizers(G)
    assert scene_norm.initialized

    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if cfg.amp else nullcontext()
    watch = ActivationWatch(model)
    nb = cfg.non_blocking_transfer
    recent = deque(maxlen=20)  # rolling window of step records into the failure
    log.info(f"Replaying up to {args.n_steps} steps (global {start_step}->{start_step + args.n_steps}) out={args.out}")

    with open(args.out, "w") as fout:
        for i in range(args.n_steps):
            step = start_step + i
            images, raw_patches, raw_cls, labels = train_loader.next()
            images = images.to(device, non_blocking=nb)
            raw_patches = raw_patches.to(device=device, dtype=torch.float32, non_blocking=nb)
            raw_cls = raw_cls.to(device=device, dtype=torch.float32, non_blocking=nb)
            scene_t = scene_norm(raw_patches)
            cls_t = cls_norm(raw_cls.unsqueeze(1)).squeeze(1)
            dstats = data_stats(raw_patches, raw_cls, scene_t, cls_t)

            watch.reset()
            optimizer.zero_grad()
            metrics = training_step(
                model=model, images=images, scene_target=scene_t, cls_target=cls_t,
                raw_scene_target=raw_patches, raw_cls_target=raw_cls,
                scene_denorm=scene_norm.destandardize, cls_denorm=cls_norm.destandardize,
                enable_scene_patches_loss=cfg.enable_scene_patches_loss,
                enable_scene_cls_loss=cfg.enable_scene_cls_loss,
                glimpse_size_px=glimpse_size_px, canvas_grid_size=G,
                n_full_start_branches=cfg.n_full_start_branches,
                n_random_start_branches=cfg.n_random_start_branches,
                chunk_size=cfg.chunk_size, continue_prob=cfg.continue_prob,
                min_viewpoint_scale=cfg.min_viewpoint_scale, amp_ctx=amp_ctx,
            )
            lstats = loss_components(metrics)
            gstats = grad_stats(model)
            top_act = watch.top(8, by="absmax")
            data_bad = not (dstats["raw_patches"]["finite"] and dstats["raw_cls"]["finite"]
                            and dstats["norm_scene"]["finite"] and dstats["norm_cls"]["finite"])
            loss_bad = lstats["total_loss"] != lstats["total_loss"]  # NaN check
            grad_bad = bool(gstats["nonfinite_modules"])

            rec = {"step": step, "i": i, "data": dstats, "loss": lstats,
                   "grad_total_norm": gstats["total_norm"], "grad_nonfinite_modules": gstats["nonfinite_modules"],
                   "top_act_absmax": [[n, a, l2v] for n, a, l2v in top_act]}
            recent.append(rec)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()

            if data_bad or loss_bad or grad_bad:
                log.error("=" * 72)
                log.error(f"NON-FINITE at step {step} (i={i}, in-task offset={step - start_step})")
                log.error(f"  DATA: raw_absmax={dstats['raw_patches']['absmax']:.4e} "
                          f"norm_absmax={dstats['norm_scene']['absmax']:.4e} data_bad={data_bad} "
                          f"worst_sample={dstats['worst_sample_in_batch']} (|x|={dstats['worst_sample_absmax']:.4e})")
                log.error(f"  LOSS: {lstats}")
                log.error(f"  forward-activation ORIGIN: {watch.origin_modules() or '(no nonfinite activation)'}")
                log.error("  top activations (absmax | l2):")
                for n, a, l2v in watch.top(20, by="absmax"):
                    log.error(f"      {n:42s} absmax={a:.4e}  l2={l2v:.4e}")
                log.error(f"  grad total_norm={gstats['total_norm']:.4e} nf_mods={gstats['nonfinite_modules']}")
                log.error("  per-module grad (norm | absmax | finite):")
                for mod, gd in sorted(gstats["per_module"].items(), key=lambda kv: kv[1]["norm"], reverse=True):
                    log.error(f"      {mod:30s} norm={gd['norm']:.4e} absmax={gd['absmax']:.4e} finite={gd['finite']}")
                log.error("  --- last 20 steps trajectory (loss / grad_norm / data_absmax / top_act) ---")
                for r in recent:
                    ta = r["top_act_absmax"][0] if r["top_act_absmax"] else ["-", 0, 0]
                    rd = r["data"]["raw_patches"]["absmax"]
                    log.error(f"      step {r['step']}: loss={r['loss']['total_loss']:.4f} "
                              f"grad_norm={r['grad_total_norm']:.3e} raw_absmax={rd:.3e} "
                              f"top_act={ta[0]}={ta[1]:.3e}")
                log.error("=" * 72)
                break

            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            if i % args.log_every == 0:
                ta = " ".join(f"{n}={a:.2e}" for n, a, _ in top_act[:4])
                log.info(f"step {step} (i={i}) loss={lstats['total_loss']:.4f} "
                         f"grad_norm={gstats['total_norm']:.3e} raw_absmax={dstats['raw_patches']['absmax']:.3e} "
                         f"norm_absmax={dstats['norm_scene']['absmax']:.3e} n_gl={lstats['n_glimpses']} | act: {ta}")
        else:
            log.info(f"Completed {args.n_steps} steps, NO non-finite. Trigger stochastic/glimpse-dependent; "
                     f"rerun with a different --seed. Trajectory in {args.out}")


if __name__ == "__main__":
    main()

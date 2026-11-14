"""Fine-tune an existing gsplat checkpoint on edited images with the poses FIXED (Week 2 rebuild, "fine-tune" mode).

The scratch rebuild re-trains from the sparse points (stage 4 on the edited scene). Fine-tuning instead starts from the
trained Gaussians, so most of the room is already right and only the edited region has to change: far fewer steps.
Same losses / learning rates as gsplat's examples/simple_trainer.py (scaled down), DefaultStrategy densification kept
for a while so new geometry can appear where the object changed, opacity resets switched off (they would erase the
parts of the room that do not change).

Needs CUDA + gsplat. NOT run in the authoring sandbox (no GPU): run `python tools/w2_selftest.py` on Kaggle first.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np


def lr_for(step: int, steps: int, base: float, final_factor: float = 0.01) -> float:
    return base * final_factor ** (step / max(steps, 1))


def split_names(names: list[str], every: int) -> tuple[list[str], list[str]]:
    """gsplat convention: sorted registered images, index % every == 0 are held out."""
    names = sorted(names)
    val = [n for i, n in enumerate(names) if every and i % every == 0]
    return [n for n in names if n not in set(val)], val


def scene_scale_from_centers(centers: np.ndarray) -> float:
    c = centers.mean(0)
    return float(np.max(np.linalg.norm(centers - c, axis=1)))


def finetune(ckpt_path: Path, scene_dir: Path, out_dir: Path, *, steps: int = 3000, device: str = "cuda",
             holdout_every: int = 8, lr_scale: float = 0.25, sh_degree: int = 3, densify: bool = True,
             seed: int = 0, progress=None, log_every: int = 250, poses: dict | None = None) -> dict:
    import cv2
    import torch
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy

    from common.colmap_io import read_model
    from common.gs_io import ckpt_to_ply

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ---- data: poses are fixed, images are the (edited) scene images --------------------------------------------------
    model = read_model(Path(scene_dir) / "sparse" / "0")
    by_name = {im.name: im for im in model.images.values()}
    train_names, val_names = split_names(list(by_name), holdout_every)
    # poses must be in the CHECKPOINT's space (gsplat normalizes the world by default): common/gs_space.py
    if poses is not None:
        centers = np.stack([-poses[n][:3, :3].T @ poses[n][:3, 3] for n in by_name if n in poses])
    else:
        centers = np.stack([-im.R.T @ im.tvec for im in by_name.values()])
    scene_scale = scene_scale_from_centers(centers) * 1.1

    def load(name):
        im, cam = by_name[name], model.cameras[by_name[name].camera_id]
        img = cv2.cvtColor(cv2.imread(str(Path(scene_dir) / "images" / name)), cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (cam.height, cam.width):
            raise RuntimeError(f"{name}: image {img.shape[1::-1]} != camera {(cam.width, cam.height)} (edited frames "
                               "must be in scene/images space)")
        V = np.eye(4, dtype=np.float32)
        if poses is not None and name in poses:
            V = np.asarray(poses[name], dtype=np.float32)
        else:
            V[:3, :3], V[:3, 3] = im.R, im.tvec
        return (torch.from_numpy(img).to(device), torch.from_numpy(V).to(device),
                torch.from_numpy(cam.K().astype(np.float32)).to(device), cam.width, cam.height)

    data = {n: load(n) for n in train_names + val_names}

    # ---- model --------------------------------------------------------------------------------------------------------
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v.detach().float().to(device)) for k, v in ck["splats"].items()})
    base_lr = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3,
               "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{"params": splats[k], "lr": base_lr[k] * lr_scale, "name": k}], eps=1e-15)
                  for k in splats.keys()}
    strategy = DefaultStrategy(verbose=False, refine_start_iter=0, refine_stop_iter=int(0.7 * steps) if densify else 0,
                               refine_every=100, reset_every=10 ** 9)
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)
    n_start = len(splats["means"])

    try:
        from fused_ssim import fused_ssim
    except Exception:
        fused_ssim = None

    def render(name):
        gt, V, K, W, H = data[name]
        colors = torch.cat([splats["sh0"], splats["shN"]], 1)
        out, alpha, info = rasterization(splats["means"], splats["quats"], torch.exp(splats["scales"]),
                                         torch.sigmoid(splats["opacities"]), colors, V[None], K[None], W, H,
                                         sh_degree=sh_degree, packed=False, absgrad=False)
        return out.clamp(0, 1), gt[None].float() / 255.0, info

    # ---- train --------------------------------------------------------------------------------------------------------
    t0 = time.perf_counter()
    for step in range(steps):
        pred, gt, info = render(train_names[int(rng.integers(len(train_names)))])
        strategy.step_pre_backward(splats, optimizers, state, step, info)
        loss = F_l1(pred, gt) * 0.8
        loss = loss + (0.2 * (1.0 - fused_ssim(pred.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2), padding="valid"))
                       if fused_ssim else 0.2 * F_l1(pred, gt))
        loss.backward()
        for k, opt in optimizers.items():
            for g in opt.param_groups:
                g["lr"] = lr_for(step, steps, base_lr[k] * lr_scale) if k == "means" else base_lr[k] * lr_scale
            opt.step()
            opt.zero_grad(set_to_none=True)
        strategy.step_post_backward(splats, optimizers, state, step, info, packed=False)
        if progress and step % 20 == 0:
            progress(step / steps)
        if step % log_every == 0 or step == steps - 1:
            print(f"[finetune] step {step}/{steps} loss {float(loss.detach()):.4f} gaussians {len(splats['means'])}", flush=True)
    train_s = time.perf_counter() - t0

    # ---- evaluate on the held-out frames, save --------------------------------------------------------------------------
    psnrs, ssims = [], []
    with torch.no_grad():
        for n in val_names:
            pred, gt, _ = render(n)
            psnrs.append(float(-10 * math.log10(max(float(((pred - gt) ** 2).mean()), 1e-10))))
            if fused_ssim:
                ssims.append(float(fused_ssim(pred.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2), padding="valid")))
    out_dir = Path(out_dir)
    (out_dir / "ckpts").mkdir(parents=True, exist_ok=True)
    ckpt_out = out_dir / "ckpts" / f"ckpt_{steps - 1}.pt"
    torch.save({"step": steps - 1, "splats": splats.state_dict()}, ckpt_out)
    (out_dir / "ply").mkdir(exist_ok=True)
    ply_out = out_dir / "ply" / f"point_cloud_{steps - 1}.ply"
    ckpt_to_ply(ckpt_out, ply_out)
    res = {"trainer": "gsplat_finetune", "iters": steps, "num_gaussians": int(len(splats["means"])),
           "num_gaussians_start": int(n_start), "train_seconds": round(train_s, 1),
           "total_seconds": round(time.perf_counter() - t_start, 1),
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None,
           "heldout": {"psnr": float(np.mean(psnrs)) if psnrs else None, "ssim": float(np.mean(ssims)) if ssims else None,
                       "lpips": None, "n_frames": len(val_names)},
           "ckpt": str(ckpt_out), "ply": str(ply_out), "lr_scale": lr_scale, "densify": densify}
    (out_dir / "stats").mkdir(exist_ok=True)
    (out_dir / "stats" / f"val_step{steps - 1:04d}.json").write_text(json.dumps(res["heldout"] | {"num_GS": res["num_gaussians"]}))
    return res


def F_l1(a, b):
    return (a - b).abs().mean()

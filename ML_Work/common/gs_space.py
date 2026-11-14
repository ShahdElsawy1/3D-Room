"""Poses that match a gsplat checkpoint (fixes "fine-tune PSNR 8" / empty 3D lift in Week 2).

Why this exists
---------------
gsplat's examples/simple_trainer.py defaults to `normalize_world_space=True`: Parser re-centres, rotates (up axis) and
rescales the COLMAP scene before training, so the Gaussians in the checkpoint live in a *normalized* space. Rendering
that checkpoint with the raw COLMAP world->camera poses gives garbage (Week 2 self-test: loss ~0.5, held-out PSNR 8.2
vs 26.8 in Week 1). Everything in Week 2 that renders or lifts through the checkpoint (splat depth, 3D lift,
fine-tune) must therefore use the poses in the SAME space as the checkpoint.

`week2/splat_poses.json` holds them: {"space": "raw"|"normalized", "psnr": {...}, "poses": {name: 4x4 world->cam}}.
It is produced by stages/w2_calibrate.py, which does not assume which space the checkpoint is in: it renders a few
training views with both candidate pose sets, and keeps the one that reproduces the photos (higher PSNR).

Pure-numpy helpers are unit-tested on CPU; `calibrate` needs CUDA + gsplat (+ gsplat's examples folder for Parser).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

FILE = "splat_poses.json"


# ----------------------------------------------------------------------------- pure numpy
def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """(N,4,4) camera->world  ->  (N,4,4) world->camera."""
    return np.linalg.inv(np.asarray(c2w, dtype=np.float64))


def apply_world_transform(w2c: np.ndarray, T: np.ndarray) -> np.ndarray:
    """World similarity T (4x4, raw world -> normalized world) applied to a world->camera pose.
    c2w' = T @ c2w, then the rotation block is re-orthonormalized (the scale lives in the world, not in the camera),
    as gsplat's Parser does. Only used for tests / when Parser's own poses are not available."""
    c2w = np.asarray(T, dtype=np.float64) @ np.linalg.inv(np.asarray(w2c, dtype=np.float64))
    c2w[:3, :3] /= np.linalg.norm(c2w[:3, 0])
    return np.linalg.inv(c2w)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return -10.0 * math.log10(max(mse, 1e-10))


def pick_space(psnr_raw: float, psnr_norm: float | None, margin: float = 1.0) -> str:
    """Raw wins ties (nothing to repair); normalized must beat raw by `margin` dB."""
    if psnr_norm is None:
        return "raw"
    return "normalized" if psnr_norm > psnr_raw + margin else "raw"


def save_poses(path: Path, space: str, poses: dict[str, np.ndarray], psnrs: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"space": space, "psnr": psnrs,
                                "poses": {n: np.asarray(V, float).tolist() for n, V in poses.items()}}))


def load_poses(path: Path) -> dict[str, np.ndarray] | None:
    """name -> 4x4 world->camera in the checkpoint's space, or None when stage w2_calibrate has not run."""
    path = Path(path)
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return {n: np.asarray(V, dtype=np.float64) for n, V in d["poses"].items()}


# ----------------------------------------------------------------------------- GPU / gsplat part
def normalized_poses_from_gsplat(scene_dir: Path, gsplat_dir: Path, factor: int = 1, test_every: int = 8) -> dict:
    """Use gsplat's own Parser (the code that normalized the training scene) -> name -> 4x4 world->camera."""
    ex = Path(gsplat_dir) / "examples"
    if not (ex / "datasets" / "colmap.py").exists():
        raise FileNotFoundError(f"{ex}/datasets/colmap.py not found - set GSPLAT_DIR (setup_kaggle.sh gsplat)")
    sys.path.insert(0, str(ex))
    from datasets.colmap import Parser  # type: ignore
    parser = Parser(data_dir=str(scene_dir), factor=factor, normalize=True, test_every=test_every)
    w2c = c2w_to_w2c(parser.camtoworlds)
    return {n: w2c[i] for i, n in enumerate(parser.image_names)}


def render_rgb(splats: dict, V: np.ndarray, K: np.ndarray, W: int, H: int, device: str = "cuda") -> np.ndarray:
    import torch
    from gsplat.rendering import rasterization
    with torch.no_grad():
        colors = torch.cat([splats["sh0"], splats["shN"]], 1)
        out, _, _ = rasterization(splats["means"], splats["quats"], torch.exp(splats["scales"]),
                                  torch.sigmoid(splats["opacities"]), colors,
                                  torch.from_numpy(np.asarray(V, np.float32)).to(device)[None],
                                  torch.from_numpy(np.asarray(K, np.float32)).to(device)[None], W, H,
                                  sh_degree=int(round(math.sqrt(colors.shape[1]) - 1)), packed=False)
        return out[0].clamp(0, 1).cpu().numpy()


def calibrate(ckpt: Path, scene_dir: Path, gsplat_dir: Path | None, *, factor: int = 1, n_views: int = 4,
              device: str = "cuda") -> tuple[str, dict, dict]:
    """Returns (space, {name: 4x4 w2c in the checkpoint's space}, {"raw": dB, "normalized": dB|None})."""
    import cv2
    import torch

    from common.colmap_io import read_model

    model = read_model(Path(scene_dir) / "sparse" / "0")
    by_name = {im.name: im for im in model.images.values()}
    raw = {}
    for n, im in by_name.items():
        V = np.eye(4)
        V[:3, :3], V[:3, 3] = im.R, im.tvec
        raw[n] = V
    norm, norm_err = None, None
    try:
        norm = normalized_poses_from_gsplat(scene_dir, gsplat_dir, factor) if gsplat_dir else None
        if norm is not None:
            norm = {n: V for n, V in norm.items() if n in raw}
    except Exception as e:                                                   # noqa: BLE001
        norm_err = f"{type(e).__name__}: {e}"
        print(f"[calibrate] normalized poses unavailable: {norm_err}", flush=True)

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    splats = {k: v.detach().float().to(device) for k, v in ck["splats"].items()}
    names = sorted(by_name)
    probe = [names[i] for i in np.linspace(1, len(names) - 2, n_views).astype(int)]   # skip the held-out index-0 view
    res = {"raw": [], "normalized": []}
    for n in probe:
        cam = model.cameras[by_name[n].camera_id]
        img = cv2.cvtColor(cv2.imread(str(Path(scene_dir) / "images" / n)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for tag, poses in (("raw", raw), ("normalized", norm)):
            if poses is None or n not in poses:
                continue
            res[tag].append(psnr(render_rgb(splats, poses[n], cam.K(), cam.width, cam.height, device), img))
    p_raw = float(np.mean(res["raw"])) if res["raw"] else float("nan")
    p_norm = float(np.mean(res["normalized"])) if res["normalized"] else None
    space = pick_space(p_raw, p_norm)
    return space, (norm if space == "normalized" else raw), {"raw": p_raw, "normalized": p_norm,
                                                             "normalized_error": norm_err, "probe_views": probe}

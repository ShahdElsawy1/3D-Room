"""Lift a reference mask to 3D and reproject it into the other frames (tracking method 2).

Why: SAM 2 video propagation follows an object frame to frame, but a 1-4 fps room video has large viewpoint changes and
the object leaves the view and comes back. We already have, for every frame, the camera pose (SfM) and a dense depth map
rendered from the trained splat, so the object can be placed in 3D once and looked up in every frame:

    ref mask + ref depth  -> 3D points of the object
    each other frame      -> project the points, keep the ones that are IN FRONT of the splat surface there (visibility
                             check, so an occluding sofa hides the object), close the gaps -> coarse mask + visible share

Pure numpy/cv2 (no GPU); the coarse mask is then refined by SAM 2 (common/sam_wrap.py) or used as is.
All pixel coordinates below are in scene/images pixels unless a name says `_d` (depth-map resolution = scale x scene).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Cam:
    K: np.ndarray          # 3x3 at scene/images resolution
    R: np.ndarray          # world -> camera
    t: np.ndarray
    width: int
    height: int


def _scaled(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    K2 = K.astype(np.float64).copy()
    K2[0, 0] *= sx; K2[0, 2] *= sx; K2[1, 1] *= sy; K2[1, 2] *= sy
    return K2


def unproject_mask(mask: np.ndarray, depth: np.ndarray, cam: Cam, max_points: int = 200_000,
                   erode_px: int = 2, seed: int = 0) -> np.ndarray:
    """Mask (scene-res bool) + depth map (any resolution, 0 = no surface) -> (N,3) world points of the object.
    The mask is eroded a little so boundary pixels (which sit on the background depth) do not leak in."""
    h, w = depth.shape
    m = cv2.resize((mask > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    if erode_px > 0:
        m = cv2.erode(m, np.ones((2 * erode_px + 1,) * 2, np.uint8))
    ys, xs = np.nonzero((m > 0) & (depth > 0))
    if len(xs) == 0:
        return np.zeros((0, 3))
    if len(xs) > max_points:
        sel = np.random.default_rng(seed).choice(len(xs), max_points, replace=False)
        ys, xs = ys[sel], xs[sel]
    z = depth[ys, xs].astype(np.float64)
    K = _scaled(cam.K, w / cam.width, h / cam.height)
    x = (xs + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (ys + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    return (pc - cam.t) @ cam.R           # R^T (p_cam - t), row vectors


def project_visible(pts: np.ndarray, cam: Cam, depth: np.ndarray | None, rel_tol: float = 0.06,
                    abs_tol: float = 0.02, unknown_visible: bool = False) -> dict:
    """Project world points into `cam`. `depth` (depth-map resolution, 0 = no surface) is that frame's splat depth.
    Returns uv_d (N,2) in depth-map pixels, z, in_view, visible (in view AND not behind the surface seen there)."""
    h, w = depth.shape if depth is not None else (cam.height, cam.width)
    K = _scaled(cam.K, w / cam.width, h / cam.height)
    pc = pts @ cam.R.T + cam.t
    z = pc[:, 2]
    zc = np.clip(z, 1e-9, None)
    u = pc[:, 0] / zc * K[0, 0] + K[0, 2]
    v = pc[:, 1] / zc * K[1, 1] + K[1, 2]
    in_view = (z > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    vis = in_view.copy()
    if depth is not None and in_view.any():
        d = np.zeros_like(z)
        ui, vi = np.floor(u[in_view]).astype(int), np.floor(v[in_view]).astype(int)
        d[in_view] = depth[vi, ui]
        known = d > 0
        ok = np.abs(z - d) <= np.maximum(abs_tol, rel_tol * z)
        vis = in_view & ((known & ok) | (~known & unknown_visible))
    return {"u": u, "v": v, "z": z, "in_view": in_view, "visible": vis, "size_d": (w, h)}


def points_to_mask(proj: dict, out_hw: tuple[int, int], close_frac: float = 0.012, min_points: int = 30) -> np.ndarray:
    """Splat the visible points, close the gaps between them, fill holes, resize to out_hw=(H,W)."""
    w, h = proj["size_d"]
    vis = proj["visible"]
    canvas = np.zeros((h, w), np.uint8)
    if vis.sum() >= min_points:
        canvas[np.clip(proj["v"][vis].astype(int), 0, h - 1), np.clip(proj["u"][vis].astype(int), 0, w - 1)] = 1
        k = max(3, int(round(close_frac * max(h, w))) | 1)
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        ff = canvas.copy()                                  # fill holes: flood the background from a border, invert
        pad = cv2.copyMakeBorder(ff, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        mask = np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8)
        cv2.floodFill(pad, mask, (0, 0), 2)
        canvas = ((pad[1:-1, 1:-1] != 2) | (canvas > 0)).astype(np.uint8)
    H, W = out_hw
    return cv2.resize(canvas, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)


def lift_to_frame(pts: np.ndarray, cam: Cam, depth: np.ndarray | None, out_hw: tuple[int, int], **kw) -> dict:
    """Everything tracking needs for one target frame: coarse mask, how much of the object is visible/in view."""
    proj = project_visible(pts, cam, depth, **{k: v for k, v in kw.items() if k in ("rel_tol", "abs_tol", "unknown_visible")})
    n = max(len(pts), 1)
    mask = points_to_mask(proj, out_hw, close_frac=kw.get("close_frac", 0.012))
    in_view, vis = float(proj["in_view"].sum() / n), float(proj["visible"].sum() / n)
    return {"mask": mask, "in_view_frac": in_view, "visible_frac": vis,
            "occluded_frac": 0.0 if in_view == 0 else float(1 - vis / in_view)}


def prompts_from_mask(mask: np.ndarray, n_pos: int = 3, n_neg: int = 3, seed: int = 0):
    """Box + positive points inside the (eroded) mask + negative points just outside it, for SAM refinement."""
    m = (mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)
    rng = np.random.default_rng(seed)
    k = max(3, int(0.05 * max(box[2] - box[0], box[3] - box[1])) | 1)
    inner = cv2.erode(m, np.ones((k, k), np.uint8))
    iy, ix = np.nonzero(inner if inner.any() else m)
    pos = [(int(ix[i]), int(iy[i])) for i in rng.choice(len(ix), min(n_pos, len(ix)), replace=False)]
    ring = cv2.dilate(m, np.ones((3 * k, 3 * k), np.uint8)) - cv2.dilate(m, np.ones((k, k), np.uint8))
    ry, rx = np.nonzero(ring > 0)
    neg = [(int(rx[i]), int(ry[i])) for i in rng.choice(len(rx), min(n_neg, len(rx)), replace=False)] if len(rx) else []
    return {"box": box, "pos": pos, "neg": neg}

"""Small, dependency-light geometry toolbox (numpy only).

Conventions (same as COLMAP / VGGT, OpenCV style):
    * a camera is described by world->camera:  x_cam = R @ x_world + t
    * +x right, +y down, +z forward
    * quaternions are (w, x, y, z)
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- rotations
def qvec2rotmat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def rotmat2qvec(R) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    q = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if q[0] < 0:
        q = -q
    return q


def camera_center(R, t) -> np.ndarray:
    """World-space camera centre from world->camera (R, t)."""
    return -np.asarray(R).T @ np.asarray(t)


# ---------------------------------------------------------------- Sim(3)
def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares Sim(3):  dst ~= s * R @ src + t   (src, dst: Nx3)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / n
    s = float(np.trace(np.diag(D) @ S) / var_s) if with_scale and var_s > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def robust_umeyama(src, dst, iters: int = 5, keep: float = 0.8, min_pts: int = 12):
    """Trimmed Umeyama: repeatedly refit on the best `keep` fraction by residual.

    Used to merge VGGT chunks from the pixel-aligned point maps of the shared frames.
    Returns (s, R, t, rmse_of_kept).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    idx = np.arange(len(src))
    s, R, t = umeyama(src[idx], dst[idx])
    for _ in range(iters):
        res = np.linalg.norm((s * (src @ R.T) + t) - dst, axis=1)
        k = max(min_pts, int(keep * len(res)))
        idx = np.argsort(res)[:k]
        s, R, t = umeyama(src[idx], dst[idx])
    res = np.linalg.norm((s * (src[idx] @ R.T) + t) - dst[idx], axis=1)
    return s, R, t, float(np.sqrt((res ** 2).mean()))


def apply_sim3_to_w2c(R_w2c, t_w2c, s, R, t):
    """Re-express a world->camera pose after the WORLD is mapped by x' = s R x + t.

    x_cam = Rw x + tw  and  x = R^T (x' - t) / s
      => x_cam' = Rw R^T (x' - t) + s*tw   (camera coords scaled by s to stay metric
                                            in the new world's units)
    """
    R_new = R_w2c @ R.T
    t_new = -R_new @ t + s * t_w2c
    return R_new, t_new


# ---------------------------------------------------------------- projection
def unproject_depth(depth: np.ndarray, K: np.ndarray, R_w2c: np.ndarray, t_w2c: np.ndarray):
    """Depth map (H,W) -> world points (H,W,3).  depth = z in camera frame."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float64) + 0.5, np.arange(H, dtype=np.float64) + 0.5)
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    cam = np.stack([x, y, depth], axis=-1).reshape(-1, 3)
    world = (cam - t_w2c) @ R_w2c            # R^T (x_cam - t), written with row vectors
    return world.reshape(H, W, 3)


def project(points_w: np.ndarray, K: np.ndarray, R_w2c: np.ndarray, t_w2c: np.ndarray):
    """World points (N,3) -> pixels (N,2), depth (N,)."""
    cam = points_w @ R_w2c.T + t_w2c
    z = cam[:, 2]
    uv = cam[:, :2] / np.clip(z[:, None], 1e-9, None)
    uv = uv * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    return uv, z


def rescale_intrinsics(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Pinhole intrinsics after resizing the image by (sx, sy) (pixel-centre convention
    where the principal point scales linearly, as VGGT/COLMAP use)."""
    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def pose_stats(centers: np.ndarray) -> dict:
    """Trajectory summary used in the comparison report."""
    if len(centers) < 2:
        return {"path_length": 0.0, "bbox_diag": 0.0}
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    return {
        "path_length": float(steps.sum()),
        "median_step": float(np.median(steps)),
        "max_step": float(steps.max()),
        "bbox_diag": float(np.linalg.norm(centers.max(0) - centers.min(0))),
    }

"""frames/images (original, distorted) <-> scene/images (undistorted PINHOLE copy the splat was trained on).

When GLOMAP's distortion was big enough to undistort (poses/glomap/glomap_meta.json: undistorted=true), the two image
spaces differ in size and pixels. B's contract wants masks in frames/images; the cameras (and the splat) live in
scene/images. These maps convert masks/edited frames between the two (SIMPLE_RADIAL: x_d = x (1 + k r^2)).
When `same_pixels` is true in week2/inputs.json nothing here is needed.
"""
from __future__ import annotations

import cv2
import numpy as np


def _grid(w: int, h: int):
    u, v = np.meshgrid(np.arange(w, dtype=np.float64) + 0.5, np.arange(h, dtype=np.float64) + 0.5)
    return u, v


def scene_to_frames_map(raw: dict, scene_cam: dict):
    """For every SCENE pixel: where is it in the original frame? -> (map_x, map_y) for cv2.remap(frames_img, ...)."""
    f, cx, cy, k = raw["params"][:4]
    p = scene_cam["params"]
    fx, fy, cxs, cys = p[0], p[1], p[2], p[3]
    u, v = _grid(scene_cam["width"], scene_cam["height"])
    x, y = (u - cxs) / fx, (v - cys) / fy
    d = 1 + k * (x * x + y * y)
    return ((f * x * d + cx - 0.5).astype(np.float32), (f * y * d + cy - 0.5).astype(np.float32))


def frames_to_scene_map(raw: dict, scene_cam: dict):
    """For every FRAME pixel: where is it in the undistorted scene image? -> maps for cv2.remap(scene_img, ...)."""
    f, cx, cy, k = raw["params"][:4]
    p = scene_cam["params"]
    fx, fy, cxs, cys = p[0], p[1], p[2], p[3]
    u, v = _grid(raw["width"], raw["height"])
    pts = np.stack([u, v], -1).reshape(-1, 1, 2)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float64)
    und = cv2.undistortPoints(pts, K, np.array([k, 0, 0, 0], np.float64)).reshape(raw["height"], raw["width"], 2)
    return ((und[..., 0] * fx + cxs - 0.5).astype(np.float32), (und[..., 1] * fy + cys - 0.5).astype(np.float32))


def remap(img: np.ndarray, maps, nearest: bool = False) -> np.ndarray:
    return cv2.remap(img, maps[0], maps[1], cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)

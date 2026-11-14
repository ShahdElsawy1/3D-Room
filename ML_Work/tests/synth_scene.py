"""Tiny synthetic 'room' (floor + 3 textured walls) rendered with plane homographies from a moving camera.
Used to smoke-test the pose stages end to end without a real video. Known intrinsics, tiny/no distortion."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _texture(rng, size=640):
    img = np.zeros((size, size, 3), np.float32)
    for s in (8, 16, 40, 100):
        n = rng.random((size // s + 2, size // s + 2, 3)).astype(np.float32)
        img += cv2.resize(n, (size, size), interpolation=cv2.INTER_CUBIC) * (s ** 0.5)
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)
    for _ in range(120):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        x, y = rng.integers(0, size, 2)
        if rng.random() < 0.5:
            cv2.circle(img, (int(x), int(y)), int(rng.integers(4, 25)), c, -1)
        else:
            cv2.rectangle(img, (int(x), int(y)), (int(x + rng.integers(5, 40)), int(y + rng.integers(5, 40))), c, -1)
    return img


def make_frames(out_dir, n=30, size_wh=(640, 480), f=520.0, seed=0) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    W, H = size_wh
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float64)
    planes = [  # origin, u-axis, v-axis
        ((-2, 1, 1), (4, 0, 0), (0, 0, 4)),          # floor (y down => y=+1)
        ((-2, -1.5, 5), (4, 0, 0), (0, 2.5, 0)),     # back wall
        ((-2, -1.5, 1), (0, 0, 4), (0, 2.5, 0)),     # left wall
        ((2, -1.5, 1), (0, 0, 4), (0, 2.5, 0)),      # right wall
        ((-2, -1.5, 1), (4, 0, 0), (0, 0, 4)),       # ceiling
    ]
    tex = [_texture(rng) for _ in planes]
    Ts = 640
    poses = {}
    for i in range(n):
        a = i / max(n - 1, 1)
        cx = -0.6 + 1.2 * a
        yaw = np.radians(12 * np.sin(a * np.pi * 2))
        R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        C = np.array([cx, 0.1 * np.sin(a * 6), 0.0])
        t = -R @ C
        canvas = np.full((H, W, 3), 30, np.uint8)
        order = sorted(range(len(planes)), key=lambda k: -np.linalg.norm(
            R @ (np.array(planes[k][0]) + 0.5 * np.array(planes[k][1]) + 0.5 * np.array(planes[k][2])) + t))
        for k in order:
            O, U, V = (np.array(v, float) for v in planes[k])
            M = np.stack([R @ U / Ts, R @ V / Ts, R @ O + t], axis=1)
            Hm = K @ M
            warped = cv2.warpPerspective(tex[k], Hm, (W, H), flags=cv2.INTER_LINEAR)
            mask = cv2.warpPerspective(np.full((Ts, Ts), 255, np.uint8), Hm, (W, H), flags=cv2.INTER_NEAREST)
            canvas[mask > 0] = warped[mask > 0]
        name = f"{i + 1:06d}.jpg"
        cv2.imwrite(str(out / name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        poses[name] = (R, t)
    return {"K": K, "poses": poses, "size_wh": size_wh}

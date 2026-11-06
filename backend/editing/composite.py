"""The guarantee of spec §1: nothing outside the mask changes.

Neither model takes a binary mask, and both cards say edits may leak outside the requested region. So the model's
output is only ever used through this composite:  out = orig * (1 - a) + edit * a,  with a == 0 everywhere outside
the dilated mask. Those pixels are copied from the original array, so they are bit-identical by construction, and
`assert_unchanged_outside` re-checks it for every frame that is written.
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import Box


class CompositeError(RuntimeError):
    pass


def alpha_map(region: np.ndarray, feather_px: int) -> np.ndarray:
    """float32 HxW in [0, 1]: 0 outside `region`, ramping to 1 over `feather_px` pixels *inside* it."""
    if feather_px <= 0:
        return region.astype(np.float32)
    dist = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)  # distance to the nearest outside pixel
    return np.clip(dist / float(feather_px), 0.0, 1.0).astype(np.float32)


def composite(orig: np.ndarray, edit_crop: np.ndarray, box: Box, alpha: np.ndarray) -> np.ndarray:
    """Paste `edit_crop` (box-sized RGB uint8) into a copy of `orig` through `alpha`."""
    if edit_crop.shape[:2] != (box.h, box.w):
        raise CompositeError(f"edit is {edit_crop.shape[1]}x{edit_crop.shape[0]}, box is {box.w}x{box.h}")
    outside = alpha.copy()
    outside[box.slices] = 0
    if outside.any():
        raise CompositeError("mask region extends outside the crop box")
    out = orig.copy()
    a = alpha[box.slices]
    sel = a > 0
    o = orig[box.slices].astype(np.float32)
    e = edit_crop.astype(np.float32)
    blend = np.clip(np.rint(o * (1 - a[..., None]) + e * a[..., None]), 0, 255).astype(np.uint8)
    region = out[box.slices]
    region[sel] = blend[sel]
    return out


def changed_outside(orig: np.ndarray, out: np.ndarray, region: np.ndarray) -> int:
    """Number of pixels outside `region` whose value differs (must be 0)."""
    diff = (orig != out).any(axis=-1)
    return int((diff & ~region).sum())


def assert_unchanged_outside(orig: np.ndarray, out: np.ndarray, region: np.ndarray, name: str = "") -> None:
    n = changed_outside(orig, out, region)
    if n:
        raise CompositeError(f"{name}: {n} pixels outside the mask changed")

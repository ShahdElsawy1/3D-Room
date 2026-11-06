"""Masks, crop boxes and the resize to / from the model's resolution.

Frames keep their Week 1 size (e.g. 960x540). Only a crop around the object goes to the model, resized so both
sides are multiples of ALIGN, and the result is resized back to the crop's exact size before compositing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ALIGN = 32  # 16x autoencoder x 2x2 patches; neither model card states a rule, 32 is the safe choice (research §3)


class MaskError(ValueError):
    pass


@dataclass(frozen=True)
class Box:
    x0: int
    y0: int
    x1: int  # exclusive
    y1: int  # exclusive

    @property
    def w(self) -> int: return self.x1 - self.x0
    @property
    def h(self) -> int: return self.y1 - self.y0
    @property
    def slices(self): return slice(self.y0, self.y1), slice(self.x0, self.x1)
    def as_list(self) -> list[int]: return [self.x0, self.y0, self.x1, self.y1]

    def grid1000(self, img_w: int, img_h: int) -> list[int]:
        """FLUX Klein's box format: [top, left, bottom, right] on a 0-1000 grid of the image passed to the model."""
        return [round(self.y0 / img_h * 1000), round(self.x0 / img_w * 1000),
                round(self.y1 / img_h * 1000), round(self.x1 / img_w * 1000)]


def load_image(path: Path) -> np.ndarray:
    """RGB uint8 HxWx3, decoded exactly like the 3DGS trainer decodes the frame."""
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    """Person A's mask: 8-bit single channel, 0 = background, 255 = object, same W x H as the frame -> bool HxW."""
    m = Image.open(path)
    if m.size != size:
        raise MaskError(f"{path.name}: mask is {m.size[0]}x{m.size[1]}, frame is {size[0]}x{size[1]}")
    m = np.asarray(m.convert("L")) > 127
    if not m.any():
        raise MaskError(f"{path.name}: mask is empty")
    return m


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def bbox(mask: np.ndarray) -> Box:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise MaskError("empty mask")
    return Box(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def crop_box(region: np.ndarray, margin: float = 0.15, min_side: int = 96) -> Box:
    """Box around `region` (the dilated mask) + margin of context, at least min_side, clamped to the image."""
    H, W = region.shape
    b = bbox(region)
    pad = int(round(margin * max(b.w, b.h)))
    x0, y0, x1, y1 = b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad
    for lo, hi, limit, i in ((x0, x1, W, 0), (y0, y1, H, 1)):
        short = min_side - (hi - lo)
        if short > 0:
            lo, hi = lo - short // 2, hi + (short - short // 2)
        lo, hi = max(0, lo), min(limit, hi)
        if i == 0:
            x0, x1 = lo, hi
        else:
            y0, y1 = lo, hi
    return Box(x0, y0, x1, y1)


def model_size(w: int, h: int, long_side: int, align: int = ALIGN) -> tuple[int, int]:
    """(W, H) for the model: long side ~= long_side, both multiples of `align`, aspect kept as well as possible."""
    s = long_side / max(w, h)
    return (max(align, int(round(w * s / align)) * align), max(align, int(round(h * s / align)) * align))


def to_model(crop: np.ndarray, size: tuple[int, int]) -> Image.Image:
    return Image.fromarray(crop).resize(size, Image.LANCZOS)


def from_model(img: Image.Image, box: Box) -> np.ndarray:
    """Model output -> RGB uint8 array of exactly the crop's size."""
    return np.asarray(img.convert("RGB").resize((box.w, box.h), Image.LANCZOS))

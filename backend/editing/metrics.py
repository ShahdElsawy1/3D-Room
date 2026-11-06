"""Measurements for the spec's known problems (§4.3): editing outside the mask, colour drift across frames."""
from __future__ import annotations

import math

import cv2
import numpy as np


def psnr(a: np.ndarray, b: np.ndarray, where: np.ndarray) -> float | None:
    """PSNR (dB) between two uint8 images over the pixels in `where`. inf = identical, None = no pixels."""
    if not where.any():
        return None
    d = (a[where].astype(np.float64) - b[where].astype(np.float64)) ** 2
    mse = float(d.mean())
    return math.inf if mse == 0 else round(10 * math.log10(255.0 ** 2 / mse), 2)


def raw_leak_psnr(orig_crop: np.ndarray, raw_crop: np.ndarray, region_crop: np.ndarray) -> float | None:
    """How much the *raw* model output changed the crop outside the mask (before compositing). Higher = less leak."""
    return psnr(orig_crop, raw_crop, ~region_crop)


def mean_lab(img: np.ndarray, where: np.ndarray) -> list[float]:
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)  # L 0..100, a/b about -128..127
    return [round(float(v), 2) for v in lab[where].mean(axis=0)]


def delta_e(lab1: list[float], lab2: list[float]) -> float:
    """CIE76 colour difference. ~2.3 is a just-noticeable difference."""
    return round(float(np.linalg.norm(np.subtract(lab1, lab2))), 2)


def json_num(x):
    """inf is not valid JSON."""
    return "inf" if isinstance(x, float) and math.isinf(x) else x

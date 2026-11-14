"""Mask files and `masks/index.json` - the contract with Person B (backend + diffusion) and Person C (UI).

    <project>/edits/<edit_id>/masks/<frame_stem>.png   8-bit single channel, 0 = background, 255 = object,
                                                        exactly the size of frames/images/<frame_name>
    <project>/edits/<edit_id>/masks/index.json         one entry per frame of frames/images (same names)
        {"ref_frame": "000012.jpg", "method": "fused", "frames": [
            {"name": "000012.jpg", "present": true, "source": "click", "score": 0.97, "area_px": 41230}, ...]}
    source: click (reference) | propagated (SAM 2 video) | lifted (3D reprojection) | reid | user (fallback marking)
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

SOURCES = ("click", "propagated", "lifted", "reid", "user")


def edit_dir(project: Path, edit_id: str) -> Path:
    return Path(project) / "edits" / edit_id


def frame_names(project: Path) -> list[str]:
    d = Path(project) / "frames" / "images"
    return sorted(p.name for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def save_mask(path: Path, mask: np.ndarray) -> int:
    """bool/0-1/0-255 array -> PNG 0/255 uint8. Returns the object area in pixels."""
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), m):
        raise IOError(f"could not write {path}")
    return int((m > 0).sum())


def load_mask(path: Path) -> np.ndarray | None:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[..., 0]
    return m > 127


def read_index(path: Path) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {"ref_frame": None, "method": None, "frames": []}


def write_index(path: Path, idx: dict, all_names: list[str]) -> dict:
    """Write index.json with exactly one entry per frame (missing ones = not present), in frame order."""
    by = {f["name"]: f for f in idx.get("frames", [])}
    idx = dict(idx)
    idx["frames"] = [by.get(n, {"name": n, "present": False, "source": None, "score": 0.0, "area_px": 0})
                     for n in all_names]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(idx, indent=2))
    return idx


def entry(name: str, present: bool, source: str | None, score: float, area_px: int) -> dict:
    return {"name": name, "present": bool(present), "source": source if present else (source or None),
            "score": round(float(score), 4), "area_px": int(area_px)}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def clean_mask(m: np.ndarray, min_area_frac: float = 2e-4, keep_largest: bool = False, hole_frac: float = 5e-4) -> np.ndarray:
    """Drop specks, fill small holes. Diffusion crops come from the mask, so ragged noise would hurt."""
    m = (np.asarray(m) > 0).astype(np.uint8)
    h, w = m.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    areas = stats[1:, cv2.CC_STAT_AREA]
    for i, a in enumerate(areas, start=1):
        if a >= min_area_frac * h * w:
            keep[lab == i] = 1
    if keep_largest and len(areas):
        keep = (lab == 1 + int(np.argmax(areas))).astype(np.uint8)
    inv = (1 - keep).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for i in range(1, n):
        x, y, ww, hh, a = stats[i]
        touches = x == 0 or y == 0 or x + ww == w or y + hh == h
        if not touches and a <= hole_frac * h * w:
            keep[lab == i] = 1
    return keep.astype(bool)


def validate(project: Path, edit_id: str, masks_sub: str = "masks") -> list[str]:
    """Contract check used by tests and before handing masks to Person B. Empty list = OK."""
    d = edit_dir(project, edit_id) / masks_sub
    names = frame_names(project)
    problems = []
    idx_path = d / "index.json"
    if not idx_path.exists():
        return [f"{idx_path} missing"]
    idx = read_index(idx_path)
    got = [f["name"] for f in idx["frames"]]
    if got != names:
        problems.append(f"index.json frames != frames/images ({len(got)} vs {len(names)} entries / order)")
    for f in idx["frames"]:
        p = d / (Path(f["name"]).stem + ".png")
        if f["present"]:
            m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            img = cv2.imread(str(Path(project) / "frames" / "images" / f["name"]))
            if m is None:
                problems.append(f"{f['name']}: present but {p.name} missing")
            else:
                if m.ndim != 2 or m.dtype != np.uint8:
                    problems.append(f"{p.name}: must be 8-bit single channel")
                elif img is not None and m.shape != img.shape[:2]:
                    problems.append(f"{p.name}: size {m.shape[::-1]} != frame {img.shape[1::-1]}")
                elif set(np.unique(m)) - {0, 255}:
                    problems.append(f"{p.name}: values other than 0/255")
            if f.get("source") not in SOURCES:
                problems.append(f"{f['name']}: bad source {f.get('source')!r}")
        elif p.exists():
            problems.append(f"{f['name']}: not present but {p.name} exists")
    return problems

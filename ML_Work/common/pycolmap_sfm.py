"""Structure-from-Motion through pycolmap >= 4 (`pip install pycolmap-cuda12==4.2.1`).

Why this exists: the standalone `glomap` + conda `colmap` route kept breaking on Kaggle (libfaiss ABI, OpenGL SIFT on a
headless box, RetriangulateTracks). pycolmap 4.x ships GLOMAP as `pycolmap.global_mapping`, GPU SIFT through CUDA (no
OpenGL) and everything in one pip wheel. The calls below are the ones that ran end-to-end in the team's Kaggle notebook
(720 frames: 713 registered, 1.10 px reprojection error).

IMPORTANT - the gsplat examples need a *different* package that is also called `pycolmap` (a fork with SceneManager).
Installing both into one environment makes one of them disappear. So the pycolmap 4 wheel is installed into its own
folder (`pip install --target DIR ...`) and only the pose stage, which is its own process, puts it on sys.path:

    export W1_PYCOLMAP_DIR=/kaggle/working/tools/pycolmap4

Everything that does not need pycolmap itself (pair lists, pose statistics, pruning decisions, quality verdict) is plain
numpy so it can be unit-tested without the library.
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

from .colmap_io import SparseModel

PAIR_ID_MOD = 2147483647


# ----------------------------------------------------------------------------- import / capability
def import_pycolmap():
    """Import pycolmap, preferring the isolated folder in $W1_PYCOLMAP_DIR (see module docstring)."""
    d = os.environ.get("W1_PYCOLMAP_DIR", "")
    if d and Path(d).is_dir() and d not in sys.path:
        sys.path.insert(0, d)
    import pycolmap  # noqa: WPS433
    return pycolmap


def available() -> tuple[bool, str]:
    """(usable, reason). Usable = pycolmap with the built-in global mapper (>= 4.x), not the gsplat fork."""
    try:
        pc = import_pycolmap()
    except Exception as e:                                   # ImportError, or a broken wheel
        return False, f"pycolmap not importable ({type(e).__name__}: {e})"
    if not hasattr(pc, "global_mapping"):
        return False, f"pycolmap {getattr(pc, '__version__', '?')} has no global_mapping (a fork or an old version)"
    return True, f"pycolmap {getattr(pc, '__version__', '?')} (cuda={getattr(pc, 'has_cuda', False)})"


def frame_index(name: str) -> int:
    return int(re.findall(r"\d+", name)[-1])


# ----------------------------------------------------------------------------- features
def focal_factor(width: int, height: int) -> float:
    """Prior for f / longer-side from the aspect ratio (phone video: 16:9 -> ~0.73, 4:3 -> ~0.89).
    The notebook's refined focal came out at 0.726 for a 9:16 phone video, i.e. the prior is close."""
    return 0.89 if max(width, height) / min(width, height) < 1.5 else 0.73


def extract_features(pc, db: Path, imgs: Path, *, ff: float | None, max_image: int, max_features: int,
                     camera_model: str, use_gpu: bool) -> dict:
    import cv2
    names = sorted(p.name for p in imgs.glob("*.jpg"))
    h, w = cv2.imread(str(imgs / names[0])).shape[:2]
    if ff is None:
        ff = focal_factor(w, h)
    reader = pc.ImageReaderOptions()
    reader.camera_model = camera_model
    if ff:
        reader.default_focal_length_factor = float(ff)
    ext = pc.FeatureExtractionOptions()
    ext.max_image_size = max_image
    ext.sift.max_num_features = max_features
    if hasattr(ext, "use_gpu"):
        ext.use_gpu = bool(use_gpu)
    if db.exists():
        db.unlink()
    pc.extract_features(str(db), str(imgs), camera_mode=pc.CameraMode.SINGLE, reader_options=reader,
                        extraction_options=ext)
    con = sqlite3.connect(db)
    try:
        if ff:      # flag the focal length as a prior: the global mapper behaves worse without it
            try:
                con.execute("update cameras set prior_focal_length=1")
                con.commit()
            except sqlite3.OperationalError as e:
                print(f"[pycolmap] WARNING could not flag the focal prior: {e}", flush=True)
        kp = con.execute("select avg(rows), min(rows), max(rows) from keypoints").fetchone()
    finally:
        con.close()
    return {"focal_factor": float(ff or 0), "image_wh": [w, h], "keypoints_avg_min_max": [round(x) for x in kp]}


# ----------------------------------------------------------------------------- matching
def hybrid_pairs(n: int, window: int, loop_stride: int) -> list[tuple[int, int]]:
    """Sequential window (every frame vs the next `window`) + all pairs among every `loop_stride`-th 'key' frame.
    The second part is a cheap stand-in for loop detection (COLMAP's needs a vocabulary tree file): a room video
    comes back to places it has already seen, and these pairs let the global mapper tie the ends together."""
    pairs = {(i, j) for i in range(n) for j in range(i + 1, min(n, i + window + 1))}
    if loop_stride and loop_stride > 0:
        keys = list(range(0, n, loop_stride))
        pairs |= {(keys[a], keys[b]) for a in range(len(keys)) for b in range(a + 1, len(keys))}
    return sorted(pairs)


def match(pc, db: Path, imgs: Path, *, mode: str, window: int, loop_stride: int, work: Path, use_gpu: bool) -> dict:
    names = sorted(p.name for p in imgs.glob("*.jpg"))
    n = len(names)
    con = sqlite3.connect(db)
    con.execute("delete from matches")
    con.execute("delete from two_view_geometries")
    con.commit()
    con.close()
    mopt = None
    if hasattr(pc, "FeatureMatchingOptions"):
        mopt = pc.FeatureMatchingOptions()
        if hasattr(mopt, "use_gpu"):
            mopt.use_gpu = bool(use_gpu)
    kw = {"matching_options": mopt} if mopt is not None else {}
    t0 = time.perf_counter()
    if mode == "exhaustive":
        pc.match_exhaustive(str(db), **kw)
        n_pairs = n * (n - 1) // 2
    elif mode == "sequential":
        po = pc.SequentialPairingOptions()
        po.overlap = window
        po.quadratic_overlap = True
        pc.match_sequential(str(db), pairing_options=po, **kw)
        n_pairs = None
    else:                                                    # hybrid
        pairs = hybrid_pairs(n, window, loop_stride)
        work.mkdir(parents=True, exist_ok=True)
        lst = work / "pairs.txt"
        lst.write_text("".join(f"{names[i]} {names[j]}\n" for i, j in pairs))
        io = pc.ImportedPairingOptions()
        io.match_list_path = str(lst)
        pc.match_image_pairs(str(db), pairing_options=io, **kw)
        n_pairs = len(pairs)
    seconds = time.perf_counter() - t0
    return {"mode": mode, "pairs": n_pairs, "exhaustive_pairs": n * (n - 1) // 2, "seconds": round(seconds, 1),
            **matching_stats(db)}


def matching_stats(db: Path, min_inliers: int = 15) -> dict:
    """Verified pairs and frames that have no verified neighbour within +-2 frames (weak links in the chain)."""
    con = sqlite3.connect(db)
    try:
        nm = dict(con.execute("select image_id, name from images"))
        ver = [(p // PAIR_ID_MOD, p % PAIR_ID_MOD)
               for (p,) in con.execute("select pair_id from two_view_geometries where rows >= ?", (min_inliers,))]
    finally:
        con.close()
    deg = {frame_index(v): 0 for v in nm.values()}
    for a, b in ver:
        if a in nm and b in nm:
            ia, ib = frame_index(nm[a]), frame_index(nm[b])
            if abs(ia - ib) <= 2:
                deg[ia] += 1
                deg[ib] += 1
    weak = sorted(i for i, d in deg.items() if d == 0)
    return {"verified_pairs": len(ver), "weak_frames": weak[:50], "n_weak_frames": len(weak)}


# ----------------------------------------------------------------------------- pose statistics (numpy only)
def rotation_steps(model: SparseModel) -> tuple[np.ndarray, np.ndarray]:
    """Rotation angle (deg) between consecutive frame numbers. Returns (first frame of each step, degrees)."""
    by = {frame_index(im.name): im for im in model.images.values()}
    ids = sorted(by)
    at, rot = [], []
    for a, b in zip(ids[:-1], ids[1:]):
        if b - a != 1:
            continue
        c = (np.trace(by[a].R @ by[b].R.T) - 1) / 2
        rot.append(float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
        at.append(a)
    return np.array(at, dtype=int), np.array(rot, dtype=float)


def prune_candidates(model: SparseModel, jump_deg: float, far_factor: float = 2.0) -> dict:
    """Frames to drop: both ends of a consecutive-frame rotation jump above `jump_deg`, and cameras lying far outside
    the point cloud (> far_factor x its 90th-percentile radius). Frames are returned by file name."""
    by_idx = {frame_index(im.name): im.name for im in model.images.values()}
    at, rot = rotation_steps(model)
    bad = set()
    for a, r in zip(at, rot):
        if r > jump_deg:
            bad |= {by_idx[int(a)], by_idx[int(a) + 1]}
    far = []
    pts = model.points.xyz
    if len(pts) >= 10:
        ctr = np.median(pts, axis=0)
        rad = np.percentile(np.linalg.norm(pts - ctr, axis=1), 90)
        for im in model.images.values():
            c = -im.R.T @ im.tvec
            if np.linalg.norm(c - ctr) > far_factor * rad:
                far.append(im.name)
    return {"names": sorted(bad | set(far)), "jump_frames": sorted(bad), "far_frames": sorted(far),
            "median_rot_deg": float(np.median(rot)) if len(rot) else 0.0,
            "max_rot_deg": float(rot.max()) if len(rot) else 0.0}


def assess(model: SparseModel, n_total: int, *, min_registered: float = 0.9, max_error: float = 1.5,
           max_jump_fraction: float = 0.03) -> dict:
    """Is this reconstruction trustworthy? Same criteria as the notebook's verdict + registered fraction."""
    at, rot = rotation_steps(model)
    thr = max(25.0, 4 * float(np.median(rot))) if len(rot) else 25.0
    jumps = int((rot > thr).sum()) if len(rot) else 0
    err = float(np.mean(model.points.error)) if len(model.points.xyz) else float("inf")
    reg = len(model.images)
    reasons = []
    if reg < min_registered * n_total:
        reasons.append(f"only {reg}/{n_total} frames registered")
    if err > max_error:
        reasons.append(f"mean reprojection error {err:.2f}px > {max_error}")
    if len(rot) and jumps > max_jump_fraction * len(rot):
        reasons.append(f"{jumps} of {len(rot)} consecutive-frame rotation jumps > {thr:.0f} deg")
    return {"ok": not reasons, "reasons": reasons, "registered": reg, "total": n_total,
            "mean_reproj_error_px": None if err == float("inf") else round(err, 3), "rotation_jumps": jumps,
            "jump_threshold_deg": round(thr, 1), "points3D": int(len(model.points.xyz))}


# ----------------------------------------------------------------------------- mapping / pruning / undistortion
def map_model(pc, db: Path, imgs: Path, out: Path, mapper: str) -> tuple[float, str]:
    """Run `global` (GLOMAP) or `incremental` mapping, write the biggest model to out/0. Returns (seconds, dir)."""
    import shutil
    shutil.rmtree(out, ignore_errors=True)
    raw = out / "_raw"
    raw.mkdir(parents=True)
    t0 = time.perf_counter()
    recs = (pc.global_mapping if mapper == "global" else pc.incremental_mapping)(str(db), str(imgs), str(raw))
    seconds = time.perf_counter() - t0
    if not recs:
        raise RuntimeError(f"{mapper} mapping produced no model (too few verified pairs? see the matching stats)")
    best = max(recs.values(), key=lambda r: r.num_images())
    (out / "0").mkdir(parents=True)
    best.write(str(out / "0"))
    return seconds, str(out / "0")


def prune_with_pycolmap(pc, model_dir: Path, out_dir: Path, names: list[str]) -> int:
    """Deregister frames `names` from the model in model_dir and write the result to out_dir. Returns #remaining."""
    rec = pc.Reconstruction(str(model_dir))
    by_name = {im.name: im for im in rec.images.values()}
    for n in names:
        if n in by_name:
            rec.deregister_frame(by_name[n].frame_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec.write(str(out_dir))
    return rec.num_images()


def undistort(pc, model_dir: Path, imgs: Path, out: Path, max_image_size: int = -1) -> None:
    """Undistort to PINHOLE: writes out/images/*.jpg and out/sparse (same layout the CLI's image_undistorter makes)."""
    import shutil
    shutil.rmtree(out, ignore_errors=True)
    uo = pc.UndistortCameraOptions()
    if max_image_size and max_image_size > 0:
        uo.max_image_size = max_image_size
    pc.undistort_images(str(out), str(model_dir), str(imgs), undistort_options=uo)

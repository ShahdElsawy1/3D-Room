"""Chunked-VGGT utilities that need no torch: chunk planning and Sim(3) chunk merging.

VGGT's memory grows with the number of frames, so long videos are processed in
overlapping chunks.  Each chunk lives in its own arbitrary coordinate frame/scale.
Because the *same image* appears in two consecutive chunks, its pixel-aligned depth maps
give exact point correspondences, so the merge is a robust Sim(3) fit on those points
(far more stable than aligning a handful of near-collinear camera centres).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import apply_sim3_to_w2c, robust_umeyama, unproject_depth


def plan_chunks(n: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """[(start, end), ...] covering 0..n with `overlap` shared frames between neighbours."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    if n <= size:
        return [(0, n)]
    chunks, s = [], 0
    while True:
        e = min(s + size, n)
        chunks.append((s, e))
        if e == n:
            break
        s = e - overlap
    # a tiny trailing chunk is a poor VGGT input: replace it with a full-size final chunk
    if len(chunks) > 1 and chunks[-1][1] - chunks[-1][0] < size // 2:
        chunks[-1] = (max(0, n - size), n)
        if chunks[-1][0] <= chunks[-2][0]:      # degenerate: just drop the redundant chunk
            chunks.pop()
            chunks[-1] = (chunks[-1][0], n)
    return chunks


@dataclass
class Chunk:
    start: int
    R: np.ndarray       # (S,3,3) world->cam rotation, chunk-local world
    t: np.ndarray       # (S,3)
    K: np.ndarray       # (S,3,3) at the VGGT working resolution
    depth: np.ndarray   # (S,H,W) z-depth
    conf: np.ndarray    # (S,H,W)

    @property
    def end(self) -> int:
        return self.start + len(self.R)


def merge_chunks(chunks: list[Chunk], n_samples: int = 30000, seed: int = 0):
    """Chain-align chunks into the first chunk's world.

    Returns (R, t, K, depth, conf, diagnostics) with arrays indexed by global frame.
    """
    rng = np.random.default_rng(seed)
    N = chunks[-1].end
    S0 = chunks[0]
    H, W = S0.depth.shape[1:]
    R = np.zeros((N, 3, 3)); t = np.zeros((N, 3)); K = np.zeros((N, 3, 3))
    depth = np.zeros((N, H, W), np.float32); conf = np.zeros((N, H, W), np.float32)
    filled = np.zeros(N, bool)
    diag = []

    def put(gi, Rg, tg, Kg, dg, cg):
        R[gi], t[gi], K[gi], depth[gi], conf[gi] = Rg, tg, Kg, dg, cg
        filled[gi] = True

    for j in range(len(S0.R)):
        put(S0.start + j, S0.R[j], S0.t[j], S0.K[j], S0.depth[j], S0.conf[j])

    for ci in range(1, len(chunks)):
        c, prev = chunks[ci], chunks[ci - 1]
        shared = list(range(c.start, prev.end))
        if not shared:
            raise ValueError(f"chunk {ci} does not overlap chunk {ci-1}")
        per = max(200, n_samples // len(shared))
        src, dst = [], []
        for gi in shared:
            j = gi - c.start
            P_cur = unproject_depth(c.depth[j].astype(np.float64), c.K[j], c.R[j], c.t[j]).reshape(-1, 3)
            P_prev = unproject_depth(depth[gi].astype(np.float64), K[gi], R[gi], t[gi]).reshape(-1, 3)
            w = np.minimum(c.conf[j], conf[gi]).ravel()
            ok = np.isfinite(P_cur).all(1) & np.isfinite(P_prev).all(1) & (c.depth[j].ravel() > 0) & (depth[gi].ravel() > 0)
            if ok.sum() < 50:
                continue
            thr = np.percentile(w[ok], 50)
            idx = np.flatnonzero(ok & (w >= thr))
            take = rng.choice(idx, size=min(len(idx), per), replace=False)
            src.append(P_cur[take]); dst.append(P_prev[take])
        if not src:
            raise RuntimeError(f"no valid correspondences between chunk {ci-1} and {ci}")
        src, dst = np.concatenate(src), np.concatenate(dst)
        s, Rm, tm, rmse = robust_umeyama(src, dst)
        med_depth = float(np.median(depth[shared[0]][depth[shared[0]] > 0]))
        half = len(shared) // 2
        for j in range(len(c.R)):
            gi = c.start + j
            if filled[gi] and j < half:        # first half of the overlap: keep the earlier chunk
                continue
            Rn, tn = apply_sim3_to_w2c(c.R[j], c.t[j], s, Rm, tm)
            put(gi, Rn, tn, c.K[j], (c.depth[j] * s).astype(np.float32), c.conf[j])
        diag.append({"chunk": ci, "shared_frames": len(shared), "scale": float(s),
                     "rmse": rmse, "rmse_over_median_depth": rmse / max(med_depth, 1e-9)})
    if not filled.all():
        raise RuntimeError("some frames were not covered by any chunk")
    return R, t, K, depth, conf, diag

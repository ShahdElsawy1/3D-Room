"""Synthetic check of the chunked-VGGT merge: render depth maps of a box room from known
cameras, give every chunk its own random Sim(3) 'gauge' (as VGGT would), merge, and verify the
merged trajectory equals the ground truth up to ONE global Sim(3)."""
import numpy as np

from common.geometry import apply_sim3_to_w2c, camera_center, qvec2rotmat, umeyama
from common.vggt_merge import Chunk, merge_chunks, plan_chunks

H, W = 36, 64
K = np.array([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]])
HALF = 5.0


def look_at(center, target, up=(0, -1, 0)):
    z = target - center; z /= np.linalg.norm(z)
    x = np.cross(np.array(up, float), z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_c2w = np.stack([x, y, z], 1)
    R = R_c2w.T
    return R, -R @ center


def render_depth(R, t):
    C = -R.T @ t
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    d_w = d_cam @ R                                  # R^T d  (row-vector form)
    with np.errstate(divide="ignore"):
        t_pos = (HALF - C) / d_w
        t_neg = (-HALF - C) / d_w
    cand = np.where(d_w > 0, t_pos, t_neg)
    return cand.min(-1)                               # z-depth since d_cam.z == 1


def make_chunks(n, size, overlap, seed=0, noise=0.0005):
    rng = np.random.default_rng(seed)
    poses = []
    for i in range(n):
        ang = 2 * np.pi * i / n
        center = np.array([2.5 * np.cos(ang), 0.3 * np.sin(3 * ang), 2.5 * np.sin(ang)])
        target = np.array([1.5 * np.cos(ang + 1.2), 0.0, 1.5 * np.sin(ang + 1.2)])
        poses.append(look_at(center, target))
    chunks = []
    for s, e in plan_chunks(n, size, overlap):
        q = rng.normal(size=4)
        Rg, tg, sg = qvec2rotmat(q / np.linalg.norm(q)), rng.normal(size=3) * 3, rng.uniform(0.3, 3.0)
        Rs, ts, Ks, Ds, Cs = [], [], [], [], []
        for i in range(s, e):
            R, t = poses[i]
            Rn, tn = apply_sim3_to_w2c(R, t, sg, Rg, tg)
            Rs.append(Rn); ts.append(tn); Ks.append(K)
            Ds.append(render_depth(R, t) * sg * (1 + noise * rng.normal(size=(H, W))))
            Cs.append(rng.uniform(1, 5, size=(H, W)))
        chunks.append(Chunk(s, np.array(Rs), np.array(ts), np.array(Ks), np.array(Ds), np.array(Cs)))
    return poses, chunks


def test_plan_chunks_properties():
    for n in (5, 32, 33, 50, 100, 137):
        plan = plan_chunks(n, 32, 8)
        assert plan[0][0] == 0 and plan[-1][1] == n
        for (a, b), (c, d) in zip(plan, plan[1:]):
            assert c < b and c > a                      # consecutive chunks overlap and advance
            assert b - c >= 1
        assert all(b - a <= 32 for a, b in plan)
        if len(plan) > 1:
            assert plan[-1][1] - plan[-1][0] >= 16      # no tiny tail chunk


def test_merge_recovers_trajectory_up_to_global_sim3():
    poses, chunks = make_chunks(n=70, size=24, overlap=8)
    assert len(chunks) >= 3
    R, t, Kk, depth, conf, diag = merge_chunks(chunks)
    gt = np.array([camera_center(*p) for p in poses])
    est = np.array([camera_center(R[i], t[i]) for i in range(len(poses))])
    s, Ra, ta = umeyama(est, gt)
    err = np.linalg.norm(s * est @ Ra.T + ta - gt, axis=1)
    extent = np.linalg.norm(gt.max(0) - gt.min(0))
    assert err.max() / extent < 5e-3, err.max() / extent
    assert all(d["rmse_over_median_depth"] < 0.01 for d in diag)
    # rotations too: relative rotation between consecutive cameras must match ground truth
    for i in range(len(poses) - 1):
        rel_gt = poses[i + 1][0] @ poses[i][0].T
        rel = R[i + 1] @ R[i].T
        assert np.allclose(rel, rel_gt, atol=2e-2)

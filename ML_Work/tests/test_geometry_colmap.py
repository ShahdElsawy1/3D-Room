import numpy as np

from common.colmap_io import Camera, Image, Points3D, SparseModel, read_model, write_model
from common.geometry import (apply_sim3_to_w2c, qvec2rotmat, robust_umeyama, rotmat2qvec,
                             umeyama, unproject_depth, project)


def rand_rot(rng):
    q = rng.normal(size=4)
    return qvec2rotmat(q / np.linalg.norm(q))


def test_quaternion_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        R = rand_rot(rng)
        assert np.allclose(qvec2rotmat(rotmat2qvec(R)), R, atol=1e-9)


def test_umeyama_recovers_sim3():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(500, 3))
    R, t, s = rand_rot(rng), rng.normal(size=3), 2.7
    dst = s * src @ R.T + t
    s2, R2, t2 = umeyama(src, dst)
    assert abs(s2 - s) < 1e-9 and np.allclose(R2, R) and np.allclose(t2, t)


def test_robust_umeyama_ignores_outliers():
    rng = np.random.default_rng(2)
    src = rng.normal(size=(2000, 3))
    R, t, s = rand_rot(rng), rng.normal(size=3), 0.5
    dst = s * src @ R.T + t
    dst[:300] += rng.normal(scale=5, size=(300, 3))      # 15% gross outliers
    s2, R2, t2, rmse = robust_umeyama(src, dst)
    assert abs(s2 - s) < 1e-3 and np.allclose(R2, R, atol=1e-3) and rmse < 1e-2


def test_apply_sim3_keeps_projection_consistent():
    """A world point must project to the same pixel before/after re-gauging the world."""
    rng = np.random.default_rng(3)
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1.0]])
    Rw, tw = rand_rot(rng), np.array([0.1, -0.2, 3.0])
    X = np.array([[0.3, 0.2, 1.0]])
    s, R, t = 1.8, rand_rot(rng), rng.normal(size=3)
    X2 = s * X @ R.T + t
    Rn, tn = apply_sim3_to_w2c(Rw, tw, s, R, t)
    uv1, z1 = project(X, K, Rw, tw)
    uv2, z2 = project(X2, K, Rn, tn)
    assert np.allclose(uv1, uv2, atol=1e-8) and np.allclose(z1 * s, z2, atol=1e-8)


def test_unproject_project_roundtrip():
    K = np.array([[400, 0, 160], [0, 400, 120], [0, 0, 1.0]])
    rng = np.random.default_rng(4)
    R, t = rand_rot(rng), np.array([0.5, 0.1, -0.3])
    depth = rng.uniform(1, 4, size=(24, 32))
    P = unproject_depth(depth, K, R, t)
    uv, z = project(P.reshape(-1, 3), K, R, t)
    uu, vv = np.meshgrid(np.arange(32) + 0.5, np.arange(24) + 0.5)
    assert np.allclose(uv[:, 0], uu.ravel(), atol=1e-6) and np.allclose(uv[:, 1], vv.ravel(), atol=1e-6)
    assert np.allclose(z, depth.ravel())


def make_model(n=5):
    rng = np.random.default_rng(5)
    cams = {1: Camera(1, "PINHOLE", 960, 540, np.array([800.5, 801.25, 480.0, 270.0]))}
    imgs = {i + 1: Image(i + 1, rotmat2qvec(rand_rot(rng)), rng.normal(size=3), 1, f"{i+1:06d}.jpg") for i in range(n)}
    pts = Points3D(rng.normal(size=(50, 3)), rng.integers(0, 255, size=(50, 3)).astype(np.uint8))
    return SparseModel(cams, imgs, pts)


def test_colmap_text_and_binary_roundtrip(tmp_path):
    m = make_model()
    for kind, kw in (("bin", dict(text=False, binary=True)), ("txt", dict(text=True, binary=False))):
        d = tmp_path / kind
        write_model(m, d, **kw)
        r = read_model(d)
        assert r.summary() == m.summary()
        assert np.allclose(r.cameras[1].params, m.cameras[1].params)
        for i in m.images:
            assert r.images[i].name == m.images[i].name
            assert np.allclose(r.images[i].qvec, m.images[i].qvec, atol=1e-9)
            assert np.allclose(r.images[i].tvec, m.images[i].tvec, atol=1e-9)
        assert np.allclose(r.points.xyz, m.points.xyz, atol=1e-5)
        assert (r.points.rgb == m.points.rgb).all()

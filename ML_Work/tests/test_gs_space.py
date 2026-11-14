import numpy as np

from common import gs_space as G


def _pose(seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    V = np.eye(4); V[:3, :3] = Q; V[:3, 3] = rng.normal(size=3)
    return V


def test_world_transform_keeps_camera_relationship():
    # a point seen by a camera must project to the same camera-space point after the SAME similarity moves both
    V = _pose(0)
    T = np.eye(4); T[:3, :3] = 0.5 * _pose(1)[:3, :3]; T[:3, 3] = [1, 2, 3]       # scale 0.5 + rotation + shift
    p = np.array([0.3, -0.2, 2.0, 1.0])
    Vn = G.apply_world_transform(V, T)
    cam_before = (V @ p)[:3]
    cam_after = (Vn @ (T @ p))[:3]
    assert np.allclose(cam_before * 0.5, cam_after, atol=1e-9)   # only the scale changes camera-space distances


def test_c2w_w2c_roundtrip():
    c2w = np.stack([np.linalg.inv(_pose(i)) for i in range(3)])
    assert np.allclose(np.linalg.inv(G.c2w_to_w2c(c2w)), c2w)


def test_pick_space():
    assert G.pick_space(8.2, 26.0) == "normalized"
    assert G.pick_space(26.0, 8.0) == "raw"
    assert G.pick_space(25.0, 25.3) == "raw"        # inside the margin: leave the poses alone
    assert G.pick_space(25.0, None) == "raw"


def test_save_load_poses(tmp_path):
    P = {"000001.jpg": _pose(3), "000002.jpg": _pose(4)}
    G.save_poses(tmp_path / G.FILE, "normalized", P, {"raw": 8.0, "normalized": 26.0})
    back = G.load_poses(tmp_path / G.FILE)
    assert set(back) == set(P) and np.allclose(back["000001.jpg"], P["000001.jpg"])
    assert G.load_poses(tmp_path / "nope.json") is None

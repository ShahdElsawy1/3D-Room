"""Tests for the Week 1 update: relative paths, pycolmap helpers (numpy part), week2 inputs stage, pycolmap smoke."""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from common import pycolmap_sfm as psfm
from common.colmap_io import Camera, Image, Points3D, SparseModel, write_model
from common.config import Config, rel_to_project
from common.geometry import rotmat2qvec


# ----------------------------------------------------------------------------- relative paths
def test_rel_and_resolve_roundtrip(tmp_path):
    cfg = Config(root=str(tmp_path), project_id="demo")
    legacy = "/kaggle/working/workspace/demo/gs/gsplat_7000/ckpts/ckpt_6999.pt"
    assert cfg.rel(legacy) == "gs/gsplat_7000/ckpts/ckpt_6999.pt"
    assert cfg.resolve(legacy) == tmp_path / "demo" / "gs/gsplat_7000/ckpts/ckpt_6999.pt"
    assert cfg.resolve("gs/a.pt") == tmp_path / "demo" / "gs/a.pt"
    assert cfg.rel(None) is None and cfg.resolve("") is None
    assert cfg.rel(tmp_path / "demo" / "scene") == "scene"
    assert rel_to_project("C:\\data\\demo\\export\\scene.ply", "demo") == "export/scene.ply"


# ----------------------------------------------------------------------------- pure-numpy SfM helpers
def _model(n=20, jump_at=None, far_idx=None):
    cams = {1: Camera(1, "PINHOLE", 100, 80, np.array([100.0, 100.0, 50.0, 40.0]))}
    imgs = {}
    for i in range(n):
        ang = np.radians(2.0 * i + (60 if jump_at is not None and i > jump_at else 0))
        R = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
        c = np.array([0.05 * i, 0, 0.0])
        if far_idx == i:
            c = np.array([500.0, 0, 0])
        imgs[i + 1] = Image(i + 1, rotmat2qvec(R), -R @ c, 1, f"{i + 1:06d}.jpg")
    rng = np.random.default_rng(0)
    pts = Points3D(rng.normal(size=(300, 3)), np.zeros((300, 3), np.uint8), error=np.full(300, 0.5))
    return SparseModel(cams, imgs, pts)


def test_hybrid_pairs_window_plus_keyframe_cliques():
    p = psfm.hybrid_pairs(30, window=3, loop_stride=10)
    assert (0, 3) in p and (0, 4) not in p                   # window
    assert (0, 20) in p and (10, 20) in p                    # key frames 0,10,20 are all paired
    assert all(i < j for i, j in p) and len(p) == len(set(p))
    assert psfm.hybrid_pairs(30, 3, 0) == sorted({(i, j) for i in range(30) for j in range(i + 1, min(30, i + 4))})


def test_assess_ok_and_failures():
    assert psfm.assess(_model(), 20)["ok"]
    r = psfm.assess(_model(), 40)                            # only half registered
    assert not r["ok"] and "registered" in r["reasons"][0]
    r = psfm.assess(_model(jump_at=9), 20, max_jump_fraction=0.0)
    assert not r["ok"] and r["rotation_jumps"] == 1
    bad = _model(); bad.points.error[:] = 3.0
    assert not psfm.assess(bad, 20)["ok"]


def test_prune_candidates_jump_and_far():
    c = psfm.prune_candidates(_model(jump_at=9), jump_deg=45)
    assert c["jump_frames"] == ["000010.jpg", "000011.jpg"] and not c["far_frames"]
    c = psfm.prune_candidates(_model(far_idx=5), jump_deg=45)
    assert c["far_frames"] == ["000006.jpg"]
    assert psfm.prune_candidates(_model(), jump_deg=45)["names"] == []


def test_focal_factor_by_aspect():
    assert psfm.focal_factor(540, 960) == 0.73 and psfm.focal_factor(640, 480) == 0.89


def test_pycolmap_unavailable_is_reported_not_crashing(monkeypatch):
    monkeypatch.delenv("W1_PYCOLMAP_DIR", raising=False)
    monkeypatch.setitem(sys.modules, "pycolmap", None)       # import raises ImportError
    ok, why = psfm.available()
    assert not ok and "not importable" in why


# ----------------------------------------------------------------------------- week2 stage (CPU, no depth rendering)
def _project(tmp_path, undistorted=False):
    import s6_prepare_week2  # noqa: F401  (stages/ is on sys.path via conftest)
    cfg = Config(root=str(tmp_path), project_id="p")
    cfg.ensure()
    m = _model(16)
    scene_imgs = cfg.scene_dir / "images"
    scene_imgs.mkdir(parents=True)
    size = (98, 78) if undistorted else (100, 80)
    for im in m.images.values():
        cv2.imwrite(str(scene_imgs / im.name), np.zeros((size[1], size[0], 3), np.uint8))
        cv2.imwrite(str(cfg.images_dir / im.name), np.zeros((80, 100, 3), np.uint8))
    write_model(m, cfg.scene_dir / "sparse" / "0")
    (cfg.scene_dir / "chosen.json").write_text(json.dumps({"method": "glomap", "from": "poses/glomap"}))
    (cfg.poses_dir("glomap")).mkdir(parents=True)
    (cfg.poses_dir("glomap") / "glomap_meta.json").write_text(json.dumps({"undistorted": undistorted}))
    cfg.frames_index.write_text(json.dumps({"kept": 16, "image_size_wh": [100, 80], "fps": 2}))
    (cfg.export_dir / "cameras.json").write_text("[]")
    (cfg.export_dir / "scene_meta.json").write_text("{}")
    (cfg.export_dir / "scene.splat").write_bytes(b"0" * 32)
    return cfg


def test_week2_inputs_written_with_relative_paths(tmp_path):
    import s6_prepare_week2 as s6
    cfg = _project(tmp_path)
    s6.main(["--root", str(tmp_path), "--project", "p", "--strict"])
    d = json.loads((cfg.p("week2") / "inputs.json").read_text())
    assert d["images"]["same_pixels"] is True and d["images"]["scene_size_wh"] == [100, 80]
    assert d["pose_method"] == "glomap" and d["registered_frames"] == 16
    assert d["heldout"]["names"] == ["000001.jpg", "000009.jpg"]            # every 8th by default
    assert d["depth"]["kind"] == "sparse_only" and d["checkpoint"] is None
    assert len(list((cfg.p("week2") / "sparse_depth").glob("*.npy"))) == 16
    blob = json.dumps(d)
    assert str(tmp_path) not in blob and "/kaggle" not in blob               # nothing absolute
    for r in d["required_files"]:
        assert not Path(r).is_absolute() and cfg.p(r).exists(), r
    assert "week2/inputs.json" in d["required_files"] and "scene/images" in d["required_files"]


def test_week2_flags_undistorted_scene(tmp_path):
    import s6_prepare_week2 as s6
    cfg = _project(tmp_path, undistorted=True)
    s6.main(["--root", str(tmp_path), "--project", "p"])
    d = json.loads((cfg.p("week2") / "inputs.json").read_text())
    assert d["images"]["same_pixels"] is False and d["images"]["scene_size_wh"] == [98, 78]
    assert d["images"]["scene_is_undistorted_copy"] and d["images"]["recommended_edit_space"] == "scene/images"


def test_week2_missing_scene_does_not_fail_week1(tmp_path):
    import s6_prepare_week2 as s6
    cfg = Config(root=str(tmp_path), project_id="q"); cfg.ensure()
    with pytest.raises(SystemExit):                                          # explicit: "run stage 3 first"
        s6.main(["--root", str(tmp_path), "--project", "q"])


def test_week2_sparse_depth_values(tmp_path):
    import s6_prepare_week2 as s6
    cfg = _project(tmp_path)
    s6.main(["--root", str(tmp_path), "--project", "p"])
    a = np.load(cfg.p("week2") / "sparse_depth" / "000001.npy")
    assert a.ndim == 2 and a.shape[1] == 3 and (a[:, 2] > 0).all()
    assert (a[:, 0] >= 0).all() and (a[:, 0] < 100).all() and (a[:, 1] < 80).all()


# ----------------------------------------------------------------------------- real pycolmap smoke (skipped if absent)
@pytest.mark.skipif(not psfm.available()[0], reason="pycolmap>=4 not importable (set W1_PYCOLMAP_DIR)")
def test_pycolmap_backend_end_to_end(tmp_path):
    import subprocess
    sys.path.insert(0, str(Path(__file__).parent))
    from synth_scene import make_frames
    cfg = Config(root=str(tmp_path), project_id="syn"); cfg.ensure()
    make_frames(cfg.images_dir, n=16)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    r = subprocess.run([sys.executable, str(root / "stages" / "s2b_poses_glomap.py"), "--root", str(tmp_path),
                        "--project", "syn", "--backend", "pycolmap", "--window", "6", "--loop-stride", "5"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    meta = json.loads((cfg.poses_dir("glomap") / "glomap_meta.json").read_text())
    assert meta["backend_kind"] == "pycolmap" and meta["registered_frames"] >= 15
    assert meta["pycolmap"]["mappers_tried"][0]["mapper"] == "global"
    assert abs(meta["camera"]["params"][0] - 520) < 15                       # true focal 520 px

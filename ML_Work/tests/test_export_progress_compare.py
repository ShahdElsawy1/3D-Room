import json
import struct

import numpy as np
import pytest

from common.colmap_io import Camera, Image, Points3D, SparseModel, write_model
from common.config import Config, write_json
from common.geometry import rotmat2qvec, qvec2rotmat
from common.gs_io import SH_C0, cameras_json, ply_to_splat, read_gs_ply, write_gs_ply
from common.progress import Progress


def test_ply_to_splat_layout_and_decoding(tmp_path):
    rng = np.random.default_rng(0)
    n = 1000
    means = rng.normal(size=(n, 3)).astype(np.float32)
    sh0 = rng.normal(scale=0.5, size=(n, 1, 3)).astype(np.float32)
    shN = rng.normal(scale=0.1, size=(n, 15, 3)).astype(np.float32)
    op = rng.normal(size=n).astype(np.float32)
    sc = rng.normal(scale=0.3, size=(n, 3)).astype(np.float32) - 3
    q = rng.normal(size=(n, 4)).astype(np.float32)
    ply = tmp_path / "a.ply"
    write_gs_ply(ply, means, sh0, shN, op, sc, q)
    d = read_gs_ply(ply)
    assert len(d["x"]) == n and "f_rest_44" in d and "f_rest_45" not in d      # 3*15 SH rest coefficients
    info = ply_to_splat(ply, tmp_path / "a.splat", sort=False)
    raw = (tmp_path / "a.splat").read_bytes()
    assert len(raw) == 32 * n and info["num_gaussians"] == n
    rec = np.frombuffer(raw, dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    assert np.allclose(rec["p"], means) and np.allclose(rec["s"], np.exp(sc), rtol=1e-5)
    rgb = np.clip((0.5 + SH_C0 * sh0[:, 0]) * 255, 0, 255).astype(np.uint8)
    assert np.abs(rec["c"][:, :3].astype(int) - rgb.astype(int)).max() <= 1
    assert np.abs(rec["c"][:, 3].astype(int) - (255 / (1 + np.exp(-op))).astype(int)).max() <= 1
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    assert np.abs((rec["r"].astype(float) - 128) / 128 - qn).max() < 0.02


def test_ply_to_splat_sorting_puts_big_opaque_first(tmp_path):
    n = 3
    sc = np.log(np.array([[.01] * 3, [1.] * 3, [.1] * 3], np.float32))
    op = np.array([5, 5, 5], np.float32)
    write_gs_ply(tmp_path / "s.ply", np.arange(n * 3, dtype=np.float32).reshape(n, 3), np.zeros((n, 1, 3), np.float32),
                 np.zeros((n, 0, 3), np.float32), op, sc, np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32))
    ply_to_splat(tmp_path / "s.ply", tmp_path / "s.splat", sort=True)
    rec = np.frombuffer((tmp_path / "s.splat").read_bytes(), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    assert np.allclose(rec["s"][:, 0], [1.0, 0.1, 0.01], rtol=1e-4)


def test_progress_survives_dead_backend_and_maps_ranges(tmp_path):
    p = Progress(tmp_path, backend_url="http://127.0.0.1:9", task_id="7")     # nothing listens on port 9
    p.report("training", 0.5, force=True)
    p.finish("extracting frames")
    lines = [json.loads(l) for l in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert lines[0]["stage"] == "training" and lines[0]["progress"] == 67.5     # 45 + 0.5*45
    assert lines[1]["progress"] == 15.0


def fake_method(cfg, name, n, jitter, seed):
    rng = np.random.default_rng(seed)
    d = cfg.poses_dir(name)
    cams = {1: Camera(1, "PINHOLE", 960, 540, np.array([800., 800., 480., 270.]))}
    imgs = {}
    for i in range(n):
        ang = 2 * np.pi * i / n
        c = np.array([2 * np.cos(ang), 0, 2 * np.sin(ang)]) + rng.normal(scale=jitter, size=3)
        z = -c / np.linalg.norm(c)
        x = np.cross([0, -1, 0], z); x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z], 0)
        imgs[i + 1] = Image(i + 1, rotmat2qvec(R), -R @ c, 1, f"{i+1:06d}.jpg")
    pts = Points3D(rng.uniform(-3, 3, size=(2000, 3)), rng.integers(0, 255, size=(2000, 3)).astype(np.uint8))
    write_model(SparseModel(cams, imgs, pts), d / "sparse" / "0")
    (d / "images").mkdir(exist_ok=True)
    write_json(cfg.metrics_dir / f"s2{'a' if name == 'vggt' else 'b'}_poses_{name}.json",
               {"seconds": 40.0 if name == "vggt" else 400.0, "peak_vram_mb": 6000 if name == "vggt" else 900})


def test_compare_chooses_and_reports(tmp_path):
    import s3_compare_poses as s3
    import make_reports
    cfg = Config(root=str(tmp_path), project_id="p"); cfg.ensure()
    for i in range(60):
        (cfg.images_dir / f"{i+1:06d}.jpg").write_bytes(b"x")
    fake_method(cfg, "vggt", 60, 0.01, 1)
    fake_method(cfg, "glomap", 58, 0.0, 2)
    s3.main(["--root", str(tmp_path), "--project", "p", "--choose", "auto"])
    res = json.loads((cfg.metrics_dir / "pose_comparison.json").read_text())
    assert {m["method"] for m in res["methods"]} == {"vggt", "glomap"}
    assert res["trajectory_agreement"]["common_frames"] == 58
    assert res["trajectory_agreement"]["rmse_over_bbox_diag"] < 0.05
    assert res["suggested_choice"] == "vggt"                     # same coverage, 10x faster
    assert "VGGT" in res["draft_justification"]
    assert (cfg.scene_dir / "sparse" / "0" / "images.bin").exists()
    assert json.loads((cfg.scene_dir / "chosen.json").read_text())["method"] == "vggt"
    assert (cfg.metrics_dir / "pose_comparison.png").exists()
    # handoff + table generation from the measurements
    out = tmp_path / "gen"
    make_reports.main(["--root", str(tmp_path), "--project", "p", "--out", str(out)])
    assert "WEEK2_HANDOFF" in str(list(out.iterdir())) or (out / "WEEK2_HANDOFF.md").exists()
    man = json.loads(cfg.manifest_path.read_text())
    assert man["pose_method"] == "vggt" and man["registered_frames"] == 60
    assert man["intrinsics"]["1"]["model"] == "PINHOLE"
    cams = cameras_json(__import__("common.colmap_io", fromlist=["read_model"]).read_model(cfg.scene_dir / "sparse" / "0"))
    assert len(cams) == 60 and np.allclose(np.array(cams[0]["rotation"]) @ np.array(cams[0]["rotation"]).T, np.eye(3), atol=1e-8)

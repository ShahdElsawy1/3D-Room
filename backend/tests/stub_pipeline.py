#!/usr/bin/env python3
"""CPU stand-in for ML_Work/run_pipeline.py (same CLI), used to test the worker without a GPU.

REAL Person A code (run unmodified, from the ML_Work folder):
    s1_extract_frames.py (ffmpeg, blur filter)   s3_compare_poses.py --choose glomap (creates scene/, symlinks images)
    s5_export_splat.py (.ply -> .splat, cameras.json, scene_meta.json)   tools/make_reports.py (manifest.json)
    common.progress.Progress (posts real stage progress with W1_WORKER_TOKEN)
FAKED (GPU-only): poses (a PINHOLE camera circle written with A's colmap_io) and 3DGS training (random Gaussians
written with A's gs_io), with a configurable delay.

Env: STUB_ML_DIR (ML_Work), STUB_TRAIN_SECONDS (default 1), STUB_FAIL_AT ("poses" | "training"),
     STUB_CHILD_PIDFILE (training runs in a grandchild process whose pid is written here, to test tree kill).
Never writes into ML_Work (reports go to a temp dir; PYTHONDONTWRITEBYTECODE=1 avoids __pycache__).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ML = Path(os.environ["STUB_ML_DIR"]).resolve()
sys.path.insert(0, str(ML))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from common.colmap_io import Camera, Image, Points3D, SparseModel, write_model, write_points_ply  # noqa: E402
from common.config import Config, write_json  # noqa: E402
from common.gs_io import write_gs_ply  # noqa: E402
from common.progress import Progress  # noqa: E402


def run_a(script: str, args: list[str]) -> None:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, str(ML / script), *args], cwd=ML, env=env)
    if r.returncode:
        raise SystemExit(f"stage failed: {script}")


def fake_poses(cfg: Config, prog: Progress) -> None:
    prog.start("estimating poses")
    if os.environ.get("STUB_FAIL_AT") == "poses":
        prog.fail("estimating poses", "stub: simulated GLOMAP failure")
        raise SystemExit("stub: simulated failure in poses")
    names = sorted(p.name for p in cfg.images_dir.glob("*.jpg"))
    from PIL import Image as PILImage
    w, h = PILImage.open(cfg.images_dir / names[0]).size
    cams = {1: Camera(1, "PINHOLE", w, h, np.array([0.8 * w, 0.8 * w, w / 2, h / 2]))}
    imgs = {}
    for i, n in enumerate(names, 1):
        a = 2 * np.pi * i / len(names)
        imgs[i] = Image(i, np.array([np.cos(a / 2), 0, np.sin(a / 2), 0]), np.array([0, 0, 3.0]), 1, n)
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(500, 3))
    rgb = rng.integers(0, 255, (500, 3), dtype=np.uint8)
    out = cfg.poses_dir("glomap")
    write_model(SparseModel(cams, imgs, Points3D(xyz, rgb)), out / "sparse" / "0")
    write_points_ply(out / "points.ply", xyz, rgb)
    try:                                                  # same as A's stage: link, copy if links unsupported
        (out / "images").symlink_to(os.path.relpath(cfg.images_dir, out))
    except OSError:
        import shutil
        shutil.copytree(cfg.images_dir, out / "images")
    write_json(out / "glomap_meta.json", {"stub": True})
    write_json(cfg.metrics_dir / "s2b_poses_glomap.json", {"stage": "s2b_poses_glomap", "seconds": 0.5, "ok": True,
                                                           "peak_vram_mb": 0.0, "registered_frames": len(names)})
    prog.finish("estimating poses")


def fake_training(cfg: Config, prog: Progress) -> None:
    prog.start("training")
    secs = float(os.environ.get("STUB_TRAIN_SECONDS", "1"))
    child = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({secs})"])
    if os.environ.get("STUB_CHILD_PIDFILE"):
        Path(os.environ["STUB_CHILD_PIDFILE"]).write_text(str(child.pid))
    t0 = time.time()
    while child.poll() is None:
        prog.report("training", min(0.99, (time.time() - t0) / max(secs, 0.1)))
        time.sleep(0.2)
    if os.environ.get("STUB_FAIL_AT") == "training":
        prog.fail("training", "stub: CUDA out of memory")
        raise SystemExit("stub: simulated failure in training")
    n = 2000
    rng = np.random.default_rng(1)
    ply = cfg.gs_dir / "gsplat_7000" / "ply" / "point_cloud_6999.ply"
    write_gs_ply(ply, rng.normal(size=(n, 3)), rng.normal(size=(n, 1, 3)), None, rng.normal(size=n),
                 rng.normal(-3, 0.5, size=(n, 3)), rng.normal(size=(n, 4)))
    (cfg.gs_dir / "gsplat_7000" / "ckpts").mkdir(parents=True, exist_ok=True)
    (cfg.gs_dir / "gsplat_7000" / "ckpts" / "ckpt_6999.pt").write_bytes(b"\0" * 4096)
    write_json(cfg.gs_dir / "latest.json", {"trainer": "gsplat", "ply": str(ply), "ckpt": None, "iters": 7000,
                                            "num_gaussians": n, "result_dir": str(ply.parents[1])})
    write_json(cfg.metrics_dir / "s4_train_3dgs.json", {"stage": "s4_train_3dgs", "seconds": secs, "ok": True,
                                                        "peak_vram_mb": 0.0, "num_gaussians": n, "iters": 7000})
    prog.finish("training")


def main():
    ap = argparse.ArgumentParser()
    for f in ("--video", "--root", "--project", "--backend-url", "--task-id", "--method", "--train-mode", "--fps",
              "--long-side", "--trainer", "--max-frames"):
        ap.add_argument(f, default="")
    a = ap.parse_args()
    common = ["--root", a.root, "--project", a.project, "--backend-url", a.backend_url, "--task-id", a.task_id]
    cfg = Config(root=a.root, project_id=a.project)
    cfg.ensure()
    prog = Progress(cfg.metrics_dir, a.backend_url, a.task_id)

    run_a("stages/s1_extract_frames.py", ["--video", a.video, "--fps", a.fps or "2",
                                          "--long-side", a.long_side or "960", *common])
    fake_poses(cfg, prog)
    run_a("stages/s3_compare_poses.py", ["--choose", "glomap", *common])
    fake_training(cfg, prog)
    run_a("stages/s5_export_splat.py", common)
    run_a("tools/make_reports.py", ["--out", tempfile.mkdtemp(prefix="stub_reports_"), *common[:4]])
    prog.done()


if __name__ == "__main__":
    main()

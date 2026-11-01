#!/usr/bin/env python3
"""Stage 2b - camera poses with GLOMAP (global SfM on a COLMAP database).

    python stages/s2b_poses_glomap.py --project demo

Pipeline (exactly the PDF's): feature extraction -> SEQUENTIAL matching -> GLOMAP mapper
-> COLMAP-format sparse model -> convert to a 3DGS-ready COLMAP-style folder.

Backends, tried in this order (GLOMAP was merged into COLMAP 3.14 as `colmap global_mapper`):
    1. standalone `glomap mapper`
    2. `colmap global_mapper`           (COLMAP >= 3.14)
Feature extraction + matching always use the `colmap` CLI.

Outputs (poses/glomap/): images/ (symlink, or undistorted copy), sparse/0/*, points.ply, glomap_meta.json
Intrinsics: 3DGS needs undistorted PINHOLE images. We estimate SIMPLE_RADIAL, then either
  - drop k when |k| is tiny (phone video, default threshold 0.01), or
  - run `colmap image_undistorter` (images then differ slightly from frames/images; recorded in the meta).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.colmap_io import Camera, SparseModel, read_model, write_model, write_points_ply  # noqa: E402
from common.config import add_common_args, config_from_args, write_json  # noqa: E402
from common.metrics import StageMetrics  # noqa: E402
from common.progress import Progress  # noqa: E402

STAGE = "estimating poses"
ENV = dict(os.environ, QT_QPA_PLATFORM="offscreen")     # COLMAP's Qt needs this on headless boxes


def sh(cmd: list[str], log: Path, what: str) -> float:
    """Run a command, tee output to a log file, return seconds. Raises with the log tail on failure."""
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with log.open("w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=ENV)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        tail = "\n".join(log.read_text().splitlines()[-15:])
        raise RuntimeError(f"{what} failed (exit {r.returncode}). Log {log}:\n{tail}")
    print(f"[glomap] {what}: {dt:.1f}s", flush=True)
    return dt


def help_text(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd + ["-h"], capture_output=True, text=True, env=ENV, timeout=30)
        return r.stdout + r.stderr
    except Exception:
        return ""


def detect_backend() -> dict:
    colmap = shutil.which("colmap")
    glomap = shutil.which("glomap")
    has_global = bool(colmap and "global_mapper" in help_text([colmap, "help"]))
    if not colmap:
        raise RuntimeError("`colmap` CLI not found (needed for feature extraction/matching). "
                           "See scripts/setup_kaggle.sh for install options.")
    if glomap:
        mapper = [glomap, "mapper"]
    elif has_global:
        mapper = [colmap, "global_mapper"]
    else:
        raise RuntimeError("No global mapper found: need the `glomap` binary or COLMAP >= 3.14 "
                           "(`colmap global_mapper`). See scripts/setup_kaggle.sh.")
    return {"colmap": colmap, "glomap": glomap, "mapper": mapper,
            "mapper_name": "glomap mapper" if glomap else "colmap global_mapper"}


def gpu_flag(cmd_help: str, base: str, use_gpu: bool) -> list[str]:
    """COLMAP renamed SiftExtraction.use_gpu -> FeatureExtraction.use_gpu (3.12) etc."""
    for prefix in ("FeatureExtraction", "FeatureMatching", "SiftExtraction", "SiftMatching"):
        if f"{prefix}.use_gpu" in cmd_help and prefix.startswith(base):
            return [f"--{prefix}.use_gpu", "1" if use_gpu else "0"]
    return []


def pick_best_model(sparse_root: Path) -> Path:
    cands = [d for d in sorted(sparse_root.iterdir()) if d.is_dir() and
             ((d / "images.bin").exists() or (d / "images.txt").exists())]
    if not cands:
        raise RuntimeError(f"GLOMAP produced no model in {sparse_root}")
    return max(cands, key=lambda d: len(read_model(d).images))


def to_pinhole(model: SparseModel) -> tuple[SparseModel, float]:
    """SIMPLE_RADIAL/RADIAL/OPENCV -> PINHOLE (drop distortion). Returns (model, max |k1|)."""
    max_k = 0.0
    cams = {}
    for cid, c in model.cameras.items():
        p = c.params
        if c.model == "PINHOLE":
            cams[cid] = c
        elif c.model == "SIMPLE_PINHOLE":
            cams[cid] = Camera(cid, "PINHOLE", c.width, c.height, np.array([p[0], p[0], p[1], p[2]]))
        elif c.model in ("SIMPLE_RADIAL", "RADIAL"):
            max_k = max(max_k, abs(p[3]))
            cams[cid] = Camera(cid, "PINHOLE", c.width, c.height, np.array([p[0], p[0], p[1], p[2]]))
        elif c.model == "OPENCV":
            max_k = max(max_k, abs(p[4]))
            cams[cid] = Camera(cid, "PINHOLE", c.width, c.height, np.array(p[:4]))
        else:
            raise ValueError(c.model)
    return SparseModel(cams, model.images, model.points), max_k


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--overlap", type=int, default=12, help="sequential matching neighbours")
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument("--gpu", action="store_true",
                    help="GPU SIFT/matching. OFF by default: COLMAP's GPU SIFT needs an OpenGL display and crashes on "
                         "headless Kaggle (OpenGLContextManager). CPU is fast enough for a few hundred frames.")
    ap.add_argument("--no-gpu", action="store_true", help="(kept for compatibility; CPU is already the default)")
    ap.add_argument("--camera-model", default="SIMPLE_RADIAL")
    ap.add_argument("--undistort", default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--k-threshold", type=float, default=0.01)
    args = ap.parse_args(argv)

    cfg = config_from_args(args)
    prog = Progress(cfg.metrics_dir, args.backend_url, args.task_id)
    prog.start(STAGE)
    out = cfg.poses_dir("glomap")
    work = out / "_work"
    logs = cfg.metrics_dir / "logs"
    imgs = cfg.images_dir
    if not list(imgs.glob("*.jpg")):
        prog.fail(STAGE, "no frames - run stage 1 first")
        raise SystemExit("no frames found")

    with StageMetrics(cfg.metrics_dir, "s2b_poses_glomap") as m:
        try:
            be = detect_backend()
            colmap = be["colmap"]
            shutil.rmtree(work, ignore_errors=True)
            (work / "sparse").mkdir(parents=True)
            db = work / "database.db"
            use_gpu = args.gpu and not args.no_gpu

            fe_help = help_text([colmap, "feature_extractor"])
            sm_help = help_text([colmap, "sequential_matcher"])
            cmd = [colmap, "feature_extractor", "--database_path", str(db), "--image_path", str(imgs),
                   "--ImageReader.camera_model", args.camera_model, "--ImageReader.single_camera", "1",
                   "--SiftExtraction.max_num_features", str(args.max_features)]
            cmd += gpu_flag(fe_help, "FeatureExtraction", use_gpu) or gpu_flag(fe_help, "SiftExtraction", use_gpu)
            t_feat = sh(cmd, logs / "glomap_features.log", "feature extraction")
            prog.report(STAGE, 0.25)

            cmd = [colmap, "sequential_matcher", "--database_path", str(db),
                   "--SequentialMatching.overlap", str(args.overlap), "--SequentialMatching.loop_detection", "0"]
            cmd += gpu_flag(sm_help, "FeatureMatching", use_gpu) or gpu_flag(sm_help, "SiftMatching", use_gpu)
            t_match = sh(cmd, logs / "glomap_matching.log", "sequential matching")
            prog.report(STAGE, 0.5)

            t_map = sh(be["mapper"] + ["--database_path", str(db), "--image_path", str(imgs),
                                       "--output_path", str(work / "sparse")],
                       logs / "glomap_mapper.log", be["mapper_name"])
            prog.report(STAGE, 0.8)

            best = pick_best_model(work / "sparse")
            raw = read_model(best)
            n_frames = len(list(imgs.glob("*.jpg")))
            model, max_k = to_pinhole(raw)
            need_undistort = args.undistort == "always" or (args.undistort == "auto" and max_k > args.k_threshold)
            undistorted = False
            if need_undistort:
                und = work / "undist"
                sh([colmap, "image_undistorter", "--image_path", str(imgs), "--input_path", str(best),
                    "--output_path", str(und), "--output_type", "COLMAP"],
                   logs / "glomap_undistort.log", "image undistortion")
                model = read_model(und / "sparse" if (und / "sparse" / "cameras.bin").exists() or
                                   (und / "sparse" / "cameras.txt").exists() else und / "sparse" / "0")
                if out.joinpath("images").is_symlink():
                    out.joinpath("images").unlink()
                shutil.rmtree(out / "images", ignore_errors=True)
                shutil.copytree(und / "images", out / "images")
                undistorted = True
            else:
                dst = out / "images"
                if dst.is_symlink() or dst.exists():
                    dst.unlink() if dst.is_symlink() else shutil.rmtree(dst)
                try:
                    dst.symlink_to(os.path.relpath(imgs, out))
                except OSError:
                    shutil.copytree(imgs, dst)

            shutil.rmtree(out / "sparse", ignore_errors=True)
            write_model(model, out / "sparse" / "0")
            if len(model.points.xyz):
                write_points_ply(out / "points.ply", model.points.xyz, model.points.rgb)

            cam = next(iter(model.cameras.values()))
            meta = {"backend": be["mapper_name"], "camera_model_estimated": args.camera_model,
                    "max_abs_k1": max_k, "undistorted": undistorted,
                    "k_threshold": args.k_threshold,
                    "registered_frames": len(model.images), "total_frames": n_frames,
                    "points3D": int(len(model.points.xyz)),
                    "seconds": {"features": round(t_feat, 1), "matching": round(t_match, 1), "mapper": round(t_map, 1)},
                    "camera": {"model": cam.model, "width": cam.width, "height": cam.height, "params": cam.params.tolist()},
                    "note": ("images/ is an UNDISTORTED copy (pixels differ slightly from frames/images)"
                             if undistorted else "images/ -> frames/images (distortion k dropped; it was tiny)")}
            write_json(out / "glomap_meta.json", meta)
            shutil.rmtree(work, ignore_errors=True)
            extra = dict(meta)
            extra["seconds_breakdown"] = extra.pop("seconds")   # keep top-level "seconds" = stage wall time
            m.extra.update(extra)
            print(f"[glomap] registered {meta['registered_frames']}/{n_frames} frames, "
                  f"{meta['points3D']} points ({be['mapper_name']})")
            prog.report(STAGE, 1.0)
        except Exception as e:
            prog.fail(STAGE, e)
            raise


if __name__ == "__main__":
    main()

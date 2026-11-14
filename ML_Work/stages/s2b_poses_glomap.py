#!/usr/bin/env python3
"""Stage 2b - camera poses with GLOMAP (global SfM on a COLMAP database).

    python stages/s2b_poses_glomap.py --project demo

Pipeline (exactly the PDF's): feature extraction -> SEQUENTIAL matching -> GLOMAP mapper
-> COLMAP-format sparse model -> convert to a 3DGS-ready COLMAP-style folder.

--backend cli|pycolmap|auto  (auto = pycolmap when pycolmap >= 4 with global_mapping is importable, else cli)

  cli      (the original route; unchanged) tried in this order (GLOMAP was merged into COLMAP 3.14 as `colmap global_mapper`):
              1. standalone `glomap mapper`
              2. `colmap global_mapper`           (COLMAP >= 3.14)
           Feature extraction + matching use the `colmap` CLI. Of several `colmap` binaries, the first one that starts is used.
  pycolmap everything in-process via `pip install pycolmap-cuda12==4.2.1` (GPU SIFT through CUDA, built-in GLOMAP, hybrid
           matching, focal prior, GLOMAP -> incremental fallback when the result looks unreliable, pose pruning).
           See common/pycolmap_sfm.py. Needs no conda / faiss / OpenGL.

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
from common import pycolmap_sfm as psfm  # noqa: E402
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


def starts(cmd: list[str]) -> bool:
    """Does this command actually run? (a conda colmap can exist on PATH and still die on a missing/ABI-broken library)"""
    try:
        r = subprocess.run(cmd + ["-h"], capture_output=True, text=True, env=ENV, timeout=60)
    except Exception:
        return False
    txt = (r.stdout + r.stderr).lower()
    return not any(bad in txt for bad in ("undefined symbol", "error while loading shared libraries",
                                          "cannot open shared object"))


def find_working(name: str, extra: tuple[str, ...] = (), sub: str | None = None) -> str | None:
    """First executable called `name` that starts: PATH entry first (keeps the old behaviour), then `extra` paths."""
    seen = []
    for c in [shutil.which(name), *extra]:
        if c and c not in seen and Path(c).exists():
            seen.append(c)
    for c in seen:
        if starts([c, sub] if sub else [c]):
            return c
    if seen:
        print(f"[glomap] found {name} at {seen} but none of them starts (library problem?)", flush=True)
    return None


def detect_backend() -> dict:
    colmap = find_working("colmap", ("/usr/bin/colmap", "/usr/local/bin/colmap"), "feature_extractor")
    glomap = find_working("glomap", ("/usr/bin/glomap", "/usr/local/bin/glomap"), "mapper")
    has_global = bool(colmap and "global_mapper" in help_text([colmap, "help"]))
    if not colmap:
        raise RuntimeError("no working `colmap` CLI (needed for feature extraction/matching). Either install a working one "
                           "(apt-get install colmap) or use --backend pycolmap (pip install pycolmap-cuda12==4.2.1). "
                           "See scripts/setup_kaggle.sh.")
    if glomap:
        mapper = [glomap, "mapper"]
    elif has_global:
        mapper = [colmap, "global_mapper"]
    else:
        raise RuntimeError("No global mapper found: need the `glomap` binary or COLMAP >= 3.14 "
                           "(`colmap global_mapper`), or use --backend pycolmap. See scripts/setup_kaggle.sh.")
    return {"colmap": colmap, "glomap": glomap, "mapper": mapper,
            "mapper_name": "glomap mapper" if glomap else "colmap global_mapper"}


def gpu_flag(cmd_help: str, base: str, use_gpu: bool) -> list[str]:
    """COLMAP renamed SiftExtraction.use_gpu -> FeatureExtraction.use_gpu (3.12) etc."""
    for prefix in ("FeatureExtraction", "FeatureMatching", "SiftExtraction", "SiftMatching"):
        if f"{prefix}.use_gpu" in cmd_help and prefix.startswith(base):
            return [f"--{prefix}.use_gpu", "1" if use_gpu else "0"]
    return []


def run_mapper(be: dict, db: Path, imgs: Path, out: Path, logs: Path, mode: str) -> tuple[float, str]:
    """Run the global mapper. Known issue: a new `glomap` next to an old `colmap` CLI (e.g. 3.8 from conda) aborts in
    its final retriangulation step ("existing_frame.DataIds() == frame.DataIds()"). Poses are already final by then, so
    `auto` retries with --skip_retriangulation 1 (only the 3D points are not re-polished).
    Returns (seconds, label of the attempt that worked)."""
    base = be["mapper"] + ["--database_path", str(db), "--image_path", str(imgs), "--output_path", str(out)]
    skip_ok = "skip_retriangulation" in help_text(be["mapper"])
    attempts = []
    if mode in ("auto", "on"):
        attempts.append(("with retriangulation", [], "glomap_mapper.log"))
    if mode == "off" or (mode == "auto" and skip_ok):
        if not skip_ok:
            raise RuntimeError("this mapper has no --skip_retriangulation option")
        attempts.append(("retriangulation skipped", ["--skip_retriangulation", "1"], "glomap_mapper_skip_retri.log"))
    last = None
    for label, extra, logname in attempts:
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        try:
            return sh(base + extra, logs / logname, f"{be['mapper_name']} ({label})"), label
        except RuntimeError as e:
            last = e
            print(f"[glomap] attempt '{label}' failed; see {logs / logname}", flush=True)
    raise last


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


def sfm_cli(args, prog, imgs: Path, work: Path, logs: Path) -> dict:
    """The original route: colmap CLI features + sequential matching + glomap / colmap global_mapper. Unchanged logic."""
    be = detect_backend()
    colmap = be["colmap"]
    (work / "sparse").mkdir(parents=True, exist_ok=True)
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

    t_map, retri = run_mapper(be, db, imgs, work / "sparse", logs, args.retriangulation)
    prog.report(STAGE, 0.8)

    def undistort_fn(model_dir: Path, und: Path):
        sh([colmap, "image_undistorter", "--image_path", str(imgs), "--input_path", str(model_dir),
            "--output_path", str(und), "--output_type", "COLMAP"], logs / "glomap_undistort.log", "image undistortion")

    return {"best": pick_best_model(work / "sparse"), "backend": be["mapper_name"], "mapper_mode": retri,
            "seconds": {"features": t_feat, "matching": t_match, "mapper": t_map}, "undistort": undistort_fn,
            "extra": {}}


def sfm_pycolmap(args, prog, imgs: Path, work: Path, logs: Path) -> dict:
    """pycolmap >= 4 route: GPU SIFT, hybrid matching, GLOMAP (-> incremental when unreliable), pose pruning."""
    ok, why = psfm.available()
    if not ok:
        raise RuntimeError(f"--backend pycolmap not usable: {why}. Install it with: pip install --target "
                           "/kaggle/working/tools/pycolmap4 pycolmap-cuda12==4.2.1 and export "
                           "W1_PYCOLMAP_DIR=/kaggle/working/tools/pycolmap4 (see scripts/setup_kaggle.sh pycolmap).")
    pc = psfm.import_pycolmap()
    print(f"[pycolmap] {why}", flush=True)
    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    n_total = len(list(imgs.glob("*.jpg")))
    use_gpu = bool(getattr(pc, "has_cuda", False)) and not args.no_gpu

    ff = None if args.focal_factor == "auto" else float(args.focal_factor)
    t0 = time.perf_counter()
    feat = psfm.extract_features(pc, db, imgs, ff=ff, max_image=args.sift_max_image, max_features=args.max_features,
                                 camera_model=args.camera_model, use_gpu=use_gpu)
    t_feat = time.perf_counter() - t0
    print(f"[pycolmap] features {t_feat:.0f}s {feat}", flush=True)
    prog.report(STAGE, 0.2)

    mt = psfm.match(pc, db, imgs, mode=args.matcher, window=args.window, loop_stride=args.loop_stride,
                    work=work, use_gpu=use_gpu)
    t_match = mt["seconds"]
    print(f"[pycolmap] matching ({args.matcher}) {t_match:.0f}s: {mt['verified_pairs']} verified pairs, "
          f"{mt['n_weak_frames']} frames without a verified neighbour", flush=True)
    prog.report(STAGE, 0.45)

    tried = []
    order = ["global", "incremental"] if args.mapper == "auto" else [args.mapper]
    chosen, t_map = None, 0.0
    for mp in order:
        sec, mdir = psfm.map_model(pc, db, imgs, work / f"sparse_{mp}", mp)
        t_map += sec
        verdict = psfm.assess(read_model(mdir), n_total)
        verdict.update(mapper=mp, seconds=round(sec, 1), dir=mdir)
        tried.append(verdict)
        print(f"[pycolmap] {mp} mapping {sec:.0f}s: {verdict['registered']}/{n_total} frames, "
              f"{verdict['mean_reproj_error_px']} px, ok={verdict['ok']} {verdict['reasons']}", flush=True)
        chosen = verdict
        prog.report(STAGE, 0.7)
        if verdict["ok"]:
            break
    if not chosen["ok"] and len(tried) > 1:         # both tried and neither is clean: keep the one with more frames
        chosen = max(tried, key=lambda v: (v["registered"], -(v["mean_reproj_error_px"] or 1e9)))
        print(f"[pycolmap] neither mapper passed the checks; keeping '{chosen['mapper']}'", flush=True)
    best = Path(chosen["dir"])

    pruned = {"names": [], "enabled": args.prune}
    if args.prune:
        cand = psfm.prune_candidates(read_model(best), args.prune_jump_deg)
        limit = int(args.max_prune_fraction * chosen["registered"])
        if cand["names"] and len(cand["names"]) <= limit:
            clean = work / "sparse_clean" / "0"
            left = psfm.prune_with_pycolmap(pc, best, clean, cand["names"])
            print(f"[pycolmap] pruned {len(cand['names'])} frames -> {left} remain", flush=True)
            best, pruned = clean, {**cand, "enabled": True, "applied": True}
        elif cand["names"]:
            print(f"[pycolmap] {len(cand['names'])} frames flagged (> {limit} allowed); NOT pruning - the model is "
                  "suspect, check metrics/pose_comparison and consider --mapper incremental", flush=True)
            pruned = {**cand, "enabled": True, "applied": False}

    def undistort_fn(model_dir: Path, und: Path):
        psfm.undistort(pc, model_dir, imgs, und)
        print("[pycolmap] undistorted images written", flush=True)

    return {"best": best, "backend": f"pycolmap {getattr(pc, '__version__', '?')} " +
            ("global_mapping" if chosen["mapper"] == "global" else "incremental_mapping"),
            "mapper_mode": chosen["mapper"], "undistort": undistort_fn,
            "seconds": {"features": t_feat, "matching": t_match, "mapper": t_map},
            "extra": {"pycolmap": {"features": feat, "matching": {k: v for k, v in mt.items() if k != "seconds"},
                                   "mappers_tried": [{k: v for k, v in t.items() if k != "dir"} for t in tried],
                                   "pruned": {k: v for k, v in pruned.items() if k != "names"} |
                                             {"n_pruned": len(pruned["names"]) if pruned.get("applied") else 0},
                                   "gpu": use_gpu}}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--backend", default="auto", choices=["auto", "cli", "pycolmap"],
                    help="cli: colmap/glomap binaries (original). pycolmap: pip pycolmap>=4. auto: pycolmap if usable")
    ap.add_argument("--overlap", type=int, default=12, help="[cli] sequential matching neighbours")
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument("--gpu", action="store_true",
                    help="[cli] GPU SIFT/matching. OFF by default: COLMAP's GPU SIFT needs an OpenGL display and crashes on "
                         "headless Kaggle (OpenGLContextManager). CPU is fast enough for a few hundred frames. "
                         "(pycolmap uses CUDA, not OpenGL, so it uses the GPU automatically; --no-gpu turns that off)")
    ap.add_argument("--no-gpu", action="store_true", help="(cli: kept for compatibility; pycolmap: force CPU)")
    ap.add_argument("--retriangulation", default="auto", choices=["auto", "on", "off"],
                    help="[cli] auto: try the full mapper, retry without retriangulation if it crashes (old-colmap/new-glomap "
                         "mismatch); off: always skip it")
    ap.add_argument("--camera-model", default="SIMPLE_RADIAL")
    ap.add_argument("--undistort", default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--k-threshold", type=float, default=0.01)
    g = ap.add_argument_group("pycolmap backend")
    g.add_argument("--matcher", default="hybrid", choices=["hybrid", "sequential", "exhaustive"],
                   help="hybrid = sequential window + all pairs among every --loop-stride-th frame (loop-closure stand-in)")
    g.add_argument("--window", type=int, default=20, help="neighbours matched per frame (hybrid/sequential)")
    g.add_argument("--loop-stride", type=int, default=10, help="hybrid: every Nth frame is a loop-closure key frame")
    g.add_argument("--mapper", default="auto", choices=["auto", "global", "incremental"],
                   help="auto: GLOMAP first, incremental only if the global result fails the quality checks")
    g.add_argument("--focal-factor", default="auto", help="prior for f/longer side: auto (by aspect), a number, or 0 = none")
    g.add_argument("--sift-max-image", type=int, default=1600)
    g.add_argument("--prune", action=argparse.BooleanOptionalAction, default=True,
                   help="drop frames at rotation jumps / far outside the point cloud")
    g.add_argument("--prune-jump-deg", type=float, default=45.0)
    g.add_argument("--max-prune-fraction", type=float, default=0.10, help="never prune more than this share of frames")
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
            backend = args.backend
            if backend == "auto":
                backend = "pycolmap" if psfm.available()[0] else "cli"
            print(f"[glomap] backend: {backend}", flush=True)
            shutil.rmtree(work, ignore_errors=True)
            run = sfm_pycolmap if backend == "pycolmap" else sfm_cli
            res = run(args, prog, imgs, work, logs)
            best, retri = res["best"], res["mapper_mode"]
            t_feat, t_match, t_map = (res["seconds"][k] for k in ("features", "matching", "mapper"))

            raw = read_model(best)
            n_frames = len(list(imgs.glob("*.jpg")))
            model, max_k = to_pinhole(raw)
            need_undistort = args.undistort == "always" or (args.undistort == "auto" and max_k > args.k_threshold)
            undistorted = False
            if need_undistort:
                und = work / "undist"
                res["undistort"](best, und)
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
            meta = {"backend": res["backend"], "backend_kind": backend, "camera_model_estimated": args.camera_model,
                    "max_abs_k1": max_k, "undistorted": undistorted,
                    "k_threshold": args.k_threshold,
                    "registered_frames": len(model.images), "total_frames": n_frames,
                    "points3D": int(len(model.points.xyz)), "mapper_mode": retri,
                    "seconds": {"features": round(t_feat, 1), "matching": round(t_match, 1), "mapper": round(t_map, 1)},
                    "camera": {"model": cam.model, "width": cam.width, "height": cam.height, "params": cam.params.tolist()},
                    "raw_camera": (lambda rc: {"model": rc.model, "width": rc.width, "height": rc.height,
                                               "params": rc.params.tolist()})(next(iter(raw.cameras.values()))),
                    "frames_image_size_wh": [raw.cameras[next(iter(raw.cameras))].width,
                                             raw.cameras[next(iter(raw.cameras))].height],
                    "note": ("images/ is an UNDISTORTED copy (pixels differ slightly from frames/images)"
                             if undistorted else "images/ -> frames/images (distortion k dropped; it was tiny)"),
                    **res["extra"]}
            write_json(out / "glomap_meta.json", meta)
            shutil.rmtree(work, ignore_errors=True)
            extra = dict(meta)
            extra["seconds_breakdown"] = extra.pop("seconds")   # keep top-level "seconds" = stage wall time
            m.extra.update(extra)
            print(f"[glomap] registered {meta['registered_frames']}/{n_frames} frames, "
                  f"{meta['points3D']} points ({res['backend']})")
            prog.report(STAGE, 1.0)
        except Exception as e:
            prog.fail(STAGE, e)
            raise


if __name__ == "__main__":
    main()

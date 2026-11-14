#!/usr/bin/env python3
"""Stage 4 - 3D Gaussian Splatting training on the chosen COLMAP-style scene (T4-friendly).

    python stages/s4_train_3dgs.py --project demo --mode quick      # ~7k iters, sanity check
    python stages/s4_train_3dgs.py --project demo --mode full       # ~30k iters, quality
    python stages/s4_train_3dgs.py --project demo --trainer inria   # original reference code

Trainers (PDF 4.2: "original reference implementation or gsplat"):
  gsplat : examples/simple_trainer.py from a gsplat checkout (GSPLAT_DIR or --gsplat-dir)
  inria  : graphdeco-inria/gaussian-splatting train.py          (INRIA_GS_DIR or --inria-dir)

Starts from the sparse point cloud in scene/sparse/0/points3D (both trainers do this).
Logs: wall time, final #Gaussians, peak VRAM (nvidia-smi sampler + trainer's own), held-out PSNR/SSIM.
Writes gs/latest.json (consumed by stage 5) and metrics/s4_train_3dgs.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import add_common_args, config_from_args, write_json  # noqa: E402
from common.metrics import StageMetrics  # noqa: E402
from common.progress import Progress  # noqa: E402

STAGE = "training"
PROG_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")          # tqdm: " 1234/7000 [..."


def find_dir(explicit: str, env: str, candidates: list[str], marker: str) -> Path:
    for c in [explicit, os.environ.get(env, "")] + candidates:
        if c and (Path(c) / marker).exists():
            return Path(c).resolve()
    raise FileNotFoundError(f"could not find {marker}. Pass --{env.lower().split('_')[0]}-dir or set {env} "
                            f"(scripts/setup_kaggle.sh clones it for you).")


def flag(help_txt: str, name: str):
    """tyro accepts --max-steps / --max_steps depending on version; use whichever the CLI prints."""
    for style in (name.replace("_", "-"), name):
        if f"--{style}" in help_txt:
            return f"--{style}"
    return None


def run_with_progress(cmd, log_path: Path, env, prog: Progress, total: int):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        log.write(("$ " + " ".join(cmd) + "\n").encode())
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        buf = b""
        last = 0
        while True:
            chunk = p.stdout.read1(4096) if hasattr(p.stdout, "read1") else p.stdout.read(4096)
            if not chunk:
                break
            log.write(chunk)
            buf = (buf + chunk)[-2000:]
            m = None
            for m in PROG_RE.finditer(buf.decode("utf-8", "ignore")):
                pass
            if m:
                step, tot = int(m.group(1)), int(m.group(2))
                if step != last and tot > 0:
                    last = step
                    prog.report(STAGE, min(step / tot, 1.0))
        rc = p.wait()
    if rc != 0:
        tail = "\n".join(log_path.read_text(errors="ignore").splitlines()[-25:])
        raise RuntimeError(f"trainer exited with code {rc}. Log tail ({log_path}):\n{tail}")


def patch_pycolmap_numpy2() -> None:
    """The pycolmap fork used by gsplat's examples does `np.uint64(-1)`, an OverflowError on numpy>=2.
    Replace it with the identical value spelled correctly (all bits set). Idempotent."""
    import importlib.util
    spec = importlib.util.find_spec("pycolmap")          # find_spec does not import the package
    if not spec or not spec.origin:
        return
    f = Path(spec.origin).parent / "scene_manager.py"
    if f.exists():
        t = f.read_text()
        if "np.uint64(-1)" in t:
            f.write_text(t.replace("np.uint64(-1)", "np.uint64(np.iinfo(np.uint64).max)"))
            print(f"[train] patched {f} for numpy>=2 (np.uint64(-1))", flush=True)


def train_gsplat(args, cfg, iters, prog, env):
    root = find_dir(args.gsplat_dir, "GSPLAT_DIR", ["/kaggle/working/gsplat", "gsplat", "../gsplat"],
                    "examples/simple_trainer.py")
    script = root / "examples" / "simple_trainer.py"
    # gsplat's examples/datasets/ has no __init__.py, so it is a namespace package and loses against the installed
    # Hugging Face `datasets` package ("No module named 'datasets.colmap'"). A regular package in the script's own
    # directory wins, so make it one.
    for d in (root / "examples" / "datasets",):
        if d.is_dir() and not (d / "__init__.py").exists():
            (d / "__init__.py").write_text("")
            print(f"[train] created {d / '__init__.py'} (avoids clash with Hugging Face 'datasets')", flush=True)
    patch_pycolmap_numpy2()
    probe = subprocess.run([sys.executable, str(script), "default", "--help"], capture_output=True,
                           text=True, env=env, cwd=root / "examples")
    help_txt = probe.stdout
    if "--data" not in help_txt:        # probing flags failed: do not guess flag spellings
        raise RuntimeError("could not read the gsplat trainer's --help (flag names unknown). stderr tail:\n" +
                           "\n".join(probe.stderr.splitlines()[-15:]))
    res_dir = cfg.gs_dir / f"gsplat_{iters}"
    cmd = [sys.executable, str(script), "default",
           "--data_dir" if "--data_dir" in help_txt else "--data-dir", str(cfg.scene_dir.resolve()),
           flag(help_txt, "data_factor") or "--data_factor", str(cfg.train.data_factor),
           flag(help_txt, "result_dir") or "--result_dir", str(res_dir.resolve())]
    # steps_scaler rescales max_steps AND the densification / eval / save schedules together
    scaler = flag(help_txt, "steps_scaler")
    if scaler:
        cmd += [scaler, f"{iters / 30000:.6f}"]
    else:
        cmd += [flag(help_txt, "max_steps") or "--max_steps", str(iters)]
    for opt, val in (("test_every", str(cfg.train.holdout_every)),):
        f = flag(help_txt, opt)
        if f:
            cmd += [f, val]
    if flag(help_txt, "save_ply"):
        cmd += [flag(help_txt, "save_ply")]
    if flag(help_txt, "disable_viewer"):
        cmd += [flag(help_txt, "disable_viewer")]
    cmd += list(cfg.train.extra_args)
    run_with_progress(cmd, cfg.metrics_dir / "logs" / "train_gsplat.log", env, prog, iters)
    return res_dir


def train_inria(args, cfg, iters, prog, env):
    root = find_dir(args.inria_dir, "INRIA_GS_DIR", ["/kaggle/working/gaussian-splatting", "gaussian-splatting"],
                    "train.py")
    res_dir = cfg.gs_dir / f"inria_{iters}"
    cmd = [sys.executable, str(root / "train.py"), "-s", str(cfg.scene_dir.resolve()),
           "-m", str(res_dir.resolve()), "--iterations", str(iters),
           "--save_iterations", str(iters), "--test_iterations", str(iters), "-r", "1", "--eval"]
    run_with_progress(cmd + list(cfg.train.extra_args), cfg.metrics_dir / "logs" / "train_inria.log", env, prog, iters)
    return res_dir


def collect_outputs(trainer: str, res_dir: Path) -> dict:
    info = {"trainer": trainer, "result_dir": str(res_dir)}
    if trainer == "gsplat":
        ckpts = sorted((res_dir / "ckpts").glob("ckpt_*.pt"),
                       key=lambda p: int(re.search(r"ckpt_(\d+)", p.name).group(1)))
        plys = sorted((res_dir / "ply").glob("*.ply")) if (res_dir / "ply").exists() else []
        info["ckpt"] = str(ckpts[-1]) if ckpts else None
        info["ply"] = str(plys[-1]) if plys else None
        stats = {}
        for kind in ("train", "val"):
            files = sorted((res_dir / "stats").glob(f"{kind}_step*.json"),
                           key=lambda p: int(re.search(r"step(\d+)", p.name).group(1))) \
                if (res_dir / "stats").exists() else []
            if files:
                stats[kind] = json.loads(files[-1].read_text())
        info["stats"] = stats
        if "train" in stats:
            info["num_gaussians"] = stats["train"].get("num_GS")
            info["trainer_peak_mem_gb"] = stats["train"].get("mem")
            info["trainer_train_seconds"] = stats["train"].get("ellipse_time")
        if "val" in stats:
            info["heldout"] = {k: stats["val"].get(k) for k in ("psnr", "ssim", "lpips")}
    else:
        plys = sorted((res_dir / "point_cloud").glob("iteration_*/point_cloud.ply"),
                      key=lambda p: int(re.search(r"iteration_(\d+)", str(p)).group(1)))
        info["ply"] = str(plys[-1]) if plys else None
        info["ckpt"] = None
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--trainer", default="gsplat", choices=["gsplat", "inria"])
    ap.add_argument("--mode", default="quick", choices=["quick", "full", "custom"])
    ap.add_argument("--iters", type=int, default=0, help="with --mode custom (spec: 7k quick, 15-30k full)")
    ap.add_argument("--gsplat-dir", default="")
    ap.add_argument("--inria-dir", default="")
    ap.add_argument("--holdout-every", type=int, default=8, help="every Nth frame held out for PSNR/SSIM")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES for the trainer")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra args passed to the trainer")
    args = ap.parse_args(argv)

    cfg = config_from_args(args)
    cfg.train.holdout_every = args.holdout_every
    cfg.train.extra_args = args.extra
    iters = {"quick": cfg.train.quick_iters, "full": cfg.train.full_iters}.get(args.mode, args.iters)
    if iters <= 0:
        raise SystemExit("--mode custom needs --iters N")
    if not (cfg.scene_dir / "sparse" / "0").exists():
        raise SystemExit("scene/ missing - run stage 3 (compare/choose) first")

    prog = Progress(cfg.metrics_dir, args.backend_url, args.task_id)
    prog.start(STAGE)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, TORCH_CUDA_ARCH_LIST=os.environ.get("TORCH_CUDA_ARCH_LIST", "7.5"),
               PYTHONUNBUFFERED="1")
    with StageMetrics(cfg.metrics_dir, "s4_train_3dgs") as m:
        try:
            t0 = time.perf_counter()
            res_dir = (train_gsplat if args.trainer == "gsplat" else train_inria)(args, cfg, iters, prog, env)
            train_s = time.perf_counter() - t0
            info = collect_outputs(args.trainer, res_dir)
            info.update(iters=iters, mode=args.mode, train_wall_seconds=round(train_s, 1), holdout_every=args.holdout_every)
            if not (info.get("ckpt") or info.get("ply")):
                raise RuntimeError(f"trainer finished but produced no checkpoint/PLY under {res_dir}")
            for k in ("result_dir", "ckpt", "ply"):          # project-relative, not '/kaggle/working/...'
                info[k] = cfg.rel(info.get(k))
            write_json(cfg.gs_dir / "latest.json", info)
            m.extra.update({k: v for k, v in info.items() if k != "stats"})
            print(f"[train] {args.trainer} {iters} iters in {train_s/60:.1f} min; "
                  f"{info.get('num_gaussians')} Gaussians; held-out {info.get('heldout')}")
            prog.finish(STAGE)
        except Exception as e:
            prog.fail(STAGE, e)
            raise


if __name__ == "__main__":
    main()

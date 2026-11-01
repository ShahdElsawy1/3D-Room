#!/usr/bin/env python3
"""Run the whole reconstruction (what the backend's "3D reconstruction" task triggers).

    python run_pipeline.py --video room.mp4 --project demo                  # production run
    python run_pipeline.py --video room.mp4 --project demo --compare        # + VGGT-vs-GLOMAP experiment
    python run_pipeline.py --project demo --from-stage train --train-mode full

Every stage runs as its OWN PROCESS (PDF 4.3) so all VRAM is released between stages.
Real progress is reported per stage to POST {backend}/tasks/{id}/progress:
    extracting frames -> estimating poses -> training -> exporting
Exit code is non-zero on failure and the failing stage's log is in <project>/metrics/logs/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common.config import Config, add_common_args, read_json  # noqa: E402
from common.progress import Progress  # noqa: E402

ORDER = ["extract", "poses", "compare", "train", "export", "reports"]


def run(script: str, args: list[str], common: list[str]) -> None:
    cmd = [sys.executable, str(HERE / script), *common, *args]
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd)
    print(f"[run_pipeline] {script} finished in {time.perf_counter() - t0:.1f}s (exit {r.returncode})", flush=True)
    if r.returncode != 0:
        raise SystemExit(f"stage failed: {script}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--video", default="", help="input room video (needed unless --from-stage is later than 'extract')")
    ap.add_argument("--method", default="vggt", choices=["vggt", "glomap", "both"],
                    help="pose method for the production run ('both' = run the experiment and let stage 3 choose)")
    ap.add_argument("--compare", action="store_true", help="shortcut for --method both")
    ap.add_argument("--fps", default="auto")
    ap.add_argument("--long-side", type=int, default=960)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--trainer", default="gsplat", choices=["gsplat", "inria"])
    ap.add_argument("--train-mode", default="quick", choices=["quick", "full"])
    ap.add_argument("--from-stage", default="extract", choices=ORDER)
    ap.add_argument("--only", default="", choices=[""] + ORDER, help="run a single stage")
    ap.add_argument("--vggt-chunk", type=int, default=32)
    ap.add_argument("--vggt-overlap", type=int, default=8)
    ap.add_argument("--gsplat-dir", default="")
    args = ap.parse_args()
    if args.compare:
        args.method = "both"

    common = ["--root", args.root, "--project", args.project]
    if args.backend_url:
        common += ["--backend-url", args.backend_url]
    if args.task_id:
        common += ["--task-id", args.task_id]
    cfg = Config(root=args.root, project_id=args.project)
    cfg.ensure()
    prog = Progress(cfg.metrics_dir, args.backend_url, args.task_id)

    stages = [args.only] if args.only else ORDER[ORDER.index(args.from_stage):]
    t_all = time.perf_counter()
    if True:  # each stage reports its own failure (with the right stage name) before exiting
        for st in stages:
            if st == "extract":
                if not args.video:
                    raise SystemExit("--video is required for the extract stage")
                run("stages/s1_extract_frames.py",
                    ["--video", args.video, "--fps", args.fps, "--long-side", str(args.long_side),
                     "--max-frames", str(args.max_frames)], common)
            elif st == "poses":
                if args.method in ("vggt", "both"):
                    run("stages/s2a_poses_vggt.py", ["--chunk-size", str(args.vggt_chunk),
                                                     "--overlap", str(args.vggt_overlap)], common)
                if args.method in ("glomap", "both"):
                    run("stages/s2b_poses_glomap.py", [], common)
            elif st == "compare":      # also (always) materialises scene/ for the trainer
                choose = "auto" if args.method == "both" else args.method
                run("stages/s3_compare_poses.py", ["--choose", choose], common)
            elif st == "train":
                extra = ["--gsplat-dir", args.gsplat_dir] if args.gsplat_dir else []
                run("stages/s4_train_3dgs.py", ["--trainer", args.trainer, "--mode", args.train_mode, *extra], common)
            elif st == "export":
                run("stages/s5_export_splat.py", [], common)
            elif st == "reports":
                run("tools/make_reports.py", ["--out", str(HERE / "docs" / "generated")], common[:4])
    if "export" in stages:          # only a run that reached export is a finished reconstruction
        prog.done()
    print(f"\n[run_pipeline] all done in {(time.perf_counter() - t_all) / 60:.1f} min. "
          f"Splat: {cfg.export_dir / 'scene.splat'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Decide VGGT vs GLOMAP by what matters: how well a splat trained on each pose set renders held-out frames.

    python tools/compare_pose_methods.py --project demo --iters 7000 [--methods vggt glomap]

For each method: stage 3 `--choose M` -> stage 4 (same frames, same iterations, same held-out split) -> result kept in
gs/cmp_<M>/.  Afterwards YOUR previous choice (scene/, gs/latest.json, gs/gsplat_<iters>) is restored, so nothing you
had before is lost. Output: metrics/pose_method_psnr.{json,md} - the table to put in the report.
Every command's output is printed (a failing step shows its own error, not just 'exit status 1').
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import add_common_args, config_from_args, read_json  # noqa: E402

HERE = Path(__file__).resolve().parents[1]


def run(cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    r = subprocess.run(cmd, text=True)
    if r.returncode != 0:
        raise SystemExit(f"step failed ({r.returncode}): {' '.join(map(str, cmd))}  -- see the output above")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--methods", nargs="+", default=["vggt", "glomap"], choices=["vggt", "glomap"])
    ap.add_argument("--iters", type=int, default=7000)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    common = ["--root", args.root, "--project", args.project]
    missing = [m for m in args.methods if not (cfg.poses_dir(m) / "sparse" / "0").exists()]
    if missing:
        raise SystemExit(f"no poses for {missing}: run stage 2a (vggt) / 2b (glomap) on this project first")

    prev = (read_json(cfg.scene_dir / "chosen.json", {}) or {}).get("method")
    latest = cfg.gs_dir / "latest.json"
    latest_bak = latest.read_text() if latest.exists() else None
    res_dir = cfg.gs_dir / f"gsplat_{args.iters}"
    aside = cfg.gs_dir / f"_before_compare_gsplat_{args.iters}"
    if res_dir.exists():
        shutil.rmtree(aside, ignore_errors=True)
        res_dir.rename(aside)
    rows = {}
    try:
        for m in args.methods:
            run([sys.executable, str(HERE / "stages" / "s3_compare_poses.py"), *common, "--choose", m])
            run([sys.executable, str(HERE / "stages" / "s4_train_3dgs.py"), *common, "--trainer", "gsplat", "--mode", "custom",
                 "--iters", str(args.iters), "--gpu", args.gpu])
            info = read_json(latest, {})
            dst = cfg.gs_dir / f"cmp_{m}"
            shutil.rmtree(dst, ignore_errors=True)
            res_dir.rename(dst)
            (dst / "latest.json").write_text(json.dumps(info, indent=1))
            ho = info.get("heldout") or {}
            gm = read_json(cfg.poses_dir(m) / ("glomap_meta.json" if m == "glomap" else "vggt_meta.json"), {}) or {}
            rows[m] = {"psnr": ho.get("psnr"), "ssim": ho.get("ssim"), "lpips": ho.get("lpips"),
                       "num_gaussians": info.get("num_gaussians"), "train_seconds": info.get("train_wall_seconds"),
                       "peak_mem_gb": info.get("trainer_peak_mem_gb"),
                       "registered": gm.get("registered_frames"), "points3D": gm.get("points3D")}
    finally:                                         # always put the user's state back
        if aside.exists():
            shutil.rmtree(res_dir, ignore_errors=True)
            aside.rename(res_dir)
        if latest_bak is not None:
            latest.write_text(latest_bak)
        if prev:
            run([sys.executable, str(HERE / "stages" / "s3_compare_poses.py"), *common, "--choose", prev])

    def f(v, nd=3):
        return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))
    md = [f"# Pose method by held-out quality ({args.iters} iterations, same frames/split/settings)", "",
          "| method | PSNR | SSIM | LPIPS | Gaussians | train s | peak GB | registered frames | sparse points |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m, r in rows.items():
        md.append(f"| {m} | {f(r['psnr'], 2)} | {f(r['ssim'])} | {f(r['lpips'])} | {r['num_gaussians']} | "
                  f"{f(r['train_seconds'], 0)} | {f(r['peak_mem_gb'], 2)} | {r['registered']} | {r['points3D']} |")
    ps = [r["psnr"] for r in rows.values() if r["psnr"] is not None]
    if len(ps) == 2:
        d = abs(ps[0] - ps[1])
        better = max(rows, key=lambda k: rows[k]["psnr"] or 0)
        md += ["", f"PSNR difference {d:.2f} dB -> " + (f"**{better}** is better." if d >= 0.3 else
               "a tie within noise (< 0.3 dB with so few held-out frames): decide by speed, registered frames and a visual check.")]
    (cfg.metrics_dir / "pose_method_psnr.md").write_text("\n".join(md) + "\n")
    (cfg.metrics_dir / "pose_method_psnr.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(md))
    print(f"\n[compare] your previous choice ({prev}) was restored; renders to inspect: gs/cmp_<method>/")


if __name__ == "__main__":
    main()

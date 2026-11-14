#!/usr/bin/env python3
"""Stage 3 - the REQUIRED VGGT vs GLOMAP experiment, plus choosing one for the project.

    python stages/s3_compare_poses.py --project demo [--choose auto|vggt|glomap]

Run stages 2a and 2b on the SAME frames first. This stage reports, per method (PDF 4.1):
    registered frames | runtime | peak VRAM | point-cloud size | visual check (PNG)
and additionally how well the two camera trajectories agree (Sim(3)-aligned on the frames
both methods registered). Output: metrics/pose_comparison.{json,md,png}.

--choose writes the winner to scene/ as the COLMAP-style folder the 3DGS trainer and
Week 2 read:  scene/images (-> frames), scene/sparse/0/{cameras,images,points3D}.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.colmap_io import camera_centers, read_model  # noqa: E402
from common.config import add_common_args, config_from_args, read_json, write_json  # noqa: E402
from common.geometry import pose_stats, robust_umeyama  # noqa: E402


def load_method(cfg, name: str):
    d = cfg.poses_dir(name)
    sparse = d / "sparse" / "0"
    if not (sparse / "cameras.bin").exists() and not (sparse / "cameras.txt").exists():
        return None
    model = read_model(sparse)
    metric = read_json(cfg.metrics_dir / f"s2{'a' if name == 'vggt' else 'b'}_poses_{name}.json", {})
    return {"name": name, "dir": d, "model": model, "metrics": metric}


def aligned_trajectory_error(a_model, b_model):
    """Sim(3)-align B onto A using camera centres of frames both registered."""
    A = {im.name: -im.R.T @ im.tvec for im in a_model.images.values()}
    B = {im.name: -im.R.T @ im.tvec for im in b_model.images.values()}
    common = sorted(set(A) & set(B))
    if len(common) < 4:
        return None
    a = np.array([A[n] for n in common]); b = np.array([B[n] for n in common])
    s, R, t, _ = robust_umeyama(b, a, keep=0.9, min_pts=4)
    b_al = s * (b @ R.T) + t
    err = np.linalg.norm(b_al - a, axis=1)
    diag = float(np.linalg.norm(a.max(0) - a.min(0))) or 1.0
    return {"common_frames": len(common), "rmse": float(np.sqrt((err ** 2).mean())),
            "median": float(np.median(err)), "max": float(err.max()),
            "rmse_over_bbox_diag": float(np.sqrt((err ** 2).mean()) / diag),
            "_a": a, "_b_aligned": b_al}


def make_plot(methods, traj, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[compare] matplotlib missing, skipping plot ({e})")
        return False

    def pca_proj(P, ref=None):
        ref = P if ref is None else ref
        mu = ref.mean(0)
        _, _, Vt = np.linalg.svd(ref - mu, full_matrices=False)
        return (P - mu) @ Vt[:2].T

    n = len(methods)
    fig, axes = plt.subplots(1, n + (1 if traj else 0), figsize=(5.2 * (n + (1 if traj else 0)), 5))
    axes = np.atleast_1d(axes)
    k = 0
    if traj:
        a, b = traj["_a"], traj["_b_aligned"]
        a2 = pca_proj(a); b2 = pca_proj(b, a)
        ax = axes[k]; k += 1
        ax.plot(a2[:, 0], a2[:, 1], "-o", ms=2, lw=1, label=f"{methods[0]['name']} (reference)")
        ax.plot(b2[:, 0], b2[:, 1], "-o", ms=2, lw=1, label=f"{methods[1]['name']} (Sim3-aligned)")
        ax.set_title(f"camera paths, RMSE/bbox = {traj['rmse_over_bbox_diag']:.3f}")
        ax.set_aspect("equal"); ax.legend()
    for mth in methods:
        ax = axes[k]; k += 1
        pts = mth["model"].points
        if len(pts.xyz) == 0:
            ax.set_title(f"{mth['name']}: no points"); continue
        sel = np.random.default_rng(0).choice(len(pts.xyz), min(60000, len(pts.xyz)), replace=False)
        P = pts.xyz[sel]
        lo, hi = np.percentile(P, [1, 99], axis=0)                 # clip outliers for display
        keep = ((P >= lo) & (P <= hi)).all(1)
        xy = pca_proj(P[keep])
        ax.scatter(xy[:, 0], xy[:, 1], s=0.3, c=pts.rgb[sel][keep] / 255.0)
        c2 = pca_proj(camera_centers(mth["model"]), P[keep])
        ax.plot(c2[:, 0], c2[:, 1], "r-", lw=0.8)
        ax.set_title(f"{mth['name']}: {len(pts.xyz):,} pts / {len(mth['model'].images)} cams")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return True


def suggest(vggt, glomap, traj_err) -> tuple[str, str]:
    """Data-driven default + a draft justification paragraph (edit it in your report)."""
    nv, ng = len(vggt["model"].images), len(glomap["model"].images)
    total = max(nv, ng)
    sv = vggt["metrics"].get("seconds", float("nan")); sg = glomap["metrics"].get("seconds", float("nan"))
    pick = "vggt"
    reason = []
    if ng < 0.9 * nv:
        reason.append(f"GLOMAP registered only {ng}/{nv} frames vs {nv} for VGGT")
    elif nv < 0.9 * ng:
        pick = "glomap"; reason.append(f"VGGT registered only {nv}/{ng} frames vs {ng} for GLOMAP")
    else:
        # both register (almost) everything: break the tie on runtime and drift evidence
        if sg > 3 * sv:
            reason.append(f"both register ~all frames but VGGT is {sg/max(sv,1e-9):.1f}x faster ({sv:.0f}s vs {sg:.0f}s)")
        else:
            pick = "glomap"
            reason.append("both register all frames and runtimes are comparable, so the more accurate classical baseline wins")
    if traj_err and traj_err["rmse_over_bbox_diag"] > 0.05 and pick == "vggt":
        reason.append(f"note: trajectories differ by {traj_err['rmse_over_bbox_diag']*100:.1f}% of the scene size - "
                      "inspect the plot for VGGT drift before committing")
    para = (f"We chose {pick.upper()} for the project because " + "; ".join(reason) + ". "
            f"VGGT: {nv} frames, {sv:.0f}s, {vggt['metrics'].get('peak_vram_mb', 0):.0f} MiB peak VRAM. "
            f"GLOMAP: {ng} frames, {sg:.0f}s, {glomap['metrics'].get('peak_vram_mb', 0):.0f} MiB peak VRAM.")
    return pick, para


def choose(cfg, name: str):
    src = cfg.poses_dir(name)
    dst = cfg.scene_dir
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "sparse").mkdir(parents=True)
    shutil.copytree(src / "sparse" / "0", dst / "sparse" / "0")
    img_src = (src / "images").resolve()
    try:
        (dst / "images").symlink_to(os.path.relpath(img_src, dst))
    except OSError:
        shutil.copytree(img_src, dst / "images")
    for extra in ("points.ply",):
        if (src / extra).exists():
            shutil.copy2(src / extra, dst / extra)
    # project-relative paths (they stay valid after the project folder is copied to another machine)
    write_json(dst / "chosen.json", {"method": name, "from": cfg.rel(src),
                                     "depth_dir": cfg.rel(src / "depth") if (src / "depth").exists() else None})
    print(f"[compare] scene/ <- {name}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--choose", default="auto", choices=["auto", "vggt", "glomap", "none"])
    args = ap.parse_args(argv)
    cfg = config_from_args(args)

    vggt, glomap = load_method(cfg, "vggt"), load_method(cfg, "glomap")
    have = [m for m in (vggt, glomap) if m]
    if not have:
        raise SystemExit("run stage 2a and/or 2b first")

    rows = []
    for m in have:
        mm, md = m["metrics"], m["model"]
        rows.append({
            "method": m["name"], "registered_frames": len(md.images),
            "runtime_s": mm.get("seconds"), "peak_vram_mb": mm.get("peak_vram_mb"),
            "torch_peak_alloc_mb": mm.get("torch_peak_allocated_mb"),
            "points3D": int(len(md.points.xyz)),
            "trajectory": pose_stats(camera_centers(md)),
            "extra": {k: mm.get(k) for k in ("chunks", "chunk_size", "dtype", "backend", "seconds_breakdown")
                      if mm.get(k) is not None},
        })
    traj = aligned_trajectory_error(vggt["model"], glomap["model"]) if vggt and glomap else None
    png = cfg.metrics_dir / "pose_comparison.png"
    plotted = make_plot(have, traj, png)

    pick, para = (None, None)
    if vggt and glomap:
        pick, para = suggest(vggt, glomap, traj)
    result = {"frames_total": len(list(cfg.images_dir.glob("*.jpg"))), "methods": rows,
              "trajectory_agreement": {k: v for k, v in (traj or {}).items() if not k.startswith("_")} or None,
              "suggested_choice": pick, "draft_justification": para, "plot": cfg.rel(png) if plotted else None}
    write_json(cfg.metrics_dir / "pose_comparison.json", result)

    md = ["# VGGT vs GLOMAP", "", f"Frames fed to both methods: **{result['frames_total']}**", "",
          "| method | registered | runtime (s) | peak VRAM (MiB) | points3D | path length |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['method']} | {r['registered_frames']} | {r['runtime_s']} | {r['peak_vram_mb']} | "
                  f"{r['points3D']:,} | {r['trajectory']['path_length']:.3f} |")
    if traj:
        t = result["trajectory_agreement"]
        md += ["", f"Trajectory agreement (Sim3-aligned, {t['common_frames']} common frames): "
               f"RMSE = {t['rmse_over_bbox_diag']*100:.2f}% of scene extent."]
    if plotted:
        md += ["", "![visual check](pose_comparison.png)"]
    if para:
        md += ["", "## Draft justification (edit before submitting)", "", para]
    (cfg.metrics_dir / "pose_comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    choice = args.choose
    if choice == "auto":
        choice = pick or have[0]["name"]
    if choice != "none":
        if not any(m["name"] == choice for m in have):
            raise SystemExit(f"cannot choose {choice}: stage output missing")
        choose(cfg, choice)


if __name__ == "__main__":
    main()

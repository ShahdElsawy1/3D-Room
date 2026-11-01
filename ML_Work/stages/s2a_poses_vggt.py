#!/usr/bin/env python3
"""Stage 2a - camera poses with VGGT (feed-forward), chunked + fp16, merged with Sim(3).

    python stages/s2a_poses_vggt.py --project demo [--chunk-size 32 --overlap 8]

Outputs (poses/vggt/):
    images/                 -> symlink to frames/images (COLMAP-style folder)
    sparse/0/{cameras,images,points3D}.{bin,txt}
    depth/000001.npy        float16 z-depth at VGGT resolution   (Week 2 "lift to 3D")
    conf/000001.npy         float16 confidence
    points.ply              coloured sparse cloud for the visual check / three.js fallback
    vggt_meta.json          intrinsics (VGGT-res + original-res), chunk plan, merge diagnostics
Key points
  * T4 (compute capability 7.5) has no native bfloat16 -> float16 autocast. "auto" picks bf16 on cc >= 8.
  * Memory grows with the number of frames -> process `chunk_size` frames at a time with
    `overlap` shared frames, then fit a Sim(3) per chunk pair from the shared frames' depth.
  * VGGT works on a 518 px long side (multiple of 14). Intrinsics are rescaled to the real
    frame size so the COLMAP folder is directly usable by 3DGS.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.colmap_io import Camera, Image as CImage, Points3D, SparseModel, write_model, write_points_ply  # noqa: E402
from common.config import add_common_args, config_from_args, read_json, write_json  # noqa: E402
from common.geometry import rescale_intrinsics, rotmat2qvec, unproject_depth  # noqa: E402
from common.metrics import StageMetrics  # noqa: E402
from common.progress import Progress  # noqa: E402
from common.vggt_merge import Chunk, merge_chunks, plan_chunks  # noqa: E402

STAGE = "estimating poses"


# ------------------------------------------------------------------ helpers
def vggt_size_for(w: int, h: int, long_side: int = 518, patch: int = 14) -> tuple[int, int]:
    """Keep the aspect ratio, long side = 518, both sides multiples of 14 (no cropping)."""
    if w >= h:
        W = long_side
        H = max(patch, int(round(long_side * h / w / patch)) * patch)
    else:
        H = long_side
        W = max(patch, int(round(long_side * w / h / patch)) * patch)
    return W, H


def load_tensor(path: Path, size_wh):
    import torch
    im = Image.open(path).convert("RGB").resize(size_wh, Image.BICUBIC)
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).permute(2, 0, 1)


def import_vggt():
    try:
        from vggt.models.vggt import VGGT  # noqa
        return
    except ImportError:
        pass
    for cand in (os.environ.get("VGGT_DIR", ""), "/kaggle/working/vggt", "vggt", "../vggt"):
        if cand and (Path(cand) / "vggt").exists():
            sys.path.insert(0, str(Path(cand).resolve()))
            return
    raise ImportError("VGGT is not installed. Run scripts/setup_kaggle.sh (or: "
                      "git clone https://github.com/facebookresearch/vggt && export VGGT_DIR=<path>)")


def pick_dtype(name: str):
    import torch
    if not torch.cuda.is_available():
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16     # T4 = 7.5 -> float16


def load_model(model_id: str, device):
    import torch
    from vggt.models.vggt import VGGT
    try:
        model = VGGT.from_pretrained(model_id)
    except Exception as e:  # offline / hub issues: fall back to the raw checkpoint URL
        print(f"[vggt] from_pretrained failed ({e}); trying torch.hub checkpoint", flush=True)
        model = VGGT()
        url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(url))
    return model.eval().to(device)


def run_chunk(model, imgs, dtype, device):
    """imgs: (S,3,H,W) on device -> numpy (R,t,K,depth,conf) in this chunk's local frame."""
    import torch
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    use_cuda = device.type == "cuda"
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_cuda and dtype != torch.float32):
            images = imgs[None]
            tokens, ps_idx = model.aggregator(images)
            pose_enc = model.camera_head(tokens)[-1]
            extri, intri = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            depth, conf = model.depth_head(tokens, images, ps_idx)
    extri = extri[0].float().cpu().numpy()          # (S,3,4) OpenCV, camera-from-world
    intri = intri[0].float().cpu().numpy()          # (S,3,3)
    depth = depth[0, ..., 0].float().cpu().numpy()  # (S,H,W)
    conf = conf[0].float().cpu().numpy()
    del tokens, pose_enc, images
    if use_cuda:
        torch.cuda.empty_cache()
    return extri[:, :, :3], extri[:, :, 3], intri, depth, conf


def link_or_copy_images(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink():
            dst.unlink()
        else:
            return
    try:
        dst.symlink_to(os.path.relpath(src, dst.parent))
    except OSError:
        import shutil
        shutil.copytree(src, dst)


def build_points(R, t, K, depth, conf, rgb_frames, cfg, rng):
    """Sparse initial point cloud from depth maps, filtered by confidence and depth."""
    thr = np.percentile(conf, cfg.conf_percentile)
    pts, cols = [], []
    stride = cfg.pixel_stride
    for i in range(len(R)):
        d = depth[i]
        ok = (conf[i] >= thr) & (d > 0) & np.isfinite(d)
        if ok.sum() == 0:
            continue
        ok &= d <= np.percentile(d[ok], 98)                  # drop far-away sky/background spikes
        mask = np.zeros_like(ok)
        mask[::stride, ::stride] = True
        sel = ok & mask
        P = unproject_depth(d.astype(np.float64), K[i], R[i], t[i])[sel]
        pts.append(P); cols.append(rgb_frames[i][sel])
    P = np.concatenate(pts); C = np.concatenate(cols)
    # robust outlier trim around the median, then voxel de-duplication
    med = np.median(P, 0)
    dist = np.linalg.norm(P - med, axis=1)
    keep = dist <= np.percentile(dist, 99.5)
    P, C = P[keep], C[keep]
    scale = float(np.percentile(np.linalg.norm(P - med, axis=1), 90)) or 1.0
    vox = np.floor(P / (scale / 400.0)).astype(np.int64)
    _, uniq = np.unique(vox, axis=0, return_index=True)
    P, C = P[uniq], C[uniq]
    if len(P) > cfg.max_points:
        sel = rng.choice(len(P), cfg.max_points, replace=False)
        P, C = P[sel], C[sel]
    return P, C


def memory_sweep(model, paths, size_wh, dtype, device, sizes, out_path):
    """Measure how VGGT peak VRAM / time scale with the number of frames in one chunk."""
    import torch
    rows = []
    for s in sizes:
        if s > len(paths):
            continue
        idx = np.linspace(0, len(paths) - 1, s).round().astype(int)
        imgs = torch.stack([load_tensor(paths[i], size_wh) for i in idx]).to(device)
        if device.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            run_chunk(model, imgs, dtype, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            rows.append({"frames": s, "seconds": round(time.perf_counter() - t0, 2),
                         "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.type == "cuda" else None})
        except RuntimeError as e:      # CUDA OOM at this size is itself a result
            rows.append({"frames": s, "error": str(e).splitlines()[0]})
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del imgs
    write_json(out_path, {"dtype": str(dtype), "size_wh": list(size_wh), "rows": rows})
    print("[vggt] memory sweep:", rows)


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--conf-percentile", type=float, default=40)
    ap.add_argument("--max-points", type=int, default=300_000)
    ap.add_argument("--per-frame-intrinsics", action="store_true",
                    help="keep VGGT's per-frame K (default: one shared camera = median K)")
    ap.add_argument("--memory-sweep", default="", help="e.g. 8,16,32,48 : extra VRAM-vs-frames measurement")
    args = ap.parse_args(argv)

    import torch
    cfg = config_from_args(args)
    v = cfg.vggt
    v.chunk_size, v.overlap, v.dtype = args.chunk_size, args.overlap, args.dtype
    v.conf_percentile, v.max_points = args.conf_percentile, args.max_points
    prog = Progress(cfg.metrics_dir, args.backend_url, args.task_id)
    prog.start(STAGE)

    out = cfg.poses_dir("vggt")
    paths = sorted(cfg.images_dir.glob("*.jpg"))
    if not paths:
        prog.fail(STAGE, "no frames - run stage 1 first")
        raise SystemExit("no frames found in " + str(cfg.images_dir))

    with StageMetrics(cfg.metrics_dir, "s2a_poses_vggt") as m:
        try:
            import_vggt()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = pick_dtype(v.dtype)
            print(f"[vggt] device={device} dtype={dtype} frames={len(paths)}", flush=True)
            if device.type == "cpu" and len(paths) > 16:
                print("[vggt] WARNING: running on CPU - expect this to be very slow", flush=True)
            W0, H0 = Image.open(paths[0]).size
            size_wh = vggt_size_for(W0, H0, v.vggt_size)
            sx, sy = W0 / size_wh[0], H0 / size_wh[1]

            t_load = time.perf_counter()
            model = load_model(v.model_id, device)
            m.extra["model_load_s"] = round(time.perf_counter() - t_load, 2)
            prog.report(STAGE, 0.05)

            if args.memory_sweep:
                memory_sweep(model, paths, size_wh, dtype, device,
                             [int(x) for x in args.memory_sweep.split(",")],
                             cfg.metrics_dir / "vggt_memory_sweep.json")

            plan = plan_chunks(len(paths), v.chunk_size, v.overlap)
            print(f"[vggt] {len(plan)} chunk(s): {plan}", flush=True)
            chunks, rgb_frames, chunk_secs = [], [None] * len(paths), []
            for ci, (s, e) in enumerate(plan):
                t0 = time.perf_counter()
                imgs = torch.stack([load_tensor(p, size_wh) for p in paths[s:e]])
                for j in range(e - s):       # keep colours (uint8, VGGT res) for the point cloud
                    if rgb_frames[s + j] is None:
                        rgb_frames[s + j] = (imgs[j].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                R, t, K, d, c = run_chunk(model, imgs.to(device), dtype, device)
                chunks.append(Chunk(s, R.astype(np.float64), t.astype(np.float64), K.astype(np.float64), d, c))
                chunk_secs.append(round(time.perf_counter() - t0, 2))
                prog.report(STAGE, 0.05 + 0.6 * (ci + 1) / len(plan))
            peak_torch = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            R, t, K, depth, conf, diag = merge_chunks(chunks)
            prog.report(STAGE, 0.75)
            bad = [d for d in diag if d["rmse_over_median_depth"] > 0.05]
            if bad:
                print(f"[vggt] WARNING: weak chunk alignment (rmse/median depth > 5%): {bad}", flush=True)

            # ---- intrinsics: VGGT-res -> original frame size
            K_orig = np.array([rescale_intrinsics(Ki, sx, sy) for Ki in K])
            if args.per_frame_intrinsics:
                cams = {i + 1: Camera(i + 1, "PINHOLE", W0, H0,
                                      np.array([K_orig[i, 0, 0], K_orig[i, 1, 1], K_orig[i, 0, 2], K_orig[i, 1, 2]]))
                        for i in range(len(paths))}
                cam_of = lambda i: i + 1  # noqa: E731
            else:
                fx, fy = float(np.median(K_orig[:, 0, 0])), float(np.median(K_orig[:, 1, 1]))
                cams = {1: Camera(1, "PINHOLE", W0, H0, np.array([fx, fy, W0 / 2.0, H0 / 2.0]))}
                cam_of = lambda i: 1  # noqa: E731

            images = {i + 1: CImage(i + 1, rotmat2qvec(R[i]), t[i], cam_of(i), paths[i].name)
                      for i in range(len(paths))}
            rng = np.random.default_rng(0)
            P, C = build_points(R, t, K, depth, conf, rgb_frames, v, rng)
            model_out = SparseModel(cams, images, Points3D(P, C))
            write_model(model_out, out / "sparse" / "0")
            write_points_ply(out / "points.ply", P, C)
            link_or_copy_images(cfg.images_dir, out / "images")

            # ---- per-frame depth/confidence for Week 2 (back-project the object mask)
            (out / "depth").mkdir(exist_ok=True); (out / "conf").mkdir(exist_ok=True)
            for i, p in enumerate(paths):
                np.save(out / "depth" / f"{p.stem}.npy", depth[i].astype(np.float16))
                np.save(out / "conf" / f"{p.stem}.npy", conf[i].astype(np.float16))
            prog.report(STAGE, 0.95)

            write_json(out / "vggt_meta.json", {
                "model": v.model_id, "dtype": str(dtype), "vggt_size_wh": list(size_wh),
                "frame_size_wh": [W0, H0], "scale_to_frame_xy": [sx, sy],
                "depth_note": "depth/*.npy is z-depth at vggt_size_wh in the SAME world/scale as sparse/0; "
                              "intrinsics for it = intrinsics_vggt_res",
                "chunk_plan": plan, "chunk_seconds": chunk_secs, "merge": diag,
                "shared_camera": not args.per_frame_intrinsics,
                "intrinsics_vggt_res": K.tolist() if len(paths) <= 600 else "see depth maps",
                "intrinsics_frame_res": K_orig.tolist() if len(paths) <= 600 else "see cameras.txt",
                "frames": [p.name for p in paths],
            })
            c0 = cams[1]
            m.extra.update(frames=len(paths), registered_frames=len(images), chunks=len(plan),
                           chunk_size=v.chunk_size, overlap=v.overlap, dtype=str(dtype),
                           points3D=int(len(P)), torch_peak_alloc_mb_all_chunks=peak_torch,
                           fx=float(c0.params[0]), fy=float(c0.params[1]),
                           max_merge_rmse_over_depth=max([d["rmse_over_median_depth"] for d in diag] or [0.0]))
            print(f"[vggt] done: {len(images)} cameras, {len(P)} points -> {out}")
            prog.report(STAGE, 1.0)
        except Exception as e:
            prog.fail(STAGE, e)
            raise


if __name__ == "__main__":
    main()

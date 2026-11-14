#!/usr/bin/env python3
"""Stage 6 - save everything Week 2 needs, next to the Week 1 outputs.

    python stages/s6_prepare_week2.py --project demo [--render-depth auto|on|off]

Writes <project>/week2/ (all paths inside are PROJECT-RELATIVE):
    inputs.json          the one file Week 2 code reads: image spaces + sizes, poses/intrinsics, held-out split,
                         depth sources, splat/checkpoint locations, and `required_files` (what a worker must upload)
    sparse_depth/<stem>.npy   (N,3) float32 [u, v, z] of the COLMAP points seen by that frame, in scene/images pixels
                              (CPU, always written; enough for a visibility check when lifting a mask to 3D)
    depth/<stem>.npy          dense z-depth float16 rendered from the trained splat at --depth-scale x scene/images
                              resolution, 0 = nothing there. Needs a GPU + gsplat + the trainer checkpoint.
                              Valid for ANY pose method (unlike the VGGT depth, which only matches VGGT poses).

It never changes Week 1 outputs and a failure here never fails the Week 1 run (use --strict to make it fail).
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.colmap_io import read_model  # noqa: E402
from common.config import add_common_args, config_from_args, read_json, write_json  # noqa: E402
from common import gs_space  # noqa: E402
from common.geometry import project  # noqa: E402
from common.metrics import StageMetrics  # noqa: E402

SCHEMA = 1


def image_size(path: Path):
    import cv2
    im = cv2.imread(str(path))
    return None if im is None else [int(im.shape[1]), int(im.shape[0])]


def sparse_depth(cfg, model, out_dir: Path, max_points: int = 20000) -> int:
    """Per frame: the COLMAP points that project inside the image and in front of the camera -> (u, v, z)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    xyz = model.points.xyz
    rng = np.random.default_rng(0)
    n = 0
    for im in sorted(model.images.values(), key=lambda i: i.name):
        cam = model.cameras[im.camera_id]
        uv, z = project(xyz, cam.K(), im.R, im.tvec)
        ok = (z > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
        arr = np.concatenate([uv[ok], z[ok, None]], axis=1).astype(np.float32)
        if len(arr) > max_points:
            arr = arr[rng.choice(len(arr), max_points, replace=False)]
        np.save(out_dir / f"{Path(im.name).stem}.npy", arr)
        n += 1
    return n


def render_splat_depth(cfg, ckpt: Path, model, out_dir: Path, scale: float, poses=None) -> dict:
    """Expected depth ('ED') of the trained gsplat checkpoint from every training camera. GPU only."""
    import torch
    from gsplat import rasterization
    dev = "cuda:0"
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    s = {k: v.to(dev).float() for k, v in ck["splats"].items()}
    means, quats = s["means"], s["quats"]
    scales, opac = torch.exp(s["scales"]), torch.sigmoid(s["opacities"])
    colors = torch.zeros(len(means), 3, device=dev)          # only depth is wanted
    out_dir.mkdir(parents=True, exist_ok=True)
    n, size = 0, None
    with torch.no_grad():
        for im in sorted(model.images.values(), key=lambda i: i.name):
            cam = model.cameras[im.camera_id]
            W, H = cam.width, cam.height
            K = torch.from_numpy(cam.K()).float().to(dev)[None]
            V = np.eye(4, dtype=np.float32)
            if poses is not None and im.name in poses:        # poses in the checkpoint's space (common/gs_space.py)
                V = np.asarray(poses[im.name], dtype=np.float32)
            else:
                V[:3, :3], V[:3, 3] = im.R, im.tvec
            res, alpha, _ = rasterization(means, quats, scales, opac, colors, torch.from_numpy(V).to(dev)[None], K,
                                          W, H, render_mode="ED")
            d = res[0, ..., 0]
            d = torch.where(alpha[0, ..., 0] > 0.5, d, torch.zeros_like(d))
            if scale != 1.0:
                d = torch.nn.functional.interpolate(d[None, None], scale_factor=scale, mode="nearest")[0, 0]
            arr = d.cpu().numpy().astype(np.float16)
            size = [int(arr.shape[1]), int(arr.shape[0])]
            np.save(out_dir / f"{Path(im.name).stem}.npy", arr)
            n += 1
    return {"frames": n, "size_wh": size, "scale": scale}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--render-depth", default="auto", choices=["auto", "on", "off"],
                    help="auto: render when a GPU, gsplat and the checkpoint are there; on: required; off: skip")
    ap.add_argument("--depth-scale", type=float, default=0.5, help="rendered depth size relative to scene/images")
    ap.add_argument("--holdout-every", type=int, default=0, help="override; default = value recorded by stage 4 (or 8)")
    ap.add_argument("--strict", action="store_true", help="exit non-zero on failure (default: warn and exit 0)")
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    R = cfg.rel
    wk = cfg.p("week2")

    try:
        with StageMetrics(cfg.metrics_dir, "s6_prepare_week2") as m:
            t0 = time.perf_counter()
            sparse = cfg.scene_dir / "sparse" / "0"
            if not sparse.exists():
                raise SystemExit("scene/ missing - run stage 3 first")
            model = read_model(sparse)
            chosen = read_json(cfg.scene_dir / "chosen.json", {})
            method = chosen.get("method")
            idx = read_json(cfg.frames_index, {})
            latest = read_json(cfg.gs_dir / "latest.json", {})
            meta = read_json(cfg.poses_dir(method or "glomap") / ("glomap_meta.json" if method == "glomap" else "vggt_meta.json"), {})

            names = sorted(im.name for im in model.images.values())
            scene_img_dir = cfg.scene_dir / "images"
            frames_size = idx.get("image_size_wh") or (image_size(cfg.images_dir / names[0]) if names else None)
            scene_size = image_size(scene_img_dir / names[0]) if names and scene_img_dir.exists() else None
            undistorted = bool(meta.get("undistorted"))
            same_pixels = bool(frames_size and scene_size and frames_size == scene_size and not undistorted)
            every = args.holdout_every or latest.get("holdout_every") or 8
            all_names = sorted(p.name for p in scene_img_dir.glob("*.jpg")) if scene_img_dir.exists() else names
            heldout = [n for i, n in enumerate(all_names) if i % every == 0]    # gsplat: val = index % test_every == 0

            n_sparse = sparse_depth(cfg, model, wk / "sparse_depth")

            depth = {"kind": "sparse_only", "dir": None}
            ckpt = cfg.resolve(latest.get("ckpt"))
            want = args.render_depth != "off"
            if want and ckpt and ckpt.exists():
                try:
                    info = render_splat_depth(cfg, ckpt, model, wk / "depth", args.depth_scale,
                                              poses=gs_space.load_poses(wk / gs_space.FILE))
                    depth = {"kind": "splat_rendered", "dir": R(wk / "depth"),
                             "format": ".npy float16, H x W, z-depth in scene units, 0 = no surface",
                             "image_space": "scene/images scaled by `scale`", **info}
                except Exception as e:                      # no GPU / no gsplat: expected on a CPU run
                    if args.render_depth == "on":
                        raise
                    depth["note"] = f"splat depth not rendered ({type(e).__name__}: {e})"
                    print(f"[week2] {depth['note']}", flush=True)
            elif args.render_depth == "on":
                raise SystemExit("--render-depth on, but there is no trainer checkpoint (gs/latest.json -> ckpt)")
            else:
                depth["note"] = "no checkpoint (train first, or upload gs/) - splat depth skipped"
            vggt_depth = cfg.poses_dir("vggt") / "depth"
            depth["vggt_depth"] = ({"dir": R(vggt_depth), "conf_dir": R(cfg.poses_dir("vggt") / "conf"),
                                    "meta": R(cfg.poses_dir("vggt") / "vggt_meta.json"),
                                    "valid_only_if_pose_method": "vggt", "pose_method_now": method}
                                   if vggt_depth.exists() else None)

            required = [R(cfg.frames_index), R(cfg.images_dir), R(cfg.scene_dir / "sparse"), R(cfg.scene_dir / "images"),
                        R(cfg.scene_dir / "chosen.json"), R(cfg.export_dir / "cameras.json"),
                        R(cfg.export_dir / "scene_meta.json"), R(cfg.export_dir / "scene.splat"),
                        R(wk / "inputs.json"), R(wk / "sparse_depth")]
            if (wk / "depth").exists():
                required.append(R(wk / "depth"))
            if cfg.resolve(latest.get("ckpt")) and cfg.resolve(latest.get("ckpt")).exists():
                required += [R(cfg.gs_dir / "latest.json"), R(cfg.resolve(latest["ckpt"]))]
            if depth["vggt_depth"] and method == "vggt":
                required += [R(vggt_depth), R(cfg.poses_dir("vggt") / "vggt_meta.json")]
            required = [r for r in dict.fromkeys(required) if r == R(wk / "inputs.json") or (cfg.p(r)).exists()]

            cam0 = next(iter(model.cameras.values())) if model.cameras else None
            inputs = {
                "schema_version": SCHEMA, "project": cfg.project_id, "paths_are_relative_to": "the project folder",
                "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "pose_method": method, "registered_frames": len(model.images), "total_frames": idx.get("kept"),
                "images": {
                    "frames_dir": R(cfg.images_dir), "scene_dir": R(scene_img_dir), "names": names,
                    "name_rule": "same file name in frames/images, scene/images, the COLMAP model and every mask/edit",
                    "frames_size_wh": frames_size, "scene_size_wh": scene_size,
                    "scene_is_undistorted_copy": undistorted, "same_pixels": same_pixels,
                    "raw_camera": meta.get("raw_camera") if undistorted else None,   # distortion of frames/images
                    "recommended_edit_space": "scene/images",
                    "why": ("frames/images and scene/images are pixel-identical" if same_pixels else
                            "scene/images is the undistorted copy the splat was trained on (different size / pixels). "
                            "Masks and edits made on frames/images do not line up pixel-for-pixel with it; make them on "
                            "scene/images, or warp them with the camera distortion in poses/<method>/glomap_meta.json."),
                },
                "poses": {"sparse": R(sparse), "format": "COLMAP bin+txt, world->camera, OpenCV axes (+y down)",
                          "camera": ({"model": cam0.model, "width": cam0.width, "height": cam0.height,
                                      "params": cam0.params.tolist()} if cam0 else None),
                          "cameras_json": R(cfg.export_dir / "cameras.json")},
                "heldout": {"every": every, "rule": "sorted scene image names, index % every == 0 are NOT trained on",
                            "names": heldout},
                "depth": depth,
                "sparse_depth": {"dir": R(wk / "sparse_depth"), "format": ".npy (N,3) float32 [u, v, z] in scene/images "
                                 "pixels, points3D visible in that frame", "frames": n_sparse},
                "splat": {"ply": R(cfg.export_dir / "scene.ply"), "splat": R(cfg.export_dir / "scene.splat"),
                          "meta": R(cfg.export_dir / "scene_meta.json")},
                "checkpoint": R(ckpt) if ckpt and ckpt.exists() else None,
                "trainer": {k: latest.get(k) for k in ("trainer", "iters", "mode", "num_gaussians", "heldout")},
                "rebuild_rule": "reuse poses + intrinsics (no new SfM); swap edited images into a COPY of scene/images; "
                                "keep the original scene/ gs/ export/ untouched",
                "required_files": required,
            }
            write_json(wk / "inputs.json", inputs)
            m.extra.update(sparse_depth_frames=n_sparse, depth_kind=depth["kind"], seconds_total=round(time.perf_counter() - t0, 1))
            print(f"[week2] wrote {R(wk / 'inputs.json')}: {len(names)} frames, depth={depth['kind']}, "
                  f"edit space {'identical' if same_pixels else 'DIFFERS'} between frames/ and scene/images", flush=True)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[week2] FAILED (Week 1 outputs are fine): {type(e).__name__}: {e}", flush=True)
        if args.strict:
            raise


if __name__ == "__main__":
    main()

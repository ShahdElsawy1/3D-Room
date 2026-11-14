#!/usr/bin/env python3
"""Week 2 / C with A - rebuild the splat from the edited frames. Poses and intrinsics are REUSED (no new SfM).

    python stages/w2_rebuild.py --project-dir P --edit-id E --mode finetune   # start from the trained Gaussians
    python stages/w2_rebuild.py --project-dir P --edit-id E --mode scratch    # train again from the sparse points

Inputs : scene/ (sparse/0 + images), edits/E/edited/index.json + edited/<stem>.png (B), edits/E/masks (A),
         week2/inputs.json, gs/latest.json -> checkpoint (finetune only; upload gs/ with the worker's --upload-checkpoint)
Output : edits/E/export/{scene.ply, scene.splat, cameras.json, scene_meta.json}   (same names as Week 1)
         edits/E/metrics/{s4_rebuild.json, rebuild_<mode>.json}   (seconds, peak VRAM, #Gaussians, held-out PSNR/SSIM,
                                                                   how much the JPEG re-encode changed outside the mask)
         edits/E/rebuilds/<mode>/   archive of this mode's export + report (run both modes -> compare)
The original scene/, gs/ and export/ are never touched: everything is written below edits/E/.

Edited frames and JPEG: B saves edited frames as PNG (lossless). gsplat reads training images by file name, so the
rebuild scene gets `scene/images/<name>.jpg` = edited frame encoded at JPEG quality 100 / 4:4:4 (pixel change <= ~1-2
levels, measured and reported as `outside_mask_*`). Frames that were not edited are symlinks to the originals.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from common import masks as M  # noqa: E402
from common import undist  # noqa: E402
from common.config import Config, read_json, write_json  # noqa: E402
from common.metrics import StageMetrics  # noqa: E402
from common.w2_common import base_parser, cfg_from_dir, dump_json, edit_progress, load_inputs, space_maps  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
DEFAULT_ITERS = {"finetune": 3000, "scratch": 7000}


def read_edited_items(E: Path) -> list[dict]:
    """edits/E/edited/index.json -> [{"name": "000012.jpg", "file": "edited/000012.png"}], tolerant about extra keys."""
    idx = json.loads((E / "edited" / "index.json").read_text())
    items = []
    for e in idx.get("edited", idx if isinstance(idx, list) else []):
        f = e["file"] if isinstance(e, dict) else e
        stem = Path(e.get("name", f) if isinstance(e, dict) else f).stem
        items.append({"stem": stem, "file": f})
    return items


def _jpeg_params():
    p = [cv2.IMWRITE_JPEG_QUALITY, 100]
    if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR") and hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444"):
        p += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    return p


def build_rebuild_scene(cfg: Config, edit_id: str, inputs: dict, guard_px: int = 6) -> dict:
    """edits/E/scene = copy of scene/sparse + images (symlinks, with the edited frames written as JPEG q100)."""
    E = M.edit_dir(cfg.project, edit_id)
    dst = E / "scene"
    shutil.rmtree(dst, ignore_errors=True)
    (dst / "images").mkdir(parents=True)
    shutil.copytree(cfg.scene_dir / "sparse", dst / "sparse")
    if (cfg.scene_dir / "chosen.json").exists():
        shutil.copy2(cfg.scene_dir / "chosen.json", dst / "chosen.json")
    items = {i["stem"]: i for i in read_edited_items(E)}
    f2s, s2f = space_maps(cfg, inputs)
    src_imgs = cfg.scene_dir / "images"
    stats = {"edited_frames": 0, "outside_mask_max_abs_diff": 0, "outside_mask_mean_abs_diff": 0.0,
             "outside_mask_share_diff_gt2": 0.0}
    diffs_mean, diffs_share = [], []
    for p in sorted(src_imgs.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        out = dst / "images" / p.name
        it = items.get(p.stem)
        if it is None:
            try:
                os.symlink(os.path.relpath(p.resolve(), out.parent.resolve()), out)
            except OSError:
                shutil.copy2(p, out)
            continue
        ed = cv2.imread(str(E / it["file"]), cv2.IMREAD_COLOR)
        if ed is None:
            raise FileNotFoundError(E / it["file"])
        orig = cv2.imread(str(p), cv2.IMREAD_COLOR)
        mk = M.load_mask(E / "masks" / (p.stem + ".png"))
        if s2f is not None:                                   # edits live in frames/images space, the splat in scene/
            ed = undist.remap(ed, s2f)
            mk = None if mk is None else undist.remap(mk.astype(np.uint8) * 255, s2f, nearest=True) > 0
        if ed.shape != orig.shape:
            raise RuntimeError(f"{p.name}: edited frame {ed.shape[1::-1]} != scene image {orig.shape[1::-1]}")
        cv2.imwrite(str(out), ed, _jpeg_params())
        back = cv2.imread(str(out), cv2.IMREAD_COLOR)
        if mk is not None:                                    # B's guarantee: nothing changed outside the dilated mask
            k = 2 * guard_px + 1
            outside = ~(cv2.dilate(mk.astype(np.uint8), np.ones((k, k), np.uint8)) > 0)
            d = np.abs(back.astype(np.int16) - orig.astype(np.int16)).max(axis=2)[outside]
            if d.size:
                stats["outside_mask_max_abs_diff"] = max(stats["outside_mask_max_abs_diff"], int(d.max()))
                diffs_mean.append(float(d.mean()))
                diffs_share.append(float((d > 2).mean()))
        stats["edited_frames"] += 1
    if diffs_mean:
        stats["outside_mask_mean_abs_diff"] = round(float(np.mean(diffs_mean)), 4)
        stats["outside_mask_share_diff_gt2"] = round(float(np.mean(diffs_share)), 6)
    if not stats["edited_frames"]:
        raise SystemExit("edits/E/edited/index.json lists no frame that exists in scene/images")
    return stats


def sh(cmd, env=None):
    print("$", " ".join(map(str, cmd)), flush=True)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(map(str, cmd))}")


def main(argv=None):
    ap = base_parser(__doc__)
    ap.add_argument("--mode", required=True, choices=["finetune", "scratch"])
    ap.add_argument("--iters", type=int, default=0, help=f"default: {DEFAULT_ITERS}")
    ap.add_argument("--lr-scale", type=float, default=0.25, help="finetune: scale of the training learning rates")
    ap.add_argument("--no-densify", action="store_true", help="finetune: keep the number of Gaussians fixed")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args(argv)
    cfg = cfg_from_dir(args.project_dir)
    E = M.edit_dir(cfg.project, args.edit_id)
    iters = args.iters or DEFAULT_ITERS[args.mode]
    inputs = load_inputs(cfg)
    every = (inputs.get("heldout") or {}).get("every") or 8
    prog = edit_progress(cfg, args.edit_id, args)
    prog.report("training", 0.0, force=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1")
    ecfg = Config(root=str(cfg.project / "edits"), project_id=args.edit_id)   # edits/E/{scene,gs,export,metrics}
    ecfg.ensure()

    with StageMetrics(E / "metrics", "w2_rebuild") as m:
        try:
            t0 = time.perf_counter()
            prep = build_rebuild_scene(cfg, args.edit_id, inputs)
            t_prep = time.perf_counter() - t0
            print(f"[w2_rebuild] scene ready: {prep}", flush=True)
            t0 = time.perf_counter()
            common = ["--root", str(cfg.project / "edits"), "--project", args.edit_id]
            net = (["--backend-url", args.backend_url, "--task-id", args.task_id] if args.backend_url else [])
            if args.mode == "scratch":
                sh([sys.executable, str(HERE / "stages" / "s4_train_3dgs.py"), *common, *net, "--trainer", "gsplat",
                    "--mode", "custom", "--iters", str(iters), "--holdout-every", str(every), "--gpu", args.gpu], env)
                latest = read_json(ecfg.gs_dir / "latest.json", {})
            else:
                src = read_json(cfg.gs_dir / "latest.json", {})
                ckpt = cfg.resolve(src.get("ckpt"))
                if not ckpt or not ckpt.exists():
                    raise SystemExit("finetune needs the trained checkpoint (gs/latest.json -> ckpt). Upload gs/ "
                                     "(worker --upload-checkpoint) or use --mode scratch")
                from common import gs_space
                from common.gs_finetune import finetune
                sp = gs_space.load_poses(cfg.p("week2", gs_space.FILE))
                if sp is None:
                    print("[w2_rebuild] WARNING: week2/splat_poses.json missing - run stages/w2_calibrate.py first; "
                          "if the checkpoint was trained in gsplat's normalized space the fine-tune will diverge",
                          flush=True)
                os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
                res = finetune(ckpt, E / "scene", ecfg.gs_dir / f"finetune_{iters}", steps=iters, holdout_every=every,
                               lr_scale=args.lr_scale, densify=not args.no_densify, poses=sp,
                               progress=lambda f: prog.report("training", f))
                latest = {**res, "mode": "finetune", "holdout_every": every}
                for k in ("ckpt", "ply"):
                    latest[k] = ecfg.rel(latest[k])
                latest["result_dir"] = ecfg.rel(ecfg.gs_dir / f"finetune_{iters}")
                write_json(ecfg.gs_dir / "latest.json", latest)
            t_train = time.perf_counter() - t0
            t0 = time.perf_counter()
            sh([sys.executable, str(HERE / "stages" / "s5_export_splat.py"), *common, *net], env)
            t_exp = time.perf_counter() - t0

            base = read_json(cfg.gs_dir / "latest.json", {}).get("heldout") or {}
            sm = read_json(ecfg.export_dir / "scene_meta.json", {})
            report = {"mode": args.mode, "iters": iters, "edited_frames": prep["edited_frames"],
                      "seconds": {"prepare_scene": round(t_prep, 1), "train": round(t_train, 1),
                                  "export": round(t_exp, 1), "total": round(t_prep + t_train + t_exp, 1)},
                      "num_gaussians": latest.get("num_gaussians") or sm.get("num_gaussians"),
                      "heldout_edited": latest.get("heldout"), "heldout_original": base or None,
                      "jpeg_reencode": {k: v for k, v in prep.items() if k.startswith("outside")},
                      "trainer": latest.get("trainer"), "lr_scale": args.lr_scale if args.mode == "finetune" else None}
            m.extra.update(report)
            arch = E / "rebuilds" / args.mode
            shutil.rmtree(arch, ignore_errors=True)
            shutil.copytree(ecfg.export_dir, arch / "export")
            dump_json(arch / "report.json", report)
            dump_json(E / "metrics" / f"rebuild_{args.mode}.json", report)
            dump_json(E / "metrics" / "s4_rebuild.json", {**report, "peak_vram_mb": None})   # B's stub-compatible name
            print(f"[w2_rebuild] {args.mode}: {report['seconds']} gaussians {report['num_gaussians']} "
                  f"held-out {report['heldout_edited']}", flush=True)
            prog.report("exporting", 1.0, force=True)
        except Exception as e:
            prog.fail("training", e)
            raise


if __name__ == "__main__":
    main()

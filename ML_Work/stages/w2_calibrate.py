#!/usr/bin/env python3
"""Week 2 / A - make the splat checkpoint and the camera poses agree, then re-render the splat depth.

    python stages/w2_calibrate.py --project-dir P

Run it ONCE per project, after unpacking the Week 1 files and BEFORE tracking / fine-tuning. Fixes the case where
gsplat trained in a normalized world space (simple_trainer default) but Week 2 rendered it with raw COLMAP poses
(fine-tune PSNR ~8, 3D lift finds nothing). Writes:
    week2/splat_poses.json   poses in the checkpoint's space (read by w2_common.load_cams and common/gs_finetune)
    week2/depth/*.npy        splat depth re-rendered with those poses (what w2_track lifts with)
    week2/inputs.json        gets "splat_space": {space, psnr...}
Prints the PSNR of raw vs normalized poses: the right one is ~25 dB (Week 1 held-out), the wrong one ~8 dB.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import gs_space  # noqa: E402
from common.colmap_io import read_model  # noqa: E402
from common.config import read_json, write_json  # noqa: E402
from common.w2_common import cfg_from_dir  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", type=Path, required=True)
    ap.add_argument("--gsplat-dir", default=os.environ.get("GSPLAT_DIR", "/kaggle/working/gsplat"))
    ap.add_argument("--depth-scale", type=float, default=0.5)
    ap.add_argument("--force-space", choices=["raw", "normalized"], help="skip the PSNR test")
    args = ap.parse_args(argv)
    cfg = cfg_from_dir(args.project_dir)
    latest = read_json(cfg.gs_dir / "latest.json", {})
    ckpt = cfg.resolve(latest.get("ckpt"))
    if not ckpt or not ckpt.exists():
        raise SystemExit("no checkpoint (gs/latest.json -> ckpt)")
    wk = cfg.p("week2")
    space, poses, psnrs = gs_space.calibrate(ckpt, cfg.scene_dir, Path(args.gsplat_dir), factor=cfg.train.data_factor)
    if args.force_space:
        space = args.force_space
    print(f"[calibrate] PSNR with raw COLMAP poses: {psnrs['raw']:.2f} dB | with gsplat-normalized poses: "
          f"{psnrs['normalized']} -> checkpoint space = {space}", flush=True)
    if space == "raw" and psnrs["raw"] < 15:
        print("[calibrate] WARNING: neither pose set reproduces the photos (PSNR < 15). Check data_factor / the "
              "scene dir / that the checkpoint belongs to this project.", flush=True)
    gs_space.save_poses(wk / gs_space.FILE, space, poses, psnrs)

    # re-render the splat depth with the matching poses
    from s6_prepare_week2 import render_splat_depth
    model = read_model(cfg.scene_dir / "sparse" / "0")
    info = render_splat_depth(cfg, ckpt, model, wk / "depth", args.depth_scale, poses=poses)
    # the viewer's "snap to training camera" must use poses in the splat's space too
    from common.gs_io import cameras_json
    cj = cfg.export_dir / "cameras.json"
    if cj.exists():
        if not (cfg.export_dir / "cameras_raw_colmap.json").exists():
            (cfg.export_dir / "cameras_raw_colmap.json").write_text(cj.read_text())
        write_json(cj, cameras_json(model, poses=poses))
        print("[calibrate] export/cameras.json rewritten in the splat's space (original: cameras_raw_colmap.json)", flush=True)
    inputs = read_json(wk / "inputs.json", {})
    inputs["splat_space"] = {"space": space, "psnr": {k: psnrs[k] for k in ("raw", "normalized")},
                             "poses_file": cfg.rel(wk / gs_space.FILE)}
    inputs["depth"] = {**(inputs.get("depth") or {}), "kind": "splat_rendered", "dir": cfg.rel(wk / "depth"), **info}
    write_json(wk / "inputs.json", inputs)
    print(f"[calibrate] splat depth re-rendered for {info['frames']} frames ({space} poses)", flush=True)


if __name__ == "__main__":
    main()

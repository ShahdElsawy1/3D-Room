#!/usr/bin/env python3
"""Stage 5 - export for the browser viewer.

    python stages/s5_export_splat.py --project demo

Reads gs/latest.json (stage 4), then writes export/:
    scene.ply          standard 3DGS PLY (from the trainer, or converted from a gsplat .pt checkpoint)
    scene.splat        compact 32 B/Gaussian format for web viewers (antimatter15 / three.js loaders)
    cameras.json       every training camera (position + c2w rotation + intrinsics) so the viewer can
                       "snap to a training camera" - the Week 2 click-on-reference-frame baseline
    scene_meta.json    #Gaussians, file sizes, bbox, orientation hint   (GET /projects/{id}/scene)
.ksplat: produce with mkkelvin/GaussianSplats3D's `node util/create-ksplat.js export/scene.ply export/scene.ksplat`
if your viewer prefers it (optional; .splat is enough for the viewers we recommend).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.colmap_io import read_model  # noqa: E402
from common.config import add_common_args, config_from_args, read_json, write_json  # noqa: E402
from common.gs_io import ckpt_to_ply, cameras_json, ply_to_splat  # noqa: E402
from common.metrics import StageMetrics  # noqa: E402
from common.progress import Progress  # noqa: E402

STAGE = "exporting"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--no-sort", action="store_true", help="skip importance sorting of the .splat")
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    prog = Progress(cfg.metrics_dir, args.backend_url, args.task_id)
    prog.start(STAGE)

    latest = read_json(cfg.gs_dir / "latest.json")
    if not latest:
        prog.fail(STAGE, "no trained model - run stage 4")
        raise SystemExit("gs/latest.json missing - run stage 4 first")

    with StageMetrics(cfg.metrics_dir, "s5_export_splat") as m:
        try:
            out = cfg.export_dir
            out.mkdir(parents=True, exist_ok=True)
            ply = out / "scene.ply"
            lply, lckpt = cfg.resolve(latest.get("ply")), cfg.resolve(latest.get("ckpt"))   # relative or legacy absolute
            if lply and lply.exists():
                shutil.copy2(lply, ply)
                src = "trainer ply"
            elif lckpt and lckpt.exists():
                n = ckpt_to_ply(lckpt, ply)
                src = f"converted from {lckpt.name} ({n} Gaussians)"
            else:
                raise RuntimeError("neither a PLY nor a checkpoint is available")
            prog.report(STAGE, 0.3)

            info = ply_to_splat(ply, out / "scene.splat", sort=not args.no_sort)
            prog.report(STAGE, 0.7)

            model = read_model(cfg.scene_dir / "sparse" / "0")
            from common import gs_space  # noqa: E402
            write_json(out / "cameras.json", cameras_json(model, poses=gs_space.load_poses(cfg.p("week2", gs_space.FILE))))

            meta = {**info, "source": src, "trainer": latest.get("trainer"), "iters": latest.get("iters"),
                    "ply_mb": round(ply.stat().st_size / 2**20, 2),
                    "splat_mb": round((out / "scene.splat").stat().st_size / 2**20, 2),
                    "coordinate_note": "COLMAP/OpenCV world: +y is DOWN. Set the viewer's camera up-vector to (0,-1,0) "
                                       "or flip y when loading, otherwise the room appears upside-down.",
                    "cameras": "cameras.json", "num_cameras": len(model.images),
                    "heldout": latest.get("heldout")}
            write_json(out / "scene_meta.json", meta)
            m.extra.update(num_gaussians=meta["num_gaussians"], ply_mb=meta["ply_mb"], splat_mb=meta["splat_mb"])
            print(f"[export] {meta['num_gaussians']:,} Gaussians; ply {meta['ply_mb']} MB, splat {meta['splat_mb']} MB")
            prog.report(STAGE, 1.0)
        except Exception as e:
            prog.fail(STAGE, e)
            raise


if __name__ == "__main__":
    main()

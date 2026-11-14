#!/usr/bin/env python3
"""Quick GPU self-test for the Week 2 models (run ONCE on Kaggle after `setup_kaggle.sh w2`, before real work).

    python tools/w2_selftest.py --project-dir $W1_ROOT/demo [--finetune-steps 20]

Checks: CUDA, SAM 2 image model (one click), SAM 2 video model loads, gsplat rasterization + the strategy API used by
the fine-tuner, fused-ssim, and (optional) a 20-step fine-tune on the real scene/checkpoint. Prints PASS/FAIL per
check with the error, exits non-zero if any failed - so you find API surprises in 2 minutes, not after an hour.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = []


def check(name):
    def deco(fn):
        def run(*a, **k):
            t0 = time.perf_counter()
            try:
                msg = fn(*a, **k)
                RESULTS.append((name, True, f"{msg or ''} ({time.perf_counter() - t0:.1f}s)"))
            except Exception as e:                                # noqa: BLE001
                traceback.print_exc()
                RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        return run
    return deco


@check("cuda")
def t_cuda():
    import torch
    assert torch.cuda.is_available(), "no CUDA device"
    return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, {torch.cuda.device_count()} GPU(s)"


@check("sam2 image model + one click")
def t_sam_image(project: Path, model: str, dtype: str):
    import numpy as np
    from common.masks import frame_names
    from common.sam_wrap import SamImage
    from common.w2_common import read_rgb
    name = frame_names(project)[0]
    rgb = read_rgb(project / "frames" / "images" / name)
    h, w = rgb.shape[:2]
    m, s = SamImage(model, "cuda", dtype).segment(rgb, [(w / 2, h / 2, 1)])
    assert m.shape == (h, w) and m.dtype == bool, (m.shape, m.dtype)
    assert not np.isnan(s), "NaN score - try --sam-dtype float32"
    return f"mask {int(m.sum())} px, score {s:.2f}"


@check("sam2 video model loads")
def t_sam_video(model: str, dtype: str):
    from common.sam_wrap import SamVideo
    SamVideo(model, "cuda", dtype)
    return "loaded"


@check("gsplat + strategy API + fused-ssim")
def t_gsplat():
    import torch
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy
    n = 2000
    means, quats = torch.randn(n, 3, device="cuda"), torch.randn(n, 4, device="cuda")
    scales, op = torch.full((n, 3), 0.05, device="cuda"), torch.full((n,), 0.5, device="cuda")
    colors = torch.rand(n, 16, 3, device="cuda")
    V = torch.eye(4, device="cuda")[None]
    V[0, 2, 3] = 6
    K = torch.tensor([[200., 0, 100], [0, 200., 100], [0, 0, 1]], device="cuda")[None]
    out, alpha, info = rasterization(means, quats, scales, op, colors, V, K, 200, 200, sh_degree=3, packed=False)
    st = DefaultStrategy(verbose=False, refine_start_iter=0, refine_stop_iter=10, reset_every=10 ** 9)
    for attr in ("check_sanity", "initialize_state", "step_pre_backward", "step_post_backward"):
        assert hasattr(st, attr), f"DefaultStrategy.{attr} missing"
    d, _, _ = rasterization(means, quats, scales, op, torch.zeros(n, 3, device="cuda"), V, K, 200, 200, render_mode="ED")
    try:
        from fused_ssim import fused_ssim
        fused_ssim(out.permute(0, 3, 1, 2), out.permute(0, 3, 1, 2), padding="valid")
        ssim = "fused_ssim ok"
    except Exception as e:                                        # noqa: BLE001
        ssim = f"fused_ssim UNAVAILABLE ({type(e).__name__}) -> L1-only fine-tune loss"
    return f"render {tuple(out.shape)}, depth {tuple(d.shape)}; {ssim}"


@check("fine-tune loop (few steps on the real scene)")
def t_finetune(project: Path, steps: int):
    from common.config import read_json
    from common.gs_finetune import finetune
    from common.config import Config
    cfg = Config(root=str(project.parent), project_id=project.name)
    ck = cfg.resolve((read_json(cfg.gs_dir / "latest.json", {}) or {}).get("ckpt"))
    assert ck and ck.exists(), "no checkpoint in gs/latest.json (train first, or upload gs/)"
    with tempfile.TemporaryDirectory() as td:
        r = finetune(ck, cfg.scene_dir, Path(td), steps=steps, holdout_every=8, log_every=max(steps // 2, 1))
    return f"{steps} steps, {r['train_seconds']}s, held-out PSNR {r['heldout']['psnr']:.2f}, peak {r['peak_mem_gb']} GB"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", type=Path, required=True)
    ap.add_argument("--sam-model", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--sam-dtype", default="float16")
    ap.add_argument("--finetune-steps", type=int, default=20, help="0 = skip")
    a = ap.parse_args(argv)
    t_cuda()
    t_sam_image(a.project_dir.resolve(), a.sam_model, a.sam_dtype)
    t_sam_video(a.sam_model, a.sam_dtype)
    t_gsplat()
    if a.finetune_steps:
        t_finetune(a.project_dir.resolve(), a.finetune_steps)
    print("\n== Week 2 self-test ==")
    for n, ok, msg in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}: {msg}")
    sys.exit(0 if all(ok for _, ok, _ in RESULTS) else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Week 2 - diffusion edit of the tracked object, frame by frame, consistent across views (brief section 4.3).

    # 1. several seeds on the frame where the object is large and clear; look at preview_reference.png and choose one
    python stages/w2_edit.py --project-dir P --edit-id E --phase reference --prompt "make the sofa dark green leather"
    # 2. every other frame: its own mask, the same prompt, the chosen reference edit as an extra condition image
    python stages/w2_edit.py --project-dir P --edit-id E --phase rest --prompt "..." --ref-seed 1
    # or both in one go (picks --ref-seed, default = the first seed)
    python stages/w2_edit.py --project-dir P --edit-id E --phase all --prompt "..."

Inputs : frames/images, edits/E/masks (w2_segment + w2_track; frames with present=true are edited)
Output : edits/E/edited/<stem>.png + edited/index.json   <- exactly what stages/w2_rebuild.py reads
             {"edited": [{"name": "000012.jpg", "file": "edited/000012.png", "role": "reference|conditioned", "seed": 1}],
              "reference": "000012.jpg", "prompt": "...", ...}
         edits/E/edit/reference/seed_<s>.png + index.json   the candidate reference edits
         edits/E/preview_reference.png, preview_edited.png  (original above edit; picked up by tools/make_handoff.py)
         edits/E/metrics/w2_edit.json (+ w2_edit_<tag>.json per quantization setting)   seconds/frame, peak VRAM
Every output frame has the size of frames/images/<name>; pixels outside the mask grown by --grow px are bit-identical
to the original (`outside_grown_mask_max_abs_diff` in the metrics proves it). Poses and masks are never touched.

Compute (brief section 3): frames keep their Week 1 size, padded to a multiple of 16 for the model and cropped back;
the stage runs on ONE GPU (--gpu, default 0 = "GPU 0 holds the diffusion model, text encoder and VAE") as its own
process, so SAM 2 and 3DGS (GPU 1) are untouched. --quant 4bit|8bit|none and --offload are the memory knobs; every run
records peak VRAM and seconds per frame so the quantization choice is documented (tools/w2_edit_table.py).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from common import diffusion_edit as D  # noqa: E402
from common import masks as M  # noqa: E402
from common.config import write_json  # noqa: E402
from common.metrics import StageMetrics  # noqa: E402
from common.w2_common import cfg_from_dir, edit_progress, read_rgb  # noqa: E402

STAGE = "editing"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", type=Path, required=True, help="<root>/<project_id> (worker passes this)")
    ap.add_argument("--edit-id", required=True)
    ap.add_argument("--backend-url", default="")
    ap.add_argument("--task-id", default="")
    ap.add_argument("--phase", default="all", choices=["reference", "rest", "all"])
    ap.add_argument("--prompt", required=True, help='e.g. "make the sofa dark green leather"')
    ap.add_argument("--ref-frame", default="", help="reference frame (default: largest, least cut-off mask)")
    ap.add_argument("--n-seeds", type=int, default=4, help="[reference] candidates to generate")
    ap.add_argument("--seed", type=int, default=0, help="[reference] first seed; candidates use seed, seed+1, ...")
    ap.add_argument("--ref-seed", type=int, default=-1, help="[rest] which candidate is the reference edit")
    ap.add_argument("--ref-file", type=Path, default=None,
                    help="[rest] use this image as the reference edit instead (e.g. one made by hand or by Person B); "
                         "needs --ref-frame")
    ap.add_argument("--frame-seed", type=int, default=-1, help="[rest] seed of the other frames (default: the reference seed)")
    ap.add_argument("--stride", type=int, default=1, help="[rest] edit every Nth frame only (keyframes); others stay original")
    ap.add_argument("--limit", type=int, default=0, help="[rest] stop after N frames (smoke test)")
    ap.add_argument("--resume", action="store_true", help="[rest] keep frames already written by an earlier run")
    ap.add_argument("--model", default="Qwen/Qwen-Image-2.1")
    ap.add_argument("--quant", default="4bit", choices=["4bit", "8bit", "none"], help="bitsandbytes on transformer + text encoder")
    ap.add_argument("--offload", default="none", choices=["none", "model", "sequential"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--cfg", type=float, default=1.0, help=">1 turns classifier-free guidance on (doubles the time per step)")
    ap.add_argument("--gen-scale", type=float, default=1.0, help="generate at this multiple of the padded frame size")
    ap.add_argument("--fit", default="pad", choices=["pad", "resize"], help="how to reach a multiple of 16")
    ap.add_argument("--hint", default="outline", choices=["outline", "none"], help="show the model the mask as a red outline")
    ap.add_argument("--grow", type=int, default=2, help="px the mask is grown by before compositing (0 = exact mask)")
    ap.add_argument("--feather", type=int, default=2, help="soft edge px inside the grown region (0 = hard)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--tag", default="", help="name of this setting in the VRAM/timing table (default: quant_dtype_offload_steps)")
    return ap


# ----------------------------------------------------------------------------- core (the model is injected, so this runs on CPU)
def load_targets(cfg, edit_id: str) -> dict[str, np.ndarray]:
    E = M.edit_dir(cfg.project, edit_id)
    idx = M.read_index(E / "masks" / "index.json")
    out = {}
    for f in idx["frames"]:
        if f["present"]:
            m = M.load_mask(E / "masks" / (Path(f["name"]).stem + ".png"))
            if m is not None and m.any():
                out[f["name"]] = m
    if not out:
        raise SystemExit("no masks yet - run w2_segment (and w2_track) first")
    return out


def edit_one(editor, rgb: np.ndarray, mask: np.ndarray, prompt: str, seed: int, *, reference: np.ndarray | None = None,
             hint: bool = True, grow: int = 2, feather: int = 2, gen_scale: float = 1.0, fit: str = "pad") -> np.ndarray:
    """One frame: (outline hint) -> pad -> model -> crop -> composite. Returns RGB uint8 of the frame's size."""
    src = D.outline_hint(rgb, mask) if hint else rgb
    fitted, info = [], None
    for im in [src] + ([reference] if reference is not None else []):
        p, info = D.fit_to_multiple(im, mode=fit)
        fitted.append(p)
    out = editor.edit(fitted, D.build_prompt(prompt, reference is not None, hint), seed, D.gen_size(info["size_hw"], gen_scale))
    return D.composite(rgb, D.unfit(out, info), mask, grow, feather)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise IOError(f"could not write {path}")


def opts_of(a) -> dict:
    return dict(hint=a.hint == "outline", grow=a.grow, feather=a.feather, gen_scale=a.gen_scale, fit=a.fit)


def run_reference(cfg, a, editor, prog, targets, stats) -> dict:
    E = M.edit_dir(cfg.project, a.edit_id)
    ref = a.ref_frame or D.pick_reference(targets)
    if ref not in targets:
        raise SystemExit(f"--ref-frame {ref!r} has no mask (present frames: {len(targets)})")
    rgb = read_rgb(cfg.images_dir / ref)
    rd = E / "edit" / "reference"
    shutil.rmtree(rd, ignore_errors=True)
    cands, pairs = [], []
    for k in range(a.n_seeds):
        seed = a.seed + k
        t0 = time.perf_counter()
        out = edit_one(editor, rgb, targets[ref], a.prompt, seed, **opts_of(a))
        stats["reference_seconds"].append(round(time.perf_counter() - t0, 2))
        stats["outside"] = max(stats["outside"], D.outside_diff(rgb, out, targets[ref], a.grow)["max_abs_diff"])
        save_rgb(rd / f"seed_{seed}.png", out)
        cands.append({"seed": seed, "file": f"edit/reference/seed_{seed}.png"})
        pairs.append((rgb, out))
        prog.report(STAGE, 0.1 + 0.9 * (k + 1) / a.n_seeds)
    idx = {"ref_frame": ref, "prompt": a.prompt, "candidates": cands, "chosen_seed": None}
    write_json(rd / "index.json", idx)
    sheet = D.contact_sheet(pairs, cols=min(len(pairs), 4), thumb_w=360)
    cv2.imwrite(str(E / "preview_reference.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"[w2_edit] reference frame {ref}: seeds {[c['seed'] for c in cands]} -> {E / 'preview_reference.png'}", flush=True)
    return idx


def run_rest(cfg, a, editor, prog, targets, stats, ref_idx: dict | None = None) -> dict:
    E = M.edit_dir(cfg.project, a.edit_id)
    names = M.frame_names(cfg.project)
    if a.ref_file:
        if not a.ref_frame:
            raise SystemExit("--ref-file needs --ref-frame (the frame the image is the edit of)")
        ref, ref_seed = a.ref_frame, a.ref_seed
    else:
        ref_idx = ref_idx or json.loads((E / "edit" / "reference" / "index.json").read_text())
        ref = ref_idx["ref_frame"]
        seeds = [c["seed"] for c in ref_idx["candidates"]]
        ref_seed = a.ref_seed if a.ref_seed >= 0 else (seeds[0] if len(seeds) == 1 else -1)
        if ref_seed not in seeds:
            raise SystemExit(f"choose the reference edit: --ref-seed one of {seeds} (see preview_reference.png)")
    if ref not in targets:
        raise SystemExit(f"reference frame {ref} has no mask")
    orig_ref = read_rgb(cfg.images_dir / ref)
    ref_path = a.ref_file if a.ref_file else E / "edit" / "reference" / f"seed_{ref_seed}.png"
    ref_bgr = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    if ref_bgr is None:
        raise SystemExit(f"cannot read {ref_path}")
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    if ref_rgb.shape != orig_ref.shape:
        raise SystemExit(f"reference edit {ref_rgb.shape[1::-1]} != frame {orig_ref.shape[1::-1]}")
    if a.ref_file:                                           # an outside image: enforce "nothing changes outside the mask"
        ref_rgb = D.composite(orig_ref, ref_rgb, targets[ref], a.grow, a.feather)
    frame_seed = a.frame_seed if a.frame_seed >= 0 else max(ref_seed, 0)

    todo = [n for n in sorted(targets) if n != ref][::max(a.stride, 1)]
    if a.limit:
        todo = todo[:a.limit]
    entries = {}
    ip = E / "edited" / "index.json"
    if a.resume and ip.exists():
        for e in json.loads(ip.read_text()).get("edited", []):
            if (E / e["file"]).exists():
                entries[e["name"]] = e
    else:
        shutil.rmtree(E / "edited", ignore_errors=True)
    meta = {"reference": ref, "prompt": a.prompt, "model": a.model, "quant": a.quant, "dtype": a.dtype,
            "steps": a.steps, "grow": a.grow, "feather": a.feather, "hint": a.hint, "ref_seed": ref_seed,
            "frame_seed": frame_seed, "stride": a.stride}

    def flush():
        ordered = [entries[n] for n in names if n in entries]
        write_json(ip, {"edited": ordered, **meta})

    save_rgb(E / "edited" / (Path(ref).stem + ".png"), ref_rgb)
    entries[ref] = {"name": ref, "file": f"edited/{Path(ref).stem}.png", "role": "reference", "seed": ref_seed}
    flush()
    pairs = {ref: (orig_ref, ref_rgb)}
    for k, n in enumerate(todo):
        if n in entries and n != ref:
            prog.report(STAGE, 0.05 + 0.95 * (k + 1) / max(len(todo), 1))
            continue
        rgb = read_rgb(cfg.images_dir / n)
        t0 = time.perf_counter()
        out = edit_one(editor, rgb, targets[n], a.prompt, frame_seed, reference=ref_rgb, **opts_of(a))
        stats["frame_seconds"].append(round(time.perf_counter() - t0, 2))
        stats["outside"] = max(stats["outside"], D.outside_diff(rgb, out, targets[n], a.grow)["max_abs_diff"])
        save_rgb(E / "edited" / (Path(n).stem + ".png"), out)
        entries[n] = {"name": n, "file": f"edited/{Path(n).stem}.png", "role": "conditioned", "seed": frame_seed}
        flush()                                              # a crash or a session timeout leaves a valid partial index
        pairs[n] = (rgb, out)
        prog.report(STAGE, 0.05 + 0.95 * (k + 1) / max(len(todo), 1))
        print(f"[w2_edit] {k + 1}/{len(todo)} {n} {stats['frame_seconds'][-1]}s", flush=True)
    keep = sorted(pairs)[::max(len(pairs) // 12, 1)][:12]
    sheet = D.contact_sheet([pairs[n] for n in keep], cols=3, thumb_w=320)
    cv2.imwrite(str(E / "preview_edited.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    probs = D.validate_edited(cfg.project, a.edit_id)
    if probs:
        print("[w2_edit] CONTRACT PROBLEMS:", probs[:5], flush=True)
    stats["frames_edited"] = len(entries)
    return json.loads(ip.read_text())


def main(argv=None, editor_factory=None):
    a = build_parser().parse_args(argv)
    if editor_factory is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu               # before torch is imported: the diffusion model gets this GPU
    cfg = cfg_from_dir(a.project_dir)
    prog = edit_progress(cfg, a.edit_id, a)
    prog.report(STAGE, 0.0, force=True)
    E = M.edit_dir(cfg.project, a.edit_id)
    tag = a.tag or f"{a.quant}_{a.dtype}_{a.offload}_s{a.steps}"
    stats = {"reference_seconds": [], "frame_seconds": [], "outside": 0}
    with StageMetrics(E / "metrics", "w2_edit") as m:
        try:
            targets = load_targets(cfg, a.edit_id)
            t0 = time.perf_counter()
            editor = (editor_factory or (lambda: D.QwenEditor(a.model, a.dtype, a.quant, a.offload, a.steps, a.cfg)))()
            load_s = time.perf_counter() - t0
            prog.report(STAGE, 0.1, force=True)
            ref_idx = None
            if a.phase in ("reference", "all"):
                ref_idx = run_reference(cfg, a, editor, prog, targets, stats)
                if a.phase == "all" and a.ref_seed < 0:
                    a.ref_seed = ref_idx["candidates"][0]["seed"]
            if a.phase in ("rest", "all"):
                run_rest(cfg, a, editor, prog, targets, stats, ref_idx)
            fs, rs = stats["frame_seconds"], stats["reference_seconds"]
            m.extra.update(
                tag=tag, phase=a.phase, model=a.model, quant=a.quant, dtype=a.dtype, offload=a.offload, steps=a.steps,
                cfg=a.cfg, gen_scale=a.gen_scale, fit=a.fit, hint=a.hint, grow=a.grow, feather=a.feather,
                model_load_seconds=round(load_s, 2), frames_edited=stats.get("frames_edited"),
                seconds_per_frame_mean=round(float(np.mean(fs or rs)), 2) if (fs or rs) else None,
                seconds_per_frame_median=round(float(np.median(fs or rs)), 2) if (fs or rs) else None,
                reference_seconds=rs, load_peak_vram_mb=getattr(editor, "load_peak_mb", None),
                generation_peak_vram_mb=editor.peak_mb() if hasattr(editor, "peak_mb") else None,
                outside_grown_mask_max_abs_diff=stats["outside"])
            prog.report(STAGE, 1.0, force=True)
        except (Exception, SystemExit) as e:             # SystemExit = the clear "choose a seed / no masks" messages
            prog.fail(STAGE, e)
            raise
    shutil.copy2(E / "metrics" / "w2_edit.json", E / "metrics" / f"w2_edit_{tag}.json")


if __name__ == "__main__":
    main()

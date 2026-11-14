#!/usr/bin/env python3
"""VRAM and timing table of the diffusion model, one row per quantization setting (Week 2 deliverable 4).

    python tools/w2_edit_table.py --project-dir P --edit-id E [--out edits/E/metrics/diffusion_table.md]

Reads edits/E/metrics/w2_edit_<tag>.json (written by every run of stages/w2_edit.py; run the stage once per setting,
e.g. --quant 4bit, --quant 8bit, --quant none --offload model, to fill the table) and prints/writes the Markdown table.
Nothing is typed by hand: peak VRAM is nvidia-smi's whole-GPU delta and torch's allocator peak, seconds are per frame.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.w2_common import cfg_from_dir  # noqa: E402


def _f(v, nd=1, unit=""):
    return "n/a" if v is None else f"{v:.{nd}f}{unit}"


def table(E: Path) -> str | None:
    runs = []
    for p in sorted((E / "metrics").glob("w2_edit_*.json")):
        r = json.loads(p.read_text())
        if r.get("ok", True):
            runs.append(r)
    if not runs:
        return None
    rows = []
    for r in runs:
        rows.append(f"| {r.get('tag')} | {r.get('model')} | {r.get('quant')} | {r.get('dtype')} | {r.get('offload')} | "
                    f"{r.get('steps')} | {_f(r.get('model_load_seconds'), 0, ' s')} | "
                    f"{_f(r.get('seconds_per_frame_mean'), 1, ' s')} | {_f(r.get('seconds_per_frame_median'), 1, ' s')} | "
                    f"{_f(r.get('peak_vram_mb'), 0, ' MiB')} | {_f(r.get('generation_peak_vram_mb'), 0, ' MiB')} | "
                    f"{_f(r.get('load_peak_vram_mb'), 0, ' MiB')} | {r.get('frames_edited')} | "
                    f"{r.get('outside_grown_mask_max_abs_diff')} |")
    head = ("| setting | model | quantization | dtype | offload | steps | load time | s/frame (mean) | s/frame (median) | "
            "peak VRAM (nvidia-smi, GPU delta) | peak VRAM generating (torch) | VRAM after load (torch) | frames | "
            "max pixel change outside the mask |\n" + "|---" * 14 + "|")
    return head + "\n" + "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", type=Path, required=True)
    ap.add_argument("--edit-id", required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    E = cfg_from_dir(a.project_dir).project / "edits" / a.edit_id
    t = table(E)
    if t is None:
        raise SystemExit(f"no {E / 'metrics' / 'w2_edit_<tag>.json'} yet - run stages/w2_edit.py first")
    out = a.out or E / "metrics" / "diffusion_table.md"
    out.write_text(t + "\n")
    print(t)
    print(f"\n[w2_edit_table] wrote {out}")


if __name__ == "__main__":
    main()

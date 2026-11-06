#!/usr/bin/env python3
"""Report tables (M9) from real measurements: Week 1 deliverable 5 (VRAM and runtime per stage) and Week 2
deliverable 4 (VRAM and timing of the diffusion model, with the quantization choice), plus upload throughput and
rebuild times. Every number comes from a file or the backend; anything missing is printed as "not measured".

  # everything the backend knows about a project (tasks' results + stored artifact sizes)
  python tools/make_report.py --backend-url http://localhost:8000 --email me@x.com --password ... --project <id>
  # or straight from files (Person A's metrics/, bench_t4.json, edit_report.json, a saved GET /tasks/{id})
  python tools/make_report.py --metrics-dir export/metrics --bench bench_t4.json --edit-report edit_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NA = "not measured"
STAGE_NAMES = {"s1_extract_frames": "1 extract frames", "s2a_poses_vggt": "2a poses: VGGT",
               "s2b_poses_glomap": "2b poses: GLOMAP", "s3_compare_poses": "3 compare poses",
               "s4_train_3dgs": "4 train 3DGS", "s5_export_splat": "5 export splat", "s4_rebuild": "rebuild"}
STAGE_EXTRAS = ("frames_kept", "frames", "registered_frames", "points3D", "iters", "num_gaussians", "ply_mb",
                "splat_mb", "chunks", "model_load_s")


def fmt(v, nd=1):
    if v is None:
        return NA
    if v == "":
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def table(head: list[str], rows: list[list]) -> str:
    if not rows:
        return f"_{NA}_\n"
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------------- week 1
def stage_table(stages: dict) -> str:
    rows = []
    for key in sorted(stages):
        m = stages[key]
        extras = ", ".join(f"{k}={fmt(m[k], 2)}" for k in STAGE_EXTRAS if k in m)
        rows.append([STAGE_NAMES.get(key, key), m.get("seconds"), m.get("peak_vram_mb"), extras or ""])
    total = sum(float(m.get("seconds") or 0) for m in stages.values())
    if rows:
        rows.append(["**total**", round(total, 1), max(float(m.get("peak_vram_mb") or 0) for m in stages.values()), ""])
    return table(["stage", "seconds", "peak VRAM (MB)", "details"], rows)


def upload_table(uploads: dict) -> str:
    rows = [[k, u.get("mb"), u.get("seconds"), u.get("mb_per_s")] for k, u in uploads.items()]
    if rows:
        mb, s = sum(u.get("mb") or 0 for u in uploads.values()), sum(u.get("seconds") or 0 for u in uploads.values())
        rows.append(["**total**", round(mb, 2), round(s, 1), round(mb / s, 2) if s else None])
    return table(["artifact", "MB", "seconds", "MB/s"], rows)


def artifact_table(artifacts: list[dict]) -> str:
    rows = [[a["kind"], a.get("version"), round((a.get("size_bytes") or 0) / 2**20, 2)]
            for a in sorted(artifacts, key=lambda a: (a.get("version") != "original", a.get("version"), a["kind"]))]
    return table(["artifact", "version", "MB on the backend"], rows)


# ---------------------------------------------------------------------------------------------------- week 2
def diffusion_rows(label: str, section: dict) -> list:
    s = section.get("summary") or {}
    vram = s.get("peak_vram_mb") or {}
    return [label, section.get("model"), f"{section.get('option', '?')}: {section.get('precision', '?')}",
            section.get("placement"), section.get("long_side"), section.get("load_seconds"), s.get("frames"),
            s.get("seconds_per_frame"), vram.get("cuda:0"), vram.get("cuda:1"), s.get("degenerate_frames"),
            s.get("mean_raw_leak_psnr"), s.get("mean_delta_e_vs_reference")]


DIFF_HEAD = ["run", "model", "option: dtype", "placement", "crop px", "load s", "frames", "s / frame",
             "peak VRAM GPU0 (MB)", "GPU1 (MB)", "NaN/flat frames", "raw leak PSNR (dB)", "ΔE vs reference"]


def bench_table(b: dict) -> str:
    rows = [[r.get("option"), r.get("path"), r.get("linear_ms"), r.get("linear_tflops"), r.get("attention_ms"),
             r.get("attention_tflops"), r.get("nan_or_inf"), r.get("error") or ""] for r in b.get("rows", [])]
    head = f"GPU: {b.get('gpu', '?')} (compute {b.get('capability', '?')}), torch {b.get('torch', '?')}, " \
           f"overflow probe bf16={b.get('overflow_probe_bf16', '?')} fp16={b.get('overflow_probe_fp16', '?')}\n\n"
    return head + table(["option", "path", "linear ms", "linear TFLOPS", "attention ms", "attention TFLOPS",
                         "NaN/inf", "error"], rows)


# ---------------------------------------------------------------------------------------------------- inputs
def from_backend(a) -> dict:
    import requests
    base = a.backend_url.rstrip("/")
    tok = requests.post(f"{base}/auth/login", data={"username": a.email, "password": a.password}, timeout=30)
    tok.raise_for_status()
    h = {"Authorization": f"Bearer {tok.json()['access_token']}"}
    tasks = requests.get(f"{base}/projects/{a.project}/tasks", headers=h, timeout=30).json()
    arts = requests.get(f"{base}/projects/{a.project}/artifacts", headers=h, timeout=30).json()
    return {"tasks": tasks, "artifacts": arts}


def build(data: dict) -> str:
    out = ["# Measurements\n"]
    tasks = [t for t in data.get("tasks", []) if t.get("status") == "done"]
    recon = [t for t in tasks if t["type"] == "reconstruct"]
    stages = data.get("stages") or (recon[0]["result"].get("stages") if recon and recon[0].get("result") else {})
    out.append("## Week 1: runtime and VRAM per stage\n")
    out.append(stage_table(stages or {}))
    if recon:
        r = recon[0]["result"] or {}
        out.append(f"\nPipeline {fmt(r.get('pipeline_seconds'))} s, upload {fmt(r.get('upload_seconds'))} s "
                   f"(worker `{r.get('host', '?')}`).\n")
        out.append("\n## Week 1: upload Kaggle → backend (through the tunnel)\n")
        out.append(upload_table(r.get("uploads") or {}))
    if data.get("artifacts"):
        out.append("\n## Stored artifact sizes\n")
        out.append(artifact_table(data["artifacts"]))

    out.append("\n## Week 2: diffusion editing (per run)\n")
    rows = []
    for t in tasks:
        if t["type"] in ("edit_preview", "edit_apply") and (t.get("result") or {}).get("edit"):
            rows.append(diffusion_rows(f"{t['type']} {t['id'][:6]}", t["result"]["edit"]))
    for path, rep in data.get("edit_reports", []):
        for sec in ("preview", "frames"):
            if sec in rep:
                rows.append(diffusion_rows(f"{Path(path).parent.name}/{sec}", rep[sec]))
    out.append(table(DIFF_HEAD, rows))
    out.append("\n## Week 2: T4 precision micro-benchmark\n")
    out.append(bench_table(data["bench"]) if data.get("bench") else f"_{NA}_\n")

    reb = [t for t in tasks if t["type"] == "rebuild"]
    out.append("\n## Week 2: rebuild time (poses reused)\n")
    out.append(table(["task", "mode", "stage seconds", "total seconds"],
                     [[t["id"][:6], t["params"].get("mode"), (t.get("result") or {}).get("stage_seconds"),
                       (t.get("result") or {}).get("total_seconds")] for t in reb]))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend-url")
    ap.add_argument("--email")
    ap.add_argument("--password")
    ap.add_argument("--project")
    ap.add_argument("--metrics-dir", type=Path, help="Person A's metrics/ folder (s*_*.json)")
    ap.add_argument("--bench", type=Path, help="bench_t4.json")
    ap.add_argument("--edit-report", type=Path, action="append", default=[], help="edit_report.json (repeatable)")
    ap.add_argument("--task-json", type=Path, action="append", default=[], help="saved GET /tasks/{id} (repeatable)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    data: dict = {"tasks": [], "artifacts": [], "edit_reports": []}
    if a.backend_url:
        if not (a.email and a.password and a.project):
            ap.error("--backend-url needs --email, --password and --project")
        data.update(from_backend(a))
    for f in a.task_json:
        data["tasks"].append(json.loads(f.read_text()))
    if a.metrics_dir:
        data["stages"] = {}
        for f in sorted(a.metrics_dir.glob("s*_*.json")):
            d = json.loads(f.read_text())
            data["stages"][f.stem] = {k: v for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if a.bench:
        data["bench"] = json.loads(a.bench.read_text())
    for f in a.edit_report:
        data["edit_reports"].append((str(f), json.loads(f.read_text())))
    md = build(data)
    if a.out:
        a.out.write_text(md, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

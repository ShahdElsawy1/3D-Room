#!/usr/bin/env python3
"""Regenerates notebooks/kaggle_pipeline.ipynb (so the notebook stays a thin wrapper over the scripts)."""
import json
from pathlib import Path


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}


cells = [
md("""
# Week 1 - Video -> Browsable 3D Room (Person A pipeline)
**Notebook settings (right sidebar):** Accelerator = *GPU T4 x2* - Internet = *On* - Persistence = *Files only*.

Each stage below is a **separate process** (`!python ...`) so VRAM is released after every stage (PDF 4.3).
Run the cells top to bottom the first time. Re-runs can skip setup.
"""),
code("""
# 0. Get the code. EITHER clone your repo ...
REPO = "https://github.com/<your-team>/<your-repo>.git"      # <- edit
# !git clone $REPO /kaggle/working/repo
# ... OR attach the repo as a Kaggle Dataset and copy it:
# !cp -r /kaggle/input/<your-code-dataset>/week1_pipeline /kaggle/working/repo
%cd /kaggle/working/repo/week1_pipeline
"""),
code("""
# 1. One-time setup: ffmpeg, VGGT, COLMAP(+GLOMAP), gsplat.  ~10-20 min the first time (gsplat compiles CUDA on first use).
!bash scripts/setup_kaggle.sh all
"""),
code("""
# 2. Environment for this session
import os, re, shutil
for line in open("/kaggle/working/env.sh"):
    m = re.match(r"export (\\w+)=(.*)", line.strip())
    if m: os.environ[m.group(1)] = os.path.expandvars(m.group(2))
# safety: the conda env's bare python must never shadow the system python
os.environ["PATH"] = ":".join(x for x in os.environ["PATH"].split(":") if x and "/kaggle/working/cm/bin" not in x and "$PATH" not in x)
print("python :", shutil.which("python"), "| colmap:", shutil.which("colmap"), "| glomap:", shutil.which("glomap"))
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"          # T4
os.environ["W1_ROOT"] = "/kaggle/working/workspace"   # everything is written here (persisted in Output)
os.environ["W1_PROJECT"] = "demo"
VIDEO = "/kaggle/input/<your-video-dataset>/room.mp4"    # <- edit: 2-5 min, one room, slow motion
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
md("## Stage 1 - frames (ffmpeg, 1-4 fps, long side <= 960, blur filter)"),
code("""
!python stages/s1_extract_frames.py --video $VIDEO --fps auto --long-side 960
"""),
md("## Stage 2 - poses. Run BOTH for the required VGGT-vs-GLOMAP experiment (same frames)"),
code("""
# VGGT: fp16 on T4, chunks of 32 frames with 8 shared frames; also measure VRAM vs frames-per-chunk
!python stages/s2a_poses_vggt.py --chunk-size 32 --overlap 8 --memory-sweep 8,16,32,48
"""),
code("""
# GLOMAP: features -> sequential matching -> global mapper.  (skip this cell if no global mapper installed -> use --method vggt)
!python stages/s2b_poses_glomap.py --overlap 12
"""),
code("""
# Comparison report + pick the method used for the project (writes scene/)
!python stages/s3_compare_poses.py --choose auto
from IPython.display import Markdown, Image, display
display(Markdown(open(os.environ["W1_ROOT"] + "/demo/metrics/pose_comparison.md").read()))
"""),
md("## Stage 4 - 3D Gaussian Splatting (quick = 7k iters to confirm, then full = 30k)"),
code("""
!python stages/s4_train_3dgs.py --trainer gsplat --mode quick
"""),
code("""
# quality run (about 30k iters; check the quick run looked sane first!)
!python stages/s4_train_3dgs.py --trainer gsplat --mode full
"""),
md("## Stage 5 - export for the web viewer, then the generated deliverables"),
code("""
!python stages/s5_export_splat.py
!python tools/make_reports.py
!cat docs/generated/VRAM_RUNTIME_TABLE.md
"""),
code("""
# 6. Persist what matters from /kaggle/working before the session ends (PDF 4.3)
!cd $W1_ROOT/demo && zip -qr /kaggle/working/demo_export.zip export manifest.json metrics scene frames/frames_index.json
!ls -lh /kaggle/working/demo_export.zip
# frames + poses + depth (Week 2 inputs) are in $W1_ROOT/demo/{frames,poses,scene}; zip them too if the backend needs them:
# !cd $W1_ROOT/demo && zip -qr /kaggle/working/demo_week2_inputs.zip frames poses scene
"""),
md("""
### One-shot alternative
```
!python run_pipeline.py --video $VIDEO --project demo --method both --train-mode quick
```
With the backend: add `--backend-url https://<your>.trycloudflare.com --task-id <id>` and the real stage progress is POSTed to `/tasks/{id}/progress`.
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_pipeline.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)

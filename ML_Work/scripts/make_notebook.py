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
# 1. One-time setup: ffmpeg, VGGT, pycolmap 4 (GLOMAP built in, no conda), gsplat.  ~5-15 min the first time.
#    (the old route with the conda colmap/glomap binaries is still there: `setup_kaggle.sh all`, then --backend cli)
!bash scripts/setup_kaggle.sh fast
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
# GLOMAP via pycolmap 4 (auto-detected through W1_PYCOLMAP_DIR): GPU SIFT -> hybrid matching -> global mapping
# -> checks (falls back to incremental mapping if the result looks unreliable) -> prune bad frames -> undistort
!python stages/s2b_poses_glomap.py
# old route instead:  !python stages/s2b_poses_glomap.py --backend cli --overlap 12
"""),
code("""
# Comparison report + pick the method used for the project (writes scene/)
!python stages/s3_compare_poses.py --choose auto
from IPython.display import Markdown, Image, display
display(Markdown(open(os.environ["W1_ROOT"] + "/demo/metrics/pose_comparison.md").read()))
"""),
code("""
# OPTIONAL but recommended once per scene: which pose set renders held-out frames better? (trains twice at 7k iters, ~2 x 15 min;
# your current choice is restored afterwards; table -> metrics/pose_method_psnr.md). Skip when poses already agree (see the Sim(3) line above).
# !python tools/compare_pose_methods.py --iters 7000
"""),
md("## Stage 4 - 3D Gaussian Splatting (quick = 7k iters to confirm, then full = 30k). The 30k run is only worth it for the final output: on the 61-frame test it scored the same PSNR as 7k"),
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
# Week 2 inputs: week2/inputs.json (+ sparse depth, and per-frame depth rendered from the splat when the GPU is free)
!python stages/s6_prepare_week2.py
!python tools/make_reports.py
!cat docs/generated/VRAM_RUNTIME_TABLE.md
"""),
code("""
# 6. Persist what matters from /kaggle/working before the session ends (PDF 4.3)
!cd $W1_ROOT/demo && zip -qr /kaggle/working/demo_export.zip export manifest.json metrics scene frames/frames_index.json
!ls -lh /kaggle/working/demo_export.zip
# Week 2 inputs: everything listed in week2/inputs.json -> "required_files" (frames, scene, week2/, checkpoint in gs/).
import json, subprocess
req = json.load(open(os.environ["W1_ROOT"] + "/demo/week2/inputs.json"))["required_files"]
subprocess.run(["zip", "-qr", "/kaggle/working/demo_week2_inputs.zip", *req], cwd=os.environ["W1_ROOT"] + "/demo", check=True)
!ls -lh /kaggle/working/demo_week2_inputs.zip
"""),
md("""
### One-shot alternative
```
!python run_pipeline.py --video $VIDEO --project demo --method both --train-mode quick   # add --glomap-backend cli to use the colmap/glomap binaries
```
With the backend: add `--backend-url https://<your>.trycloudflare.com --task-id <id>` and the real stage progress is POSTed to `/tasks/{id}/progress`.
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_pipeline.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)

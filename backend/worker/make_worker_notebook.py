#!/usr/bin/env python3
"""Generates backend/worker/kaggle_worker.ipynb (Person B's worker notebook; Person A's notebooks are untouched).

    python backend/worker/make_worker_notebook.py
"""
import json
from pathlib import Path


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": s.strip("\n").splitlines(True)}


cells = [
md("""
# 3D-Room - Kaggle GPU worker (Person B)
This notebook turns a Kaggle GPU session into the **pull-based worker** of the backend (spec 4.3): it polls the
backend for "3D reconstruction" tasks, runs Person A's pipeline (`ML_Work/run_pipeline.py`, unmodified), reports
real stage progress, and uploads every artifact back to the backend before the session ends.

**Before running**
1. On your machine: backend running (`uvicorn app.main:app --port 8000`) **and** a Cloudflare tunnel
   (`cloudflared tunnel --url http://localhost:8000`) - see `backend/README.md` section 5. Copy the
   `https://....trycloudflare.com` URL.
2. Notebook settings (right sidebar): **Accelerator = GPU T4 x2**, **Internet = On**.
3. *Add-ons -> Secrets*: add **`W1_WORKER_TOKEN`** = contents of `backend/data/.worker_token` on your machine,
   and attach it to this notebook. Never paste the token into a cell.
"""),
code("""
# 1. Code: the repo (branch with backend/ + ML_Work/). For a private repo use a Kaggle Dataset instead (commented).
REPO_URL = "https://github.com/ShahdElsawy1/3D-Room.git"
BRANCH = "backend"
!rm -rf /kaggle/working/repo && git clone -q --depth 1 -b $BRANCH $REPO_URL /kaggle/working/repo
# !cp -r /kaggle/input/<your-code-dataset>/3D-Room /kaggle/working/repo
ML_DIR = "/kaggle/working/repo/ML_Work"
WORKER = "/kaggle/working/repo/backend/worker/kaggle_worker.py"
!ls $ML_DIR $WORKER
"""),
code("""
# 2. Person A's one-time GPU setup (VGGT, COLMAP/GLOMAP, gsplat). ~10-20 min on a fresh session.
!cd $ML_DIR && bash scripts/setup_kaggle.sh all
"""),
code("""
# 3. Environment for this session (same as Person A's notebook)
import os, re, shutil
for line in open("/kaggle/working/env.sh"):
    m = re.match(r"export (\\w+)=(.*)", line.strip())
    if m: os.environ[m.group(1)] = os.path.expandvars(m.group(2))
os.environ["PATH"] = ":".join(x for x in os.environ["PATH"].split(":") if x and "/kaggle/working/cm/bin" not in x and "$PATH" not in x)
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
print("python:", shutil.which("python"), "| colmap:", shutil.which("colmap"), "| glomap:", shutil.which("glomap"))
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
code("""
# 4. Environment fixes found during Person A's Kaggle run (third-party packages only; ML_Work is not modified).
#    - pycolmap's SceneManager breaks on numpy 2 (np.uint64(-1)) -> gsplat's simple_trainer cannot start
#    - gsplat/examples/datasets lacks __init__.py on some checkouts
import glob
for f in glob.glob("/usr/local/lib/python3*/dist-packages/pycolmap/scene_manager.py"):
    s = open(f).read()
    if "np.uint64(-1)" in s:
        open(f, "w").write(s.replace("np.uint64(-1)", "np.uint64(np.iinfo(np.uint64).max)")); print("patched", f)
gs = os.environ.get("GSPLAT_DIR", "/kaggle/working/gsplat")
open(f"{gs}/examples/datasets/__init__.py", "a").close()
# NOTE: the GLOMAP mapper also needed `--skip_retriangulation 1` in Person A's run. That flag lives in A's
# stage script, so it is NOT patched here: submit tasks with method "vggt" (the pipeline default) until A adds it.
"""),
code("""
# 5. Backend connection: URL of YOUR tunnel (changes every time a quick tunnel restarts) + token from Secrets
BACKEND_URL = "https://<random-words>.trycloudflare.com"     # <- edit
from kaggle_secrets import UserSecretsClient
os.environ["W1_WORKER_TOKEN"] = UserSecretsClient().get_secret("W1_WORKER_TOKEN")
import requests
r = requests.get(BACKEND_URL + "/worker/ping", headers={"Authorization": "Bearer " + os.environ["W1_WORKER_TOKEN"]}, timeout=20)
print(r.status_code, r.text)     # expect: 200 {"ok":true,...}   401 = wrong token   other/timeout = tunnel down
"""),
md("""
## 6. Run the worker
Runs until stopped (Interrupt), or until `--max-idle-minutes` without tasks (saves GPU quota). Each task: claim ->
download video -> pipeline (real progress) -> upload results -> done. A user's *Cancel* stops the pipeline within
one heartbeat. If this session dies, the backend re-queues the task after 5 minutes.

Optional flags: `--upload-ply` (also the ~250 MB `scene.ply`), `--upload-checkpoint` (gsplat checkpoints for the
Week 2 fine-tune). The viewer only needs the splat, which is uploaded first.
"""),
code("""
!python $WORKER --backend-url $BACKEND_URL --ml-dir $ML_DIR --root /kaggle/working/workspace --max-idle-minutes 30
"""),
md("""
## 7. After a run
* Check the task in the backend: `GET /tasks/{id}` - `result` holds per-stage runtime / peak VRAM and upload timings.
* Everything needed later is on the backend (`backend/data/workspace/<project_id>/`), so the Kaggle session can end.
* Kaggle limits (check the current values in your account): one session runs at most ~12 h and the GPU quota is
  weekly (~30 h). Stop the worker (or use `--max-idle-minutes`) when you are not demoing.
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).with_name("kaggle_worker.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)

#!/usr/bin/env python3
"""Regenerates notebooks/kaggle_week2.ipynb (Week 2 for Person A: click -> mask -> track -> rebuild -> hand-off)."""
import json
from pathlib import Path


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}


cells = [
md("""
# Week 2 - click an object -> mask -> track -> rebuild -> hand-off (Person A)
**Settings:** Accelerator = *GPU T4 x2* - Internet = *On*.

**Why a separate notebook from Week 1:** Week 1 costs ~1 h of GPU (poses + training) and is done once. Week 2 is
interactive - you try objects, fix clicks, re-run tracking - so it starts from the saved Week 1 files
(`demo_week2_inputs.zip`, built at the end of the Week 1 notebook) instead of redoing them. Different models too (SAM 2.1).

**Inputs:** add the Week 1 notebook's output (or a dataset containing `demo_week2_inputs.zip`) via *Add Input*.
**Output:** `handoff_demo.zip` for Person B and Person C (data + generated READMEs + numbers).
"""),
code("""
# 0. Code: clone your repo OR copy it from a Kaggle dataset (same as the Week 1 notebook)
# !git clone https://github.com/<your-team>/<your-repo>.git /kaggle/working/repo
# !cp -r /kaggle/input/<your-code-dataset>/week1_pipeline /kaggle/working/repo
%cd /kaggle/working/repo/week1_pipeline
"""),
code("""
# 1. Setup: gsplat (rebuild) + SAM 2.1 (masks). No VGGT / pycolmap needed here.  ~5-10 min
!bash scripts/setup_kaggle.sh w2
"""),
code("""
# 2. Environment + unpack the Week 1 results into the workspace
import os, re, shutil, glob
for line in open("/kaggle/working/env.sh"):
    m = re.match(r"export (\\w+)=(.*)", line.strip())
    if m: os.environ[m.group(1)] = os.path.expandvars(m.group(2))
os.environ["PATH"] = ":".join(x for x in os.environ["PATH"].split(":") if x and "/kaggle/working/cm/bin" not in x and "$PATH" not in x)
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
os.environ["W1_ROOT"] = "/kaggle/working/workspace"
os.environ["W1_PROJECT"] = "demo"
P = os.environ["W1_ROOT"] + "/demo"
ZIP = (glob.glob("/kaggle/input/**/demo_week2_inputs.zip", recursive=True) or ["<set me>"])[0]
print("inputs zip:", ZIP)
!mkdir -p $P && unzip -q -o "$ZIP" -d $P
import json
d = json.load(open(f"{P}/week2/inputs.json"))
print("pose method:", d["pose_method"], "| frames:", d["registered_frames"], "/", d["total_frames"], "| same image space:",
      d["images"]["same_pixels"], "| depth:", d["depth"]["kind"], "| checkpoint:", d["checkpoint"])
"""),
md("""
## 2b. Calibrate: make the splat checkpoint and the poses agree (REQUIRED)
gsplat trains in a *normalized* world space by default, so rendering the checkpoint with raw COLMAP poses gives garbage
(Week 2 first run: fine-tune PSNR 8 instead of ~26, 3D lift found nothing). This cell tests both pose sets on a few photos,
keeps the one that reproduces them, and re-renders the splat depth. Expect: right set ~25 dB, wrong set ~8 dB.
"""),
code("""
!python stages/w2_calibrate.py --project-dir $P
"""),
code("""
# 3. GPU self-test (2 min): SAM 2 image+video, gsplat API, fused-ssim, a 20-step fine-tune on YOUR scene.
#    If something FAILs, send me the printed line - it is the exact API/dtype problem.
!python tools/w2_selftest.py --project-dir $P --finetune-steps 20
"""),
md("""
## A. Pick the object (click -> mask)
Run the next cell, read the pixel coordinates off the axes, put them in `CLICKS` (x right, y down, in pixels of the shown frame).
`positive: false` = "not this" (use it to cut the mask off a neighbouring object).
"""),
code("""
import json, cv2, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
names = sorted(p.name for p in Path(P, "frames/images").glob("*.jpg"))
print(len(names), "frames:", names[0], "...", names[-1])
def show(name, clicks=None, mask_sub=None, size=9):
    img = cv2.cvtColor(cv2.imread(f"{P}/frames/images/{name}"), cv2.COLOR_BGR2RGB)
    if mask_sub:
        mp = Path(P, "edits", EDIT, mask_sub, Path(name).stem + ".png")
        if mp.exists():
            m = cv2.imread(str(mp), 0) > 127
            img = img.copy(); img[m] = (0.5 * img[m] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    plt.figure(figsize=(size, size * img.shape[0] / img.shape[1])); plt.imshow(img)
    for c in clicks or []:
        plt.scatter(c["x"], c["y"], c="lime" if c.get("positive", True) else "red", s=120, edgecolors="k")
    plt.grid(alpha=.3); plt.title(name); plt.show()
EDIT = "e1"
REF = names[43]                      # <- the frame you click on (000044: grey-blue chair large + clear); the SAME variable feeds show() and the segmenting
show(REF)
"""),
code("""
# 4. Segment: your clicks -> mask on REF.  Edit CLICKS and re-run until the mask is right.
CLICKS = [{"x": 260, "y": 440, "positive": True}]          # <- read x,y off the picture of REF (approximate); add negative clicks to trim
!python stages/w2_segment.py --project-dir $P --edit-id $EDIT --frame $REF --clicks '{json.dumps(CLICKS)}'
show(REF, CLICKS, "masks")
# Guard: the mask of REF must exist and be plausible, otherwise STOP (a wrong --frame once made a 2k px mask on another frame)
mp = Path(P, "edits", EDIT, "masks", Path(REF).stem + ".png")
assert mp.exists(), f"no mask for {REF}: the clicks were segmented on another frame"
area = int((cv2.imread(str(mp), 0) > 127).sum()); print("mask area px:", area)
assert area > 3000, "mask is tiny - wrong object? move/add clicks"
"""),
md("""
## B. Track it through the video
Default = **both** methods fused (SAM 2 video propagation + 3D lift with visibility check, SAM 2 refines the edges).
Green = your click, orange = SAM 2 propagation, blue = lifted from 3D, red border = object not in that frame.
"""),
code("""
# simplest first (PDF 4.2): SAM 2 alone. Check the grid, THEN set TRACKER = "both" to add the 3D lift.
TRACKER = "sam2"          # sam2 | both | lift
!python stages/w2_track.py --project-dir $P --edit-id $EDIT --tracker $TRACKER
!python tools/w2_preview.py --project-dir $P --edit-id $EDIT --cols 6
from IPython.display import Image, display
display(Image(f"{P}/edits/{EDIT}/preview_masks.png"))
"""),
code("""
# 5. FALLBACK (spec 4.4): frames where tracking missed the object -> mark them yourself, then re-run track.
#    Each marked frame becomes a new anchor ("user"), so the next tracking pass is better everywhere.
FIX = []      # e.g. [{"name": "000030.jpg", "clicks": [{"x": 300, "y": 200, "positive": True}]}]
if FIX:
    !python stages/w2_segment.py --project-dir $P --edit-id $EDIT --add-frames '{json.dumps(FIX)}'
    !python stages/w2_track.py --project-dir $P --edit-id $EDIT --tracker both
    !python tools/w2_preview.py --project-dir $P --edit-id $EDIT --cols 6
    display(Image(f"{P}/edits/{EDIT}/preview_masks.png"))
"""),
md("""
## C. Evaluate the tracking (this table goes in the report)
Label 10-30 frames spread over the video: either clicks on the object (SAM builds the mask) or `{"absent": true}` where the object
is not visible. Then the three methods are compared on those frames.
"""),
code("""
GT = {   # <- edit: frame -> clicks | {"absent": true}
    # "000010.jpg": {"clicks": [{"x": 400, "y": 250, "positive": True}]},
    # "000055.jpg": {"absent": True},
}
json.dump(GT, open(f"{P}/edits/{EDIT}/gt.json", "w"))
if GT:
    !python stages/w2_track.py --project-dir $P --edit-id $EDIT --tracker sam2 --out track_eval/sam2/masks
    !python stages/w2_track.py --project-dir $P --edit-id $EDIT --tracker lift --out track_eval/lift/masks
    !python tools/w2_eval_tracking.py --project-dir $P --edit-id $EDIT --gt $P/edits/$EDIT/gt.json --variants sam2 lift both
"""),
md("""
## C2. Diffusion edit (Qwen-Image-2.1): reference frame first, then every other frame
Uses the masks from A-B. Writes `edits/<id>/edited/` (what section D reads), so if you run this, **skip the fake-edit cell in D**.
Run it after sections A-C: the install upgrades `transformers`, which SAM 2 and gsplat do not use. It needs the Hugging Face token as a Kaggle
secret named `HF_TOKEN` if the model download asks for login. One GPU (`--gpu 0`), one process, 4-bit by default.
"""),
code("""
!bash scripts/setup_kaggle.sh diffusion
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    print("no HF_TOKEN secret (fine if the model downloads without login):", type(e).__name__)
"""),
code("""
# 1. Several seeds on the frame where the object is large and fully visible. Look at the sheet and pick the best seed.
PROMPT = "Change the light grey-blue upholstered chair into a dark green leather chair. Keep the exact same chair shape, size and position, with realistic leather sheen matching the room lighting."   # this room has no sofa
!python stages/w2_edit.py --project-dir $P --edit-id $EDIT --phase reference --prompt "{PROMPT}" --n-seeds 2 --gen-scale 0.7 --offload model
display(Image(f"{P}/edits/{EDIT}/preview_reference.png"))
"""),
code("""
# 2. Every other frame: its own mask, the same prompt, the chosen reference edit as an extra condition image.
#    Smoke test first: add  --limit 2  to edit only two frames and check the sheet before the full run.
REF_SEED = 0          # <- the seed you picked (the first candidate is seed 0)
!python stages/w2_edit.py --project-dir $P --edit-id $EDIT --phase rest --ref-seed $REF_SEED --prompt "{PROMPT}" --gen-scale 0.7 --offload model
display(Image(f"{P}/edits/{EDIT}/preview_edited.png"))
"""),
code("""
# 3. VRAM + seconds per frame for this quantization setting. Repeat step 2 with  --quant 8bit  or
#    --quant none --offload model  (and --tag) to fill more rows; the table below collects every run.
!python tools/w2_edit_table.py --project-dir $P --edit-id $EDIT
"""),
md("""
## D. Rebuild (Person C + A): fine-tune vs from scratch
The diffusion edit is Person B's. Until their `edited/` frames arrive, make a **fake edit** (recolour the object) to exercise the
whole chain. To use B's real edit instead: copy their `edits/<id>/edited/` folder (PNGs + `index.json`) into `$P/edits/$EDIT/` and skip the fake cell.
"""),
code("""
!python tools/w2_make_test_edit.py --project-dir $P --edit-id $EDIT --color 0,0,255
"""),
code("""
# fine-tune: start from the trained Gaussians (needs gs/ checkpoint) - minutes
!python stages/w2_rebuild.py --project-dir $P --edit-id $EDIT --mode finetune --iters 3000
"""),
code("""
# from scratch: train again on the edited scene (same iterations as Week 1 quick) - the slow baseline to compare against
!python stages/w2_rebuild.py --project-dir $P --edit-id $EDIT --mode scratch --iters 7000
"""),
md("## E. Hand-off for Person B and Person C"),
code("""
!python tools/make_handoff.py --project-dir $P --out /kaggle/working/handoff_demo.zip --no-checkpoint
!cat $P/edits/$EDIT/metrics/rebuild_comparison.md
!unzip -l /kaggle/working/handoff_demo.zip | tail -3
# Download handoff_demo.zip from the Output tab and send it to B and C. SUMMARY.md inside has every number.
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_week2.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)

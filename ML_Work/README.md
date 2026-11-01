# Week 1 — From Video to a Browsable 3D Room · Person A (Reconstruction Pipeline)

Implements exactly the Person A scope of the Week 1 brief: **ffmpeg frame extraction → VGGT and/or GLOMAP poses →
required comparison → COLMAP-style folder → 3D Gaussian Splatting on a T4 → `.ply` → `.splat`**, with a real
per-stage progress feed for the backend and everything stored where Week 2 can reuse it.

> Person B (backend/worker) and Person C (frontend/viewer) only need the two interfaces in
> [§7](#7-interfaces-for-person-b-and-person-c).

---

## 1. What runs where

| Where | What | Needs |
|---|---|---|
| **Your laptop / any CPU box** | `pytest` (15 tests), frame extraction (stage 1), reports, reading/writing COLMAP files | Python ≥ 3.10, `ffmpeg`, `pip install -r requirements.txt` |
| **Kaggle notebook, GPU T4 ×2, Internet on** | VGGT (2a), GLOMAP (2b), 3DGS training (4) | `scripts/setup_kaggle.sh` (installs VGGT, COLMAP, GLOMAP, gsplat) |
| **Backend (Person B)** | calls `run_pipeline.py` with `--task-id/--backend-url`, serves `export/` | see §7 |

Debug on CPU first (the PDF: *“don't waste GPU hours on code that could run on CPU”*): stage 1, stage 3, stage 5 (from a PLY),
reports and all tests run without a GPU.

## 2. Repo layout

```
week1_pipeline/
├─ run_pipeline.py              orchestrator: every stage = its own process (VRAM freed between stages)
├─ stages/
│  ├─ s1_extract_frames.py      ffmpeg fps (auto/1-4), resize ≤960 px, Laplacian-variance blur filter, index json
│  ├─ s2a_poses_vggt.py         VGGT, fp16, chunked + Sim(3) merge, depth/conf saved, COLMAP export
│  ├─ s2b_poses_glomap.py       COLMAP features → sequential matching → GLOMAP mapper → PINHOLE COLMAP folder
│  ├─ s3_compare_poses.py       REQUIRED experiment: frames/runtime/VRAM/point cloud + plot; picks → scene/
│  ├─ s4_train_3dgs.py          gsplat (or Inria) on scene/, 7k quick / 30k full, T4-ready
│  └─ s5_export_splat.py        scene.ply, scene.splat, cameras.json, scene_meta.json
├─ common/                      colmap_io (text+binary), geometry, vggt_merge, gs_io, metrics (VRAM), progress, config
├─ tools/make_reports.py        VRAM/runtime table + Week-2 handoff + manifest.json (generated from measurements)
├─ scripts/setup_kaggle.sh      one-time installs on Kaggle;  scripts/make_notebook.py regenerates the notebook
├─ notebooks/kaggle_pipeline.ipynb
├─ tests/                       15 tests (synthetic geometry, merge, extraction, export, compare/report)
└─ docs/generated/              VRAM_RUNTIME_TABLE.md, WEEK2_HANDOFF.md   (written by tools/make_reports.py)
```

## 3. Quick start

### 3.1 On your machine (CPU) — verify the repo, test stage 1 on your own video

```bash
cd week1_pipeline
pip install -r requirements.txt            # + ffmpeg on PATH
python -m pytest tests -q                  # 15 passed

python stages/s1_extract_frames.py --project demo --video /path/to/room.mp4 --fps auto
ls workspace/demo/frames/images | head ; cat workspace/demo/frames/frames_index.json | head -30
```
Look at 5–10 frames by eye: sharp, overlapping, the room only.  *Shoot* the video per the brief: one room, slow motion,
good light, no fast pans, 2–5 min (30–60 s is plenty while debugging).

### 3.2 On Kaggle (GPU) — the real run

1. New notebook → *Settings*: **Accelerator GPU T4 ×2**, **Internet On**.
2. Add the repo (git clone, or upload `week1_pipeline/` as a Dataset) and your video as a Dataset.
3. Open `notebooks/kaggle_pipeline.ipynb` (or paste its cells). Edit `REPO` and `VIDEO`, then run top to bottom.
   The notebook just calls the scripts below; the first run is ~10–20 min of installs (gsplat compiles CUDA once).
4. Download `/kaggle/working/demo_export.zip` (splat + metrics + manifest) **before the session ends**.

Equivalent commands (each is a separate process):

```bash
bash scripts/setup_kaggle.sh all && source /kaggle/working/env.sh
export W1_ROOT=/kaggle/working/workspace W1_PROJECT=demo TORCH_CUDA_ARCH_LIST=7.5

python stages/s1_extract_frames.py --video room.mp4 --fps auto --long-side 960
python stages/s2a_poses_vggt.py   --chunk-size 32 --overlap 8 --memory-sweep 8,16,32,48
python stages/s2b_poses_glomap.py --overlap 12
python stages/s3_compare_poses.py --choose auto          # writes metrics/pose_comparison.{md,png,json} and scene/
python stages/s4_train_3dgs.py    --trainer gsplat --mode quick     # 7k iters: does the pipeline work?
python stages/s4_train_3dgs.py    --trainer gsplat --mode full      # 30k iters: quality
python stages/s5_export_splat.py
python tools/make_reports.py                              # tables + Week-2 handoff + manifest.json
```
One shot: `python run_pipeline.py --video room.mp4 --method both --train-mode quick`
(`--method vggt|glomap` for a production run with a single pose method, `--from-stage train`, `--only export`, …).

## 4. How each requirement of the brief is met

| Brief (Person A, Week 1) | Where / how |
|---|---|
| ffmpeg fps / resize / blur filter | `s1`: fps **auto** (optical-flow motion probe → 1–4 fps) or manual; long side ≤ 960 (never upscales, even dims); drop frames below `max(15, 0.5·median)` Laplacian variance but never > 30 %; cap 400 frames |
| VGGT in chunks, fp16, merge chunks | `s2a` + `common/vggt_merge.py`: 32-frame chunks, 8 shared; T4 → **float16** (bf16 only if compute capability ≥ 8); per-pair **robust Sim(3)** from the shared frames' pixel-aligned depth (tested on synthetic data); `--memory-sweep` records VRAM vs frames |
| GLOMAP (features → sequential matching → mapper) | `s2b`; uses `glomap mapper`, or `colmap global_mapper` (GLOMAP was merged into COLMAP 3.14) |
| **Required experiment** | `s3`: registered frames, runtime, peak VRAM, point-cloud plot, Sim(3)-aligned trajectory agreement, **draft justification paragraph** from the numbers (edit it) |
| COLMAP-style folder, intrinsics checked | both methods → `images/` + `sparse/0/{cameras,images,points3D}.{bin,txt}`, **PINHOLE** at frame resolution (VGGT intrinsics rescaled from its 518-px input; GLOMAP's SIMPLE_RADIAL is undistorted via `colmap image_undistorter` when \|k₁\| > 0.01, else k is dropped and noted in `glomap_meta.json`) |
| 3DGS on T4, start from sparse cloud, 7k then 15–30k | `s4`: gsplat `simple_trainer.py` (or `--trainer inria`), `TORCH_CUDA_ARCH_LIST=7.5`, `--mode quick/full/custom --iters N`; holds out every 8th frame for PSNR/SSIM |
| `.ply` → `.splat` | `s5` (also converts gsplat `.pt` checkpoints to standard PLY); `.ksplat` optional, command in the script docstring |
| Log time, #Gaussians, size, peak VRAM | `metrics/*.json` (nvidia-smi sampler works for subprocess stages + torch peaks), `tools/make_reports.py` → `docs/generated/VRAM_RUNTIME_TABLE.md` |
| Real progress, not a timer | `common/progress.py`: `POST /tasks/{id}/progress` with stages *extracting frames → estimating poses → training → exporting* (0-15-45-90-100 %) |
| Week 2 handoff | `docs/generated/WEEK2_HANDOFF.md` + `manifest.json`; §5 below |

Week 1 deliverables checklist (§5 of the brief): repo+README ✔ · Kaggle notebook ✔ · VGGT-vs-GLOMAP report (`s3` output) ·
demo video (record Person C's UI + a run) · VRAM/runtime table (generated) · Week-2 handoff page (generated).
*You still have to run the GPU stages, look at the plots, and write the final paragraph.*

## 5. On-disk contract (Week 2 reads exactly this)

```
workspace/<project>/
  video/input.mp4
  frames/images/000001.jpg ...           resized, blur-filtered; names == names inside the COLMAP model
  frames/frames_index.json               name, source video time, blur score
  poses/vggt/   images→frames, sparse/0/, depth/*.npy, conf/*.npy, points.ply, vggt_meta.json
  poses/glomap/ images, sparse/0/, points.ply, glomap_meta.json
  scene/        images, sparse/0/, chosen.json          ← the chosen method: input to 3DGS and to Week 2 re-training
  gs/           gsplat_<iters>/ckpts/*.pt, ply/, stats/;  latest.json
  export/       scene.ply, scene.splat, cameras.json, scene_meta.json
  metrics/      s1_..json, s2a_..json, ..., pose_comparison.*, vggt_memory_sweep.json, progress.jsonl, logs/
  manifest.json
```
Facts Week 2 must know: COLMAP world is **+y down** (viewer up = (0,−1,0)); frames are even-sized, not ×8/×16;
depth `.npy` is float16 z-depth at VGGT resolution in the **same world scale** as `scene/sparse/0` (only when the chosen
method is VGGT — otherwise render depth from the splat); `common/geometry.py` has `unproject_depth` / `project` for the
"lift mask to 3D and reproject" tracker.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `VGGT is not installed` | `bash scripts/setup_kaggle.sh vggt && source /kaggle/working/env.sh` |
| CUDA OOM in VGGT | lower `--chunk-size` (16–24) — see `metrics/vggt_memory_sweep.json` for the VRAM/frames curve |
| `fused-ssim` build error (gsplat) | `pip install --no-build-isolation "git+https://github.com/rahul-goel/fused-ssim"` (needs nvcc, present on Kaggle GPU images) |
| `No module named pip/cv2/pycolmap` right after setup | `/kaggle/working/cm/bin` is on PATH and its bare python shadows the system one. Setup now exposes only `colmap`/`glomap` wrappers in `/kaggle/working/tools`; check with `!which python` (should be `/usr/local/bin/python`) |
| GLOMAP stage crashes in `OpenGLContextManager` | COLMAP GPU SIFT needs a display. The stage now uses the CPU by default (`--gpu` opts in) |
| `No global mapper found` | install `glomap` (conda-forge) or COLMAP ≥ 3.14 (`scripts/setup_kaggle.sh colmap` prints options); until then `--method vggt` |
| Only a few frames registered by GLOMAP | slower/steadier camera, more overlap (`--overlap 20`), more fps; don't fix by lowering resolution |
| `chunk alignment weak` warning | increase `--overlap` (12–16); check for fast pans between those frames |
| gsplat: flag not recognised | `s4` probes `--help` and adapts underscore/hyphen style; for other changes pass `--extra --some-flag v` |
| gsplat first run very slow | CUDA JIT compile (once); `TORCH_CUDA_ARCH_LIST=7.5` is already set |
| Splat looks upside down in the viewer | COLMAP is y-down: set viewer up-vector to (0,−1,0) / flip y |
| Session about to expire | step 6 of the notebook zips `export/ manifest.json metrics/ scene/` into `/kaggle/working` |

## 7. Interfaces for Person B and Person C

**Person B (worker/back end).** Run the reconstruction as one process and stream real progress:
```bash
python run_pipeline.py --video <uploaded.mp4> --root <storage> --project <project_id> \
       --method vggt --train-mode quick --backend-url https://<tunnel> --task-id <task_id>
```
Progress body: `{"status":"running|done|failed","stage":"extracting frames|estimating poses|training|exporting","progress":0-100,"error":null}`;
a network failure never kills a stage. Exit code ≠ 0 on failure (the failing stage also posts `status=failed`).
`GET /projects/{id}/scene` → serve `export/scene.splat` + `export/scene_meta.json`.

**Person C (viewer).** Load `export/scene.splat`; `export/cameras.json` lists every training camera
(`position`, `rotation` = camera-to-world, `fx fy cx cy`, `img_name`) for the Week-2 hooks *“load a specific camera pose”* and
*“snap to a training camera”*; `scene_meta.json` has the bbox, #Gaussians and the up-vector note.
Fallback if splat rendering fails: render `scene/points.ply` (plain coloured points) in three.js.

## 8. Honest status

* Verified here on CPU: the 15 tests — geometry, COLMAP text+binary round-trip, **VGGT chunk merge on a synthetic room
  with random per-chunk Sim(3) gauges**, ffmpeg extraction (resize, blur removal, caps, auto-fps), `.splat` encoding/decoding,
  progress feed, comparison/choose/report generation.
* **Not run in this environment (no GPU, no torch, no COLMAP):** the actual VGGT forward pass, GLOMAP, and gsplat training.
  They are written against the current public APIs (VGGT `aggregator/camera_head/depth_head`, `colmap`/`glomap` CLIs,
  gsplat `simple_trainer.py`) with defensive flag detection, but expect to fix a small thing or two on the first Kaggle run
  — the riskiest step is installing a global mapper (`scripts/setup_kaggle.sh colmap`).
* GLOMAP's standalone repo has been folded into COLMAP (3.14, `colmap global_mapper`); the brief's “GLOMAP” maps to either.

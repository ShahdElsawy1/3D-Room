# Notes for Person A (from the backend integration)

Nothing in `ML_Work/` was changed apart from the approved `W1_WORKER_TOKEN` header in `common/progress.py`
(commit `a19f58b`). These are findings from running your real outputs (`demo_export.zip`) and your CPU stages through
the backend and worker. The decisions are yours.

## 1. Absolute `/kaggle/working/...` paths in your JSON files (affects Week 2)

Your files store absolute paths of the machine they were written on:

| file | example |
|---|---|
| `manifest.json` | `"root": "/kaggle/working/workspace/demo"`, and every `"dir"`, `"sparse"`, `"splat"`, … |
| `scene/chosen.json` | `"from": "/kaggle/working/workspace/demo/poses/glomap"` |
| `gs/latest.json` | `"ply"`, `"ckpt"`, `"result_dir"` |
| `metrics/s4_train_3dgs.json` | `"result_dir"` |

The backend stores these files **byte-for-byte**. They then live at `backend/data/workspace/<project_id>/`, and in
Week 2 a new Kaggle session downloads them to `/kaggle/working/workspace/<project_id>/`. So the absolute paths inside
them **do not point to the files any more** (another root, and on the backend another OS).

**Rule for Week 2 code (tracking, rebuild): never open a path read from `manifest.json` / `latest.json` / `chosen.json`.
Build every path from the project folder with your own `Config`:**
```python
from common.config import Config
cfg = Config(root=ROOT, project_id=PROJECT_ID)          # same layout everywhere
cfg.images_dir                                          # frames/images
cfg.scene_dir / "sparse" / "0"                          # poses + intrinsics + points3D
cfg.poses_dir("vggt") / "depth"                         # VGGT depth (when uploaded)
cfg.export_dir / "scene.ply"
```
If you need a value that only exists as an absolute path (e.g. which checkpoint `latest.json` chose), keep only the
part after the project folder:
```python
def rel_to_project(abs_path: str, project_id: str) -> str:
    """'/kaggle/working/workspace/<pid>/gs/gsplat_30000/ckpts/ckpt_29999.pt' -> 'gs/gsplat_30000/ckpts/ckpt_29999.pt'"""
    p = abs_path.replace("\\", "/")
    return p.split(f"/{project_id}/", 1)[1] if f"/{project_id}/" in p else p
```
The worker runs your pipeline with `--project <backend project id>`, so the folder name is the same on Kaggle and on
the backend, and this split always works.

**Suggestion (optional):** write project-relative paths in these files (e.g. `"splat": "export/scene.splat"`). Then
they stay valid wherever the project folder is copied. The backend doesn't depend on it either way: the API never
returns these paths (its scene metadata is whitelisted).

## 2. GLOMAP fix not in the repo
In `updated-week1.ipynb` the GLOMAP mapper only worked with `--skip_retriangulation 1`, added via `sed` on Kaggle.
`stages/s2b_poses_glomap.py` in the repo doesn't have it, so a fresh run with `method: glomap` hits the same crash.
Until it is added, tasks should use `method: vggt` (the pipeline default). Two other fixes from that run (pycolmap
`np.uint64(-1)`, gsplat `datasets/__init__.py`) are third-party and are applied by the worker notebook.

## 3. The notebook's zip has no frames
`demo_export.zip` contains `frames/frames_index.json` but not `frames/images/`. That doesn't matter any more: the
worker uploads `frames/` (images + index) directly. Just don't rely on the zip for Week 2.

## 4. Symlinks
`scene/images` and `poses/<method>/images` are symlinks on Kaggle. The worker uploads them as real files (tar with
dereference), and the backend never stores links. `poses/<method>/images` is not uploaded at all, because the images
used for training are in `scene/images`. If Week 2 needs the undistorted GLOMAP images, they are those.

## 5. `progress.py` ignores HTTP errors
This is fine by design (a dead backend must not kill a stage). But a wrong token used to be invisible. The worker now
checks its token with `/worker/ping` before running anything.

## 6. Where Week 2 inputs are on the backend
`backend/data/workspace/<project_id>/` has your exact layout: `frames/`, `scene/` (sparse/0 + images +
points.ply), `poses/` (incl. VGGT `depth/`, `conf/` when VGGT ran), `export/`, `metrics/`, `manifest.json`, plus
`gs/` if the worker ran with `--upload-checkpoint`. A Kaggle job downloads any file with
`GET /worker/projects/{id}/files/<path>` (worker token).

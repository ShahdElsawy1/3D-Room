# Week 2 interfaces: proposal for Person A and Person C

**Status:** the backend side is implemented as written here (M8): endpoints in [API.md](API.md), worker contract in [WORKER_PROTOCOL.md](WORKER_PROTOCOL.md). Person A's and C's stages are still to be written; CPU stand-ins that follow this contract are in `backend/tests/stubs_w2/`. Changes are still possible: tell Person B.

Goal: agree on files and endpoints **before** M7, so the three parts can be built in parallel and plugged together.
Everything here is a proposal from Person B. Please answer the questions marked **Q** or change whatever doesn't fit.

## Who owns what (spec Week 2 §2)
| Part | Owner | Backend's role (B) |
|---|---|---|
| Click → reference mask, refine clicks, tracking, frame-marking fallback | **A** | task types, store masks, serve them to C |
| Diffusion edit of the frames + mask compositing | **B** | all of it |
| Edit UI, frame grid, before/after toggle, rebuild trigger, end-to-end | **C** | endpoints C calls |
| Rebuild the splat from edited frames, poses reused (scratch vs fine-tune) | **C with A** | rebuild task type, keep the original splat, versioned scenes |

All GPU work keeps running through the existing pull-based worker (new task types next to `reconstruct`).

## 1. Files (contract with Person A)
Everything for one edit lives under the project folder, next to Week 1's layout, which stays untouched:
```
<project>/edits/<edit_id>/
  request.json                  ref_frame, clicks, prompt, seeds (written by the backend)
  masks/<frame_stem>.png        A: 8-bit single channel, 0 = background, 255 = object,
                                   exactly the size of frames/images/<frame_name> (no padding, no resize)
  masks/index.json              A: see below
  candidates/seed_<n>.png       B: edited reference frame, one per seed (the user picks one)
  edited/<frame_stem>.png       B: final composited frames, PNG (lossless, see note)
  edit_report.json              B: per-frame seconds, peak VRAM, model, dtype, quality metrics
  scene/images/                 C/A: rebuild input = original images with edited ones replaced, same names
  gs/, export/                  C/A: rebuilt splat, same layout and file names as Week 1
```
`masks/index.json`:
```json
{
  "ref_frame": "frame_000123.jpg",
  "method": "sam2_video",
  "frames": [
    {"name": "frame_000123.jpg", "present": true,  "source": "click",      "score": 0.97, "area_px": 41230},
    {"name": "frame_000124.jpg", "present": true,  "source": "propagated", "score": 0.91, "area_px": 40011},
    {"name": "frame_000300.jpg", "present": false, "source": "propagated", "score": 0.12, "area_px": 0}
  ]
}
```
- One entry per frame in `frames/frames_index.json`, same names. Mask PNG only for `present: true`.
- `source`: `click` (reference), `propagated` / `lifted` / `reid` (tracking), `user` (fallback marking).
- The diffusion step edits exactly the frames with `present: true`. Frames the user removes in the grid become `present: false`.
- **Note on PNG:** edited frames are saved as PNG because re-saving as JPEG would change pixels outside the mask. That
  would break the "nothing outside the mask changed" guarantee. The rebuild must accept PNG next to the original JPGs
  (gsplat's COLMAP loader reads by file name from `images.bin`). **Q (A):** is it OK if
  `scene/images/frame_000123.jpg` contains PNG bytes under its original name? The other option is to save edited frames
  as JPEG quality 100 and measure the change outside the mask.

**Q (A):**
1. Do the masks come as above (per-frame PNG + index), or do you prefer another format (e.g. RLE in one JSON)?
2. Which tracking method do you start with, and on which GPU? Spec example: GPU 0 = diffusion, GPU 1 = SAM + 3DGS.
   The worker can run two processes pinned with `CUDA_VISIBLE_DEVICES` if we want segment and edit tasks to overlap.
3. Interactive click → mask goes through the queue (worker polls every few seconds). Is ~3–5 s per click acceptable,
   or should SAM stay loaded in the worker between clicks (a "session" task that stays alive while the user refines)?
4. Your stage scripts, same style as Week 1 (one process per stage, progress posted to the backend). The worker
   already calls them like this (`--project-dir <root>/<project_id> --backend-url ... --task-id ...` appended):
   - `ML_Work/stages/w2_segment.py --edit-id E --frame F --clicks '[{"x":..,"y":..,"positive":true}]'`
     or `--add-frames '[{"name":F,"clicks":[...]}]'` (fallback); the existing `masks/` is already downloaded, so
     keep the other frames' masks and update `index.json`.
   - `ML_Work/stages/w2_track.py --edit-id E` (inputs: `frames/`, `scene/`, `export/cameras.json`, `masks/`).
   - `ML_Work/stages/w2_rebuild.py --edit-id E --mode finetune|scratch` (Person C with A; inputs: `frames/`,
     `scene/`, `edits/E/edited/`, `edits/E/masks/`, `gs/` if uploaded). Output: `edits/E/export/scene.splat`,
     `cameras.json`, `scene_meta.json` (+ `edits/E/metrics/`), the same names as Week 1.
   Working examples of all three (CPU stand-ins, no real model): `backend/tests/stubs_w2/`.

## 2. Endpoints (contract with Person C)
Same conventions as Week 1 ([API.md](API.md)): JWT, task objects with `status/stage/progress/error`, signed file URLs.
One project can have several edits (edit history comes for free; undo = switch scene version).

| Step in spec §4.5 | Call | Returns |
|---|---|---|
| list training frames for the picker | `GET /projects/{id}/frames` | `[{name, index, url, thumb_url, camera_id}]` |
| 1. start an edit on a frame | `POST /projects/{id}/edits` `{ref_frame, clicks:[{x,y,positive}]}` | edit object + `segment` task |
| 2. refine the mask | `POST /edits/{eid}/clicks` `{clicks:[...]}` | new `segment` task; `GET /edits/{eid}` has `ref_mask_url` |
| 4. track | `POST /edits/{eid}/track` | `track` task (progress = frames done) |
| 4. grid with mask overlay | `GET /edits/{eid}/frames` | `[{name, url, mask_url, present, source, score}]` |
| 4. remove wrong / mark missing frames | `PATCH /edits/{eid}/frames` `{remove:[names], add:[{name, clicks}]}` | `segment` task for the added frames |
| 3+5. prompt + seeds for the first frame | `POST /edits/{eid}/preview` `{prompt, seeds:3}` | `edit_preview` task → `candidates:[{seed,url}]` |
| 5. accept or regenerate | `POST /edits/{eid}/accept` `{seed}` or `preview` again | `edit_apply` task, stage `"editing frame x of N"` |
| 6. rebuild | `POST /edits/{eid}/rebuild` `{mode:"finetune"\|"scratch"}` | `rebuild` task |
| 6. before/after | `GET /projects/{id}/scene?version=original\|<eid>` | same scene object as Week 1; `versions:[...]` lists both |

Edit object: `{id, project_id, status: "segmenting"|"mask_ready"|"tracking"|"tracked"|"previewing"|"preview_ready"|
"editing"|"edited"|"rebuilding"|"done"|"failed", ref_frame, prompt, chosen_seed, current_task, created_at}`.

**Q (C):**
1. Frame picker: are thumbnails needed (backend makes 240 px JPEGs once), or are the 960 px frames fine?
2. Click coordinates: in the frame's own pixels (what this proposal assumes) or 0–1 normalized?
3. Do you use option 1 (click on a reference frame) or option 2 (3D click → nearest training frame)? With option 2 the
   viewer still sends a frame name + 2D point, so the API is the same.
4. Before/after: two scene versions loaded on demand (proposal), or both splats loaded at once?
5. Progress: polling `GET /tasks/{id}` like Week 1 (proposal) or do you want server-sent events?

## 3. What B builds once this is agreed
- **M7** (B, Kaggle): `edit` stage script: reference frame × seeds → candidates; other frames conditioned on the chosen
  candidate; crop → latent blending → hard composite; test that every pixel outside the dilated mask is unchanged;
  per-frame seconds and peak VRAM. Model/precision per the decision on [W2_DIFFUSION_RESEARCH.md](W2_DIFFUSION_RESEARCH.md).
- **M8** (B, backend): `Edit` table, the endpoints above, task types `segment / track / edit_preview / edit_apply /
  rebuild`, artifact upload of masks/candidates/edited frames/rebuilt splat, scene versions, the original kept read-only.

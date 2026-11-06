# Worker protocol (Kaggle notebook ↔ backend)

The spec (§4.3) asks for a **pull-based worker**. The notebook polls the backend for pending tasks and sends
results back, so the notebook itself is never exposed. The only public URL is the backend's Cloudflare tunnel.

## Authentication
Every request below needs:
```
Authorization: Bearer <BACKEND_WORKER_TOKEN>
```
- The token is `BACKEND_WORKER_TOKEN` if set. Otherwise the backend generates one on first start and stores it in
  `backend/data/.worker_token`. Copy it into the Kaggle notebook as a **Kaggle Secret** (Add-ons → Secrets),
  never as plain text in a cell that gets shared.
- On Kaggle, export it as **`W1_WORKER_TOKEN`**. `ML_Work/common/progress.py` then sends it with every progress post.
  This is the only change made to Person A's pipeline, and it is backward compatible: no variable means no header.
- A user's JWT is **not** accepted on these routes, and the worker token is not accepted on user routes.
- `progress.py` ignores HTTP errors (by design, a dead backend must not kill a stage), so a wrong token would
  fail silently. The worker therefore calls `GET /worker/ping` at start-up and stops if it returns `401`.

## Task life cycle
```
pending ──claim──▶ running ──complete──▶ done
                     │ fail(retryable) and attempts < max_attempts ──▶ pending   (another claim retries it)
                     │ fail(retryable=false) or attempts used up   ──▶ failed
                     │ no progress/heartbeat for BACKEND_LEASE_SECONDS (300) ──▶ pending | failed
                     └ user cancels ──▶ cancelled (worker is told on its next heartbeat/progress post)
```
`attempts` increases on every claim. `BACKEND_MAX_ATTEMPTS` (default 2) applies to new tasks.

## Endpoints

| Method | Path | Body | Reply |
|---|---|---|---|
| GET | `/worker/ping` | – | `{"ok": true, "lease_seconds": 300}` |
| POST | `/worker/claim` | `{"worker_id": "kaggle-<session>", "types": ["reconstruct"]?}` | `200` claim (below) or `204` nothing to do |
| GET | `/worker/tasks/{id}/video` | – | the input video (supports Range) |
| POST | `/tasks/{id}/progress` | progress record (below) | `{"ok": true, "cancel": false}` |
| POST | `/worker/tasks/{id}/heartbeat` | `{"worker_id", "stage"?, "progress"?}` | `{"status", "cancel", "lease_expires_at"}` |
| POST | `/worker/tasks/{id}/complete` | `{"worker_id", "result": {...metrics}}` | `{"ok": true, "status": "done"}` |
| POST | `/worker/tasks/{id}/fail` | `{"worker_id", "error", "retryable": true}` | `{"ok", "status": "pending"\|"failed", "attempts", "max_attempts"}` |

**Claim reply**
```json
{"task": {...Task object, status "running"...},
 "lease_seconds": 300,
 "video_url": "/worker/tasks/<id>/video", "video_filename": "input.mp4",
 "video_sha256": "…", "video_size_bytes": 123456789,
 "pipeline_args": ["--method", "vggt", "--train-mode", "quick"]}
```
`pipeline_args` are passed straight to `ML_Work/run_pipeline.py`, along with
`--video <downloaded> --root <workspace> --project <project_id> --backend-url <BACKEND_URL> --task-id <id>`.

**Progress record.** This is exactly what `ML_Work/common/progress.py` sends:
`{"status": "running"|"done"|"failed", "stage": "...", "progress": 0-100, "error": null, "t": <unix time>}`.
- `running`: stage and progress are stored, and the lease is renewed.
- `failed`: the error is stored, and the task stays `running` until the worker calls `/fail`. The worker decides whether to retry.
- `done`: the pipeline is finished, but the task stays `running` at **99 %** until the worker has uploaded the results
  and calls `/complete`. Users only see 100 % once the splat is actually on the server.

**Keeping the lease alive.** Some steps post no progress for minutes (VGGT model download, GLOMAP matching). The worker
sends a heartbeat every ~60 s from a background thread. If the reply has `"cancel": true`, or if any worker call returns
`409` (the lease was lost and the task was re-queued or taken by someone else), the worker stops the pipeline and
drops the task.

## Artifact upload (M3)

All results go back through **resumable chunked uploads**. Cloudflare limits one request to 100 MB, and the real
`scene.ply` is ~250 MB. A dropped connection must also not restart a large upload.

| Method | Path | Body / headers | Reply |
|---|---|---|---|
| POST | `/worker/tasks/{id}/uploads` | `{"worker_id", "kind", "size_bytes", "sha256", "chunk_size"?}` | upload object |
| GET | `/worker/uploads/{upload_id}` | – | upload object |
| PUT | `/worker/uploads/{upload_id}/chunks/{i}` | raw bytes; headers `X-Worker-Id`, optional `X-Chunk-Sha256` | `{"index", "received", "total_chunks", "duplicate"}` |
| POST | `/worker/uploads/{upload_id}/complete` | `{"worker_id"}` | upload object, `status: "complete"` |
| GET | `/worker/projects/{pid}/files/{path}` | – | a stored file (Week 2 rebuild inputs), Range supported |

Upload object: `{"upload_id", "kind", "path", "is_archive", "size_bytes", "chunk_size", "total_chunks",
"status": "open"|"complete", "received", "missing": [chunk indices], "meta"}`.

**Algorithm (worker)**
1. `POST …/uploads` with the file's size and sha256. The reply's `missing` lists the chunks to send. If the same content
   was started before (a crash, or a retry), the **same** upload comes back with only the missing chunks. If exactly this
   content is already stored for this task, `status` is `complete` and nothing needs sending.
2. `PUT` each missing chunk (any order). Repeating a chunk is harmless. Chunk `i` is bytes
   `[i*chunk_size, (i+1)*chunk_size)`. Default chunk size is 32 MB, maximum 90 MB.
3. `POST …/complete`. The server checks that every chunk is present (`409` + `missing` otherwise), then checks the
   **whole-file sha256**. On a mismatch it returns `422` and discards all chunks, so the worker uploads again. Then it
   validates the content (below). Only then is the upload `complete`.
4. After all uploads: `POST /worker/tasks/{id}/complete`. This returns `409 {"missing": [...]}` unless the required kinds are
   there. All of the task's artifacts are then moved into the project **together**. A failed or cancelled run never
   replaces a good scene.

**Artifact kinds.** Each kind has one fixed location in Person A's layout, so the worker can't write anywhere else:

| kind | stored at | upload as | required | validated |
|---|---|---|---|---|
| `frames` | `frames/` | tar | ✔ | ≥1 jpg/png in `frames/images/`, valid `frames_index.json` |
| `scene` | `scene/` | tar | ✔ | `sparse/0/{cameras,images,points3D}.{bin,txt}`, non-empty `images/` |
| `splat` | `export/scene.splat` | file | ✔ | size is a positive multiple of 32 |
| `cameras` | `export/cameras.json` | file | ✔ | JSON list with img_name, width, height, position, rotation, fx, fy, cx, cy |
| `scene_meta` | `export/scene_meta.json` | file | ✔ | JSON |
| `ply` | `export/scene.ply` | file | | PLY header |
| `poses` | `poses/` | tar | | ≥1 method folder (VGGT depth/conf for Week 2 live here) |
| `gs` | `gs/` | tar | | non-empty (checkpoints for the Week 2 fine-tune) |
| `metrics` | `metrics/` | tar | | non-empty |
| `manifest` | `manifest.json` | file | | JSON |

**Folder archives** are tar or tar.gz files whose entries all start with the folder name (`frames/images/000001.jpg`).
Rejected: absolute paths, `..`, entries outside the folder, devices/FIFOs, and links that point outside the project.
**Symlinks** (A's `scene/images` → `../poses/<m>/images` → `../../frames/images` on Kaggle) are turned into real copies.
Their targets must already be uploaded, so the **upload order is `frames`, `poses`, `scene`, then the rest**. Using
`tar -h` (dereference) on the worker also works.

## Verified so far
- 25 backend tests: life cycle, retries, lease expiry, cancellation, worker isolation, token checks, queue order.
- Live server + Person A's real `run_pipeline.py --only extract` (CPU, real ffmpeg) with `W1_WORKER_TOKEN` set:
  all 6 progress posts accepted, and the task showed `extracting frames 15 %`. Without the token: `401`, and
  the pipeline continued.
- M3: 52 backend tests (resume after interruption, duplicate/out-of-order chunks, chunk and whole-file checksum
  failures, missing chunks, malicious archives, per-kind validation, missing artifacts, access control, 64 MB Range
  downloads).
- M3 with the team's **real Kaggle output** (`demo_export.zip`) against a live server: 8 artifacts uploaded with
  32 MB chunks. The 249 MB `scene.ply` was interrupted after 3/8 chunks and resumed with only the 5 missing.
  Stored files are byte-identical to the originals. Interrupted download resumed with `curl -C -`, and the sha matches.
- M4: the real worker (`worker/kaggle_worker.py`) against a live server with `tests/stub_pipeline.py`. That stub runs
  Person A's real stage 1 (ffmpeg), stage 3 (choose → `scene/`), stage 5 (export) and make_reports, and fakes only the GPU
  steps. 8 end-to-end tests: success (A's layout on the backend, metrics in `result`, ply skipped by default), optional
  ply/checkpoint (splat uploaded before ply), failure → retry → failed (stage error kept as the headline),
  user cancel (pipeline **and its grandchild** killed), lease lost mid-run, bad token, idle exit, resumable video download.
- Not yet: GPU run on Kaggle through a Cloudflare tunnel. Steps and expected results: [KAGGLE_RUNBOOK.md](KAGGLE_RUNBOOK.md).

## Week 2 tasks (M8)
Types `segment`, `track`, `edit_preview`, `edit_apply`, `rebuild`. The claim reply then also carries:
```json
{"edit_id": "3f9c...", "stage": "run_edit",
 "stage_args": ["frames", "--edit-id", "3f9c...", "--prompt", "...", "--model", "klein", "--precision", "bf16", ...],
 "inputs":  [{"kind": "frames", "version": "original", "path": "frames", "is_dir": true, "required": true}, ...],
 "outputs": [{"kind": "edited", "version": "3f9c...", "path": "edits/3f9c.../edited", "is_dir": true, "required": true}, ...]}
```
The worker:
1. deletes local copies of the outputs (an earlier attempt's files must not be uploaded), then downloads every input
   with `GET /worker/projects/{pid}/artifacts/{kind}?version=` (folders come as a `.tar` named by project path) into
   `<root>/<project_id>/`; a missing optional input is a `404` and skipped;
2. runs `python <script> <stage_args> --project-dir <root>/<pid> --backend-url <url> --task-id <id>`, where `stage`
   `run_edit` = `backend/editing/run_edit.py` (Person B) and `w2_segment` / `w2_track` / `w2_rebuild` =
   `<ml-dir>/stages/<stage>.py` (Person A / C, `--stages-dir` to override). The script posts progress itself;
3. uploads the outputs with the normal chunked upload. **The backend takes the version from the task**, so an edit
   task can only write `edits/<edit_id>/...` and never the original reconstruction;
4. completes with `result` = stage seconds, upload MB/s and, for diffusion tasks, the `edit_report.json` summary
   (model, dtype option, s/frame, peak VRAM per GPU, outside-mask leak, colour drift).

Required outputs: `segment`/`track` → `masks`; `edit_preview` → `candidates` + `edit_report`; `edit_apply` →
`edited` + `edit_report`; `rebuild` → `splat` + `cameras` + `scene_meta` (+ optional `metrics`, `ply`).
`--types` lets one worker per GPU take different tasks (spec §3: GPU 0 diffusion, GPU 1 SAM + 3DGS), e.g.
`CUDA_VISIBLE_DEVICES=0 ... --types edit_preview,edit_apply` and `CUDA_VISIBLE_DEVICES=1 ... --types reconstruct,segment,track,rebuild`.

Tested (M8, `tests/test_edits.py`): API rules and the whole flow through the real worker process: click → refine →
track → remove + fallback-add frames → preview (Person B's real stage, fake model) → accept → rebuild → `?version=`,
with a check that **no original file changed**; a failing stage (retry, then its own error), a missing stage script
(clear error, no retry). Person A/C's stages were replaced by CPU stand-ins in `tests/stubs_w2/` that follow the
contract above.

## Stale uploads
Unfinished uploads are removed when their task has ended or after `BACKEND_UPLOAD_STALE_HOURS` (24 h) without a chunk,
together with their `.part` file. So are orphaned `.part` files and the staging folders of failed or cancelled tasks.
Cleanup runs at server start and at most every 10 minutes when a worker polls.

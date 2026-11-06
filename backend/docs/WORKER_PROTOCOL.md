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

## Verified so far (M2)
- 25 backend tests: life cycle, retries, lease expiry, cancellation, worker isolation, token checks, queue order.
- Live server + Person A's real `run_pipeline.py --only extract` (CPU, real ffmpeg) with `W1_WORKER_TOKEN` set:
  all 6 progress posts accepted, and the task showed `extracting frames 15 %`. Without the token: `401`, and
  the pipeline continued.
- Not yet: the worker loop itself, artifact upload, and a run on Kaggle through the tunnel (M3/M4).

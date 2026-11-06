# 3D-Room backend (Person B)

FastAPI + SQLite. Users, projects, video uploads, reconstruction/edit tasks, a pull-based Kaggle worker API,
artifact storage and scene retrieval. The API reference for the frontend is in [docs/API.md](docs/API.md);
interactive docs are at `/docs` while the server runs.

Docs: [API.md](docs/API.md) (frontend) · [WORKER_PROTOCOL.md](docs/WORKER_PROTOCOL.md) · [KAGGLE_RUNBOOK.md](docs/KAGGLE_RUNBOOK.md) · [NOTES_FOR_PERSON_A.md](docs/NOTES_FOR_PERSON_A.md)

```
Browser (Person C) ──HTTPS──┐
                            ▼
        Cloudflare Tunnel (https://<name>.trycloudflare.com)
                            ▼
        FastAPI on your machine  (localhost:8000) ── SQLite + backend/data/workspace/<project_id>/
                            ▲
Kaggle notebook worker ─────┘  polls for pending tasks, runs ML_Work/run_pipeline.py,
                               POSTs real progress, uploads artifacts back      (milestones M2–M4)
```

The Kaggle notebook never needs to be reachable. It only makes outgoing HTTPS requests to the tunnel URL,
so a dropped Kaggle session can't break the API. The spec warns about this in §4.3.

## Status

| Milestone | Content | State |
|---|---|---|
| M1 | auth (JWT), projects, video upload + validation, storage layout, data model | done, tested |
| M2 | reconstruction tasks, status, real progress, worker claim/lease/retry/cancel API, worker token | done, tested |
| M3 | resumable chunked artifact upload, validation, atomic promotion, scene endpoint, signed Range downloads | done, tested (incl. real Kaggle output) |
| M4 | Kaggle worker (`worker/kaggle_worker.py`) + its own notebook, stale-upload cleanup | done, tested locally with a CPU stub; **Kaggle GPU + tunnel run pending** ([runbook](docs/KAGGLE_RUNBOOK.md)) |
| M6–M9 | Week 2: diffusion editing, edit tasks, rebuild, before/after | planned |

## 1. Run locally

Requires Python ≥ 3.10 and `ffmpeg`/`ffprobe` on PATH (Windows: `winget install Gyan.FFmpeg`).

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate            # Windows (Git Bash: source .venv/Scripts/activate); Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q         # backend only: pip install -r requirements.txt
pip install -r requirements-dev.txt   # + Person A's CPU deps -> also runs the worker end-to-end tests
python -m pytest tests -q         # 61 passed (~2 min)
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Open http://localhost:8000/docs. Use **Authorize** with your email and password after signing up.

## 2. Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `BACKEND_DATA_DIR` | `backend/data` | DB, uploads, all project artifacts |
| `BACKEND_DATABASE_URL` | `sqlite:///<data>/app.db` | any SQLAlchemy URL (Postgres works too) |
| `BACKEND_JWT_SECRET` | random, persisted in `<data>/.jwt_secret` | token signing key (≥ 32 chars if you set it) |
| `BACKEND_JWT_EXPIRE_MINUTES` | `10080` (7 days) | token lifetime |
| `BACKEND_MAX_UPLOAD_MB` | `2048` | video size limit |
| `BACKEND_VIDEO_MIN_SECONDS` / `_MAX_SECONDS` | `5` / `600` | hard limits. 2–5 min (spec §2) only produces a warning, because short debug clips are normal |
| `BACKEND_FFPROBE` | `ffprobe` | path to ffprobe |
| `BACKEND_WORKER_TOKEN` | random, persisted in `<data>/.worker_token` | secret for `/worker/*` and `POST /tasks/{id}/progress` |
| `BACKEND_LEASE_SECONDS` | `300` | a running task with no progress/heartbeat for this long is re-queued |
| `BACKEND_MAX_ATTEMPTS` | `2` | tries per task before it is marked `failed` |
| `BACKEND_UPLOAD_CHUNK_MB` | `32` | default chunk size for worker uploads (max 90, below Cloudflare's 100 MB) |
| `BACKEND_UPLOAD_MAX_ARTIFACT_MB` | `20480` | largest single artifact / unpacked archive |
| `BACKEND_FILE_URL_TTL_MINUTES` | `360` | lifetime of the signed file URLs returned by `/scene` |
| `BACKEND_UPLOAD_STALE_HOURS` | `24` | unfinished uploads with no new chunk for this long are deleted |
| `BACKEND_CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173,...` | comma-separated frontend origins |

## 3. Storage layout

Each project's folder uses the same layout as Person A's pipeline (`ML_Work/common/config.py`), so the worker
runs `run_pipeline.py --root <data>/workspace --project <project_id>` and Week 2 reads the same paths:

```
backend/data/
  app.db                         users, projects, tasks, artifacts
  workspace/<project_id>/
    video/input.<ext>            the uploaded original (replaced only by a new upload)
    frames/images/000001.jpg …   frames/frames_index.json
    poses/<method>/…             scene/sparse/0/{cameras,images,points3D}.{bin,txt}  scene/images/  scene/points.ply
    gs/…  metrics/…  manifest.json
    export/scene.splat  export/scene.ply  export/cameras.json  export/scene_meta.json
  uploads/<upload_id>.part       chunks of uploads in progress
  staging/<task_id>/…            verified uploads of a running task; moved into workspace/ when the task completes
```
Names and folders are exactly the ones Person A's pipeline writes, and files are stored byte-for-byte. A's json files
contain absolute `/kaggle/working/...` paths; the API never returns them (the scene metadata is whitelisted).
`backend/data/` is git-ignored. Back it up if the demo data matters.

## 4. Worker token (Kaggle ↔ backend)

```bash
cat backend/data/.worker_token        # or set BACKEND_WORKER_TOKEN yourself before the first start
```
On Kaggle, store it as a Secret and export it as **`W1_WORKER_TOKEN`**. Person A's `ML_Work/common/progress.py`
sends it as `Authorization: Bearer …` with every progress post (it sends no header when the variable is unset). Protocol
details: [docs/WORKER_PROTOCOL.md](docs/WORKER_PROTOCOL.md).

## 5. The Kaggle worker

`worker/kaggle_worker.py` is a single file (Python ≥ 3.10 + `requests`) that runs on Kaggle. Its notebook is
[worker/kaggle_worker.ipynb](worker/kaggle_worker.ipynb) (generated by `worker/make_worker_notebook.py`; Person A's
notebooks are not touched). Loop: `/worker/ping` (token check) → claim → download the video (resumable, sha256) →
run `ML_Work/run_pipeline.py` unmodified as a child process (it posts real progress) → heartbeat thread (cancel or
lost lease stops the pipeline and all its stage processes) → upload artifacts (viewer files first) → complete or fail.

```bash
python worker/kaggle_worker.py --backend-url https://<tunnel> --ml-dir /kaggle/working/repo/ML_Work \
       --root /kaggle/working/workspace [--upload-ply] [--upload-checkpoint] [--once] [--max-idle-minutes 30]
```
Token: env `W1_WORKER_TOKEN`. First real run on Kaggle: follow [docs/KAGGLE_RUNBOOK.md](docs/KAGGLE_RUNBOOK.md).

## 6. Expose the backend with Cloudflare Tunnel

The Kaggle worker (and teammates' browsers) need a public HTTPS URL for your local server. Nothing is
port-forwarded: `cloudflared` makes an outgoing connection to Cloudflare.

### 6a. Quick tunnel (no account, URL changes on every start). Use this for development and the demo.

```powershell
winget install --id Cloudflare.cloudflared         # once; or download cloudflared.exe from Cloudflare's GitHub releases
uvicorn app.main:app --host 127.0.0.1 --port 8000  # terminal 1
cloudflared tunnel --url http://localhost:8000     # terminal 2
```
`cloudflared` prints a line like `https://random-words.trycloudflare.com`. That is the backend URL:
- test it: `curl https://random-words.trycloudflare.com/health`
- give it to the Kaggle worker (`BACKEND_URL`, M4) and to Person C's frontend (`VITE_API_URL` or similar)
- add the frontend's own origin to `BACKEND_CORS_ORIGINS` if it is served from somewhere other than localhost

The URL dies when `cloudflared` stops. Restarting gives a new URL, which you then update in the notebook.

### 6b. Named tunnel (stable URL). Needs a free Cloudflare account and a domain on Cloudflare.

```powershell
cloudflared tunnel login                                   # opens the browser, pick your domain
cloudflared tunnel create room3d                           # creates credentials in %USERPROFILE%\.cloudflared
cloudflared tunnel route dns room3d api.<your-domain>      # DNS record pointing at the tunnel
```
`%USERPROFILE%\.cloudflared\config.yml`:
```yaml
tunnel: room3d
credentials-file: C:\Users\<you>\.cloudflared\<tunnel-id>.json
ingress:
  - hostname: api.<your-domain>
    service: http://localhost:8000
  - service: http_status:404
```
Then `cloudflared tunnel run room3d`. The URL `https://api.<your-domain>` stays the same across restarts.

### Limits that shape the design
- **Request body ≤ 100 MB** through Cloudflare. Video uploads from the browser stay below the configured limit.
  Large artifacts from Kaggle (`scene.ply`, checkpoints, ~250 MB+) go up in chunks (M3).
- Quick tunnels have no uptime guarantee. That is fine here, because the worker retries and progress posts never kill
  a stage (`ML_Work/common/progress.py`).
- The tunnel makes the API public. User routes require a JWT, and worker routes will require a separate
  `BACKEND_WORKER_TOKEN`.

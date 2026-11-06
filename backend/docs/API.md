# 3D-Room API reference (for the frontend, Person C)

Base URL: the Cloudflare tunnel URL, or `http://localhost:8000` locally. Live, always up-to-date schema:
`GET /docs` (Swagger UI) and `GET /openapi.json`. You can generate a typed client from it, e.g.
`npx openapi-typescript <base>/openapi.json -o api.d.ts`.

## Conventions
- JSON everywhere except uploads (`multipart/form-data`).
- Auth: `Authorization: Bearer <access_token>` on every route except `/health`, `/auth/signup`, `/auth/login`, `/auth/token`.
  `401` means the token is missing, invalid or expired, so send the user to the login page.
- Errors: `{"detail": "<message>"}`. Validation errors (`422`) have `detail` as a list of `{loc, msg, type}`.
- IDs are 32-character hex strings. Timestamps are ISO-8601 UTC with a `Z` suffix.
- A project that belongs to another user returns `404`, the same as one that doesn't exist.

## Pages and the calls they need (spec §4.5)

| Page | Calls |
|---|---|
| Sign up / log in | `POST /auth/signup`, `POST /auth/login`; keep `access_token` (e.g. localStorage) |
| Projects list | `GET /projects`, `POST /projects` |
| Upload + submit | `POST /projects/{id}/video`, then `POST /projects/{id}/reconstruct` |
| Progress | poll `GET /tasks/{task_id}` every 2–3 s until `status` is `done`, `failed` or `cancelled`; optional `POST /tasks/{id}/cancel` |
| Viewer | `GET /projects/{id}/scene`, then load `splat.url` |

---

## Auth

### `POST /auth/signup` → 201
```json
{"email": "user@example.com", "password": "at least 8 chars"}
```
Response (same shape for login):
```json
{"access_token": "eyJ...", "token_type": "bearer", "expires_in": 604800,
 "user": {"id": "1f0c...", "email": "user@example.com", "created_at": "2026-11-06T01:41:55Z"}}
```
`409` if the email is taken, `422` for an invalid email or a password under 8 characters. Emails are stored in lower case.

### `POST /auth/login` → 200
Same body as signup. `401 {"detail": "Incorrect email or password"}`.

### `GET /auth/me` → 200
`{"id", "email", "created_at"}`. Use it on app start to check whether a stored token is still valid.

*(`POST /auth/token` is the OAuth2 form version used by the Swagger "Authorize" button. The frontend doesn't need it.)*

---

## Projects

### Project object
```json
{
  "id": "8f311bfe7f8744f5859ac9f1433ce517",
  "name": "Living room",
  "description": "",
  "has_video": true,
  "video": {
    "filename": "room.mp4", "size_bytes": 115633, "duration_s": 182.4,
    "width": 1920, "height": 1080, "fps": 30.0, "codec": "h264", "rotation": 0,
    "warnings": []
  },
  "latest_task_id": null,
  "latest_task_status": null,
  "created_at": "2026-11-06T01:41:55.388354Z",
  "updated_at": "2026-11-06T01:41:55.512000Z"
}
```
`video` is `null` until a video is uploaded. `latest_task_status` is one of `pending | running | done | failed | cancelled`
or `null`. The project list can use it as a status badge.

| Method | Path | Body | Success | Errors |
|---|---|---|---|---|
| `POST` | `/projects` | `{"name": "...", "description": "optional"}` | `201` project | `422` empty name |
| `GET` | `/projects` | – | `200` list of the current user's projects, newest first | |
| `GET` | `/projects/{id}` | – | `200` project | `404` |
| `PATCH` | `/projects/{id}` | `{"name"?, "description"?}` | `200` project | `404`, `422` |
| `DELETE` | `/projects/{id}` | – | `204` (also deletes all stored files) | `404`, `409` a task is pending/running |

### `POST /projects/{id}/video` → 201
`multipart/form-data` with a single field **`file`**. Accepted extensions: `.mp4 .mov .m4v .mkv .webm .avi`.
```js
const fd = new FormData(); fd.append("file", fileInput.files[0]);
await fetch(`${API}/projects/${id}/video`, {method: "POST", headers: {Authorization: `Bearer ${token}`}, body: fd});
```
For an upload progress bar use `XMLHttpRequest.upload.onprogress` (fetch has no upload progress).

Response: `{"project_id": "...", "video": {...VideoMeta}}`. Show `video.warnings` to the user. They are non-blocking
(e.g. `"duration 45s is outside the recommended 2-5 min"`).

| Status | When |
|---|---|
| `413` | larger than the server limit (default 2 GB; the Cloudflare tunnel limits a single request to 100 MB, so keep demo videos below that) |
| `415` | wrong extension |
| `422` | not a readable video, no video stream, shorter than 5 s or longer than 10 min |
| `409` | a task is running on this project |

Uploading again replaces the previous video.

### `GET /projects/{id}/video` → 200 / 206
The stored video, with HTTP Range support. Because it needs the auth header, fetch it as a blob
(`URL.createObjectURL`) rather than using a bare `<video src>`.

---

## Tasks (reconstruction and progress)

### `POST /projects/{id}/reconstruct` → 202
Submits the "3D reconstruction" task. The body is optional. `{}` or no body uses the pipeline's defaults.
```json
{"method": "vggt", "train_mode": "quick", "fps": "auto", "long_side": 960}
```
| Field | Values | Meaning |
|---|---|---|
| `method` | `vggt` \| `glomap` \| `both` | camera pose method (`both` = run the VGGT-vs-GLOMAP comparison too, slower) |
| `train_mode` | `quick` \| `full` | ~7k iterations (minutes) or ~30k (better quality, ~45 min on a T4) |
| `fps` | `auto` \| `"1"`…`"4"` | frame sampling |
| `long_side` | 640–960 | frame size in pixels |

The UI can simply send `{}`. A "high quality" checkbox could map to `train_mode: "full"`.
Errors: `409` no video uploaded yet, or the project already has a pending/running task. `422` invalid option.
Response: a Task object (below), `status: "pending"`.

### Task object (`GET /tasks/{id}` → 200)
```json
{
  "id": "68fbe00a36d24d1cb5ad34616ae5b56f",
  "project_id": "8f311bfe7f8744f5859ac9f1433ce517",
  "type": "reconstruct",
  "status": "running",
  "stage": "estimating poses",
  "progress": 30.0,
  "error": null,
  "params": {"train_mode": "quick"},
  "result": null,
  "attempts": 1,
  "max_attempts": 2,
  "queue_position": null,
  "stages": [
    {"name": "extracting frames", "start": 0,  "end": 15},
    {"name": "estimating poses",  "start": 15, "end": 45},
    {"name": "training",          "start": 45, "end": 90},
    {"name": "exporting",         "start": 90, "end": 100}
  ],
  "created_at": "...Z", "started_at": "...Z", "finished_at": null, "updated_at": "...Z"
}
```

**How to draw the progress page.** The progress is real: it comes from the pipeline stages, not from a timer (spec §4.4).
- `progress` is overall 0–100. `stages` lists the segments, so the bar can show four labelled sections with the
  current `stage` highlighted.
- `status` / `stage` combinations to handle:

| `status` | `stage` | Show |
|---|---|---|
| `pending` | `queued` | "Waiting for a GPU worker", plus `queue_position` (1 = next). Nothing moves until the Kaggle worker is running. |
| `pending` | `queued (retry)` | "Retrying" plus `error` (why the previous attempt failed) |
| `running` | one of the 4 stages | the bar |
| `running` | `uploading results` | the bar at 99 %: results are being copied to the server |
| `done` | `done` | 100 %, button "Open viewer" |
| `failed` | last stage reached | `error` (failed after `max_attempts` tries). The user can submit again. |
| `cancelled` | – | "Cancelled" |

- While `running`, `error` may already contain a message for an attempt that is about to be retried. Show it
  only when `status` is `failed` or the stage is `queued (retry)`.
- Progress can go **down** when a task is retried (a new attempt starts from 0). Don't animate it as an error.

### Other task routes
| Method | Path | Success | Errors |
|---|---|---|---|
| `GET` | `/projects/{id}/tasks` | `200` all tasks of the project, newest first | `404` |
| `POST` | `/tasks/{id}/cancel` | `200` task with `status: "cancelled"` | `404`, `409` already finished |

`ProjectOut.latest_task_id` / `latest_task_status` let the projects list link straight to the progress page.

*(`POST /tasks/{id}/progress` and `/worker/*` are for the Kaggle worker only and need the worker token. The frontend
must not call them. See [WORKER_PROTOCOL.md](WORKER_PROTOCOL.md).)*

---

## Scene (viewer page)

### `GET /projects/{id}/scene` → 200
Optional `?version=original` (the default). Week 2 adds edit versions for the before/after toggle.
```json
{
  "project_id": "…", "version": "original", "task_id": "…",
  "splat":  {"url": "/projects/…/files/export/scene.splat?t=eyJ…", "size_bytes": 33731776, "sha256": "…"},
  "ply":    {"url": "/projects/…/files/export/scene.ply?t=eyJ…",   "size_bytes": 248773326, "sha256": "…"},
  "points": {"url": "/projects/…/files/scene/points.ply?t=eyJ…",   "size_bytes": 158504, "sha256": null},
  "cameras_url": "/projects/…/files/export/cameras.json?t=eyJ…",
  "cameras": [{"id": 1, "img_name": "000001.jpg", "width": 960, "height": 540,
               "position": [0.1, -0.2, 0.3], "rotation": [[1,0,0],[0,1,0],[0,0,1]],
               "fx": 771.2, "fy": 771.2, "cx": 480.0, "cy": 270.0}],
  "meta": {"num_gaussians": 1054118, "bbox_min": [-2.9, -2.1, -2.4], "bbox_max": [2.8, 3.0, 2.1],
           "median_position": [-0.3, -0.1, -0.3], "trainer": "gsplat", "iters": 30000,
           "splat_mb": 32.17, "ply_mb": 237.25, "num_cameras": 61,
           "heldout": {"psnr": 25.88, "ssim": 0.877, "lpips": 0.077}, "pose_method": "glomap",
           "coordinate_note": "COLMAP/OpenCV world: +y is DOWN. …"},
  "up_vector": [0.0, -1.0, 0.0],
  "urls_expire_at": "2026-11-06T10:12:00Z"
}
```
The example values come from the team's real Kaggle run (61 frames, 30k iterations).

- **URLs are relative**: prefix them with the API base URL. They carry a signed `?t=` token, so they work **without an
  Authorization header**. That is how splat viewers, `<img>` and `<video>` load files. They expire after
  6 h (`urls_expire_at`). When they do, call `/scene` again.
- **Load `splat.url`** in the viewer (`.splat` format, 32 bytes per Gaussian). `ply` is the full-quality file (~250 MB),
  so don't load it by default.
- **Set the camera up-vector to `up_vector` (0, −1, 0)**, otherwise the room is upside down (COLMAP's y axis points down).
- **Fallback** if splat rendering fails (spec §4.5): render `points.url` (a coloured point cloud PLY) with three.js
  `PLYLoader`. `points` can be `null` if the pose method produced no point cloud.
- **`cameras`** are the training cameras. Use them for the Week 2 hooks *"load a specific camera pose"* and *"snap to a
  training camera"*. `rotation` is camera-to-world (3×3), `position` is the camera centre in world coordinates,
  and `fx fy cx cy` are pinhole intrinsics in pixels of a `width`×`height` frame.
- Files support HTTP **Range** (`206 Partial Content`), so large downloads can resume.

Errors: `404 {"detail": "Reconstruction still in progress"}` while a task is pending/running,
`404 {"detail": "No reconstructed scene for this project yet"}` otherwise.

### `GET /projects/{id}/artifacts` → 200
What is stored for the project (useful for a debug or details panel):
`[{"kind": "splat", "version": "original", "path": "export/scene.splat", "is_dir": false, "size_bytes": 33731776,
"sha256": "…", "task_id": "…", "created_at": "…Z", "info": {"num_gaussians": 1054118}}]`.
Kinds: `video, frames, poses, scene, gs, metrics, manifest, splat, ply, cameras, scene_meta`. `path` is the location
inside the project (Person A's layout), not a server path.

### `GET /projects/{id}/files/{path}`
Downloads one stored file, e.g. `files/frames/images/000012.jpg` (Week 2 frame grid). Auth: the signed `?t=` from
`/scene`, **or** the owner's `Authorization: Bearer`. Only files that belong to a stored artifact are served.
Anything else is `404`.

---

## Coming next
- Week 2: edit sessions, masks, seed previews, accept/regenerate, frame-x-of-N progress, rebuild, `?version=` for before/after

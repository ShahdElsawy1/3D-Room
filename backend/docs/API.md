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
| Viewer | `GET /projects/{id}/scene` *(M3)* |

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

## Coming next (shapes fixed when each milestone lands)
- `GET /projects/{id}/scene` → splat URL, cameras, metadata *(M3)*
- Week 2: edit sessions, masks, seed previews, accept/regenerate, frame-x-of-N progress, rebuild, `?version=` for before/after

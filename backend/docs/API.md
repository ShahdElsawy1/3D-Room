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
| Upload + submit | `POST /projects/{id}/video`, then `POST /projects/{id}/reconstruct` *(M2)* |
| Progress | poll `GET /tasks/{task_id}` every 2 s *(M2)* |
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

## Coming next (shapes fixed when each milestone lands)
- `POST /projects/{id}/reconstruct` → task; `GET /tasks/{id}` → `{status, stage, progress 0–100, error}`, with the stages
  `extracting frames → estimating poses → training → exporting` *(M2)*
- `GET /projects/{id}/scene` → splat URL, cameras, metadata *(M3)*
- Week 2: edit sessions, masks, seed previews, accept/regenerate, frame-x-of-N progress, rebuild, `?version=` for before/after

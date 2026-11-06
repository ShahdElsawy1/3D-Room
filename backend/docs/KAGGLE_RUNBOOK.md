# Runbook: first real run on Kaggle (GPU + Cloudflare tunnel)

Everything up to here has been tested locally with a CPU stub. The steps below are the part that **can only be
checked by a person with a Kaggle account**. Tick them off in order; each has what to expect and what to do if it
doesn't happen.

## 0. Prerequisites (once)
- [ ] `git push` the `backend` branch (the notebook clones it). If the repo is private, upload `3D-Room/` as a Kaggle
      Dataset instead and use the commented line in cell 1.
- [ ] Install cloudflared on your machine: `winget install --id Cloudflare.cloudflared`.
- [ ] Kaggle: phone-verified account (needed for GPU + Internet).

## 1. Start the backend and the tunnel (your machine)
```powershell
cd 3D-Room\backend
.venv\Scripts\activate
uvicorn app.main:app --host 127.0.0.1 --port 8000          # terminal 1
cloudflared tunnel --url http://localhost:8000              # terminal 2 -> prints https://<words>.trycloudflare.com
type data\.worker_token                                     # the worker token
```
- [ ] `curl https://<words>.trycloudflare.com/health` returns `{"status":"ok",...}` (from any network, e.g. your phone).

## 2. Kaggle notebook
- [ ] Import `backend/worker/kaggle_worker.ipynb` (File → Import notebook).
- [ ] Settings: **GPU T4 x2**, **Internet on**. Add-ons → Secrets: `W1_WORKER_TOKEN` = the token, attached to the notebook.
- [ ] Cells 1–4 (clone, A's setup ~10–20 min, env, environment fixes). Expect `python: /usr/local/bin/python` and two Tesla T4.
- [ ] Cell 5 with your tunnel URL: expect `200 {"ok":true,"lease_seconds":300}`.
      `401` = wrong token secret; timeout = tunnel not running or wrong URL.

## 3. Submit a task (your machine, or /docs in the browser)
Use a real 2–5 min room video, under 100 MB because the browser upload goes through the tunnel too.
```bash
B=https://<words>.trycloudflare.com
TOK=$(curl -s -X POST $B/auth/signup -H 'content-type: application/json' -d '{"email":"me@x.com","password":"password123"}' | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
PID=$(curl -s -X POST $B/projects -H "Authorization: Bearer $TOK" -H 'content-type: application/json' -d '{"name":"Demo"}' | python -c "import sys,json;print(json.load(sys.stdin)['id'])")
curl -s -X POST $B/projects/$PID/video -H "Authorization: Bearer $TOK" -F "file=@room.mp4"
curl -s -X POST $B/projects/$PID/reconstruct -H "Authorization: Bearer $TOK" -H 'content-type: application/json' -d '{"method":"vggt","train_mode":"quick"}'
```
`method: vggt` until Person A adds GLOMAP's `--skip_retriangulation` fix (see NOTES_FOR_PERSON_A.md).

## 4. Run the worker (cell 6) and watch
| Check | Expected | If not |
|---|---|---|
| - [ ] worker log | `connected to …`, `claimed task …`, `video downloaded` | `token rejected`: secret. Claim errors: tunnel |
| - [ ] `GET /tasks/{id}` every few seconds | `stage` moves *extracting frames → estimating poses → training → exporting*, `progress` rises | progress stuck at 0 while the log shows `[progress]` lines: `W1_WORKER_TOKEN` not set in the environment of cell 6 |
| - [ ] upload phase | `stage: "uploading results"`, 99 %; log lines `uploaded frames/splat/…` | `failed after 6 tries`: tunnel dropped for > ~2 min. The task re-queues (lease) and the next run resumes uploads |
| - [ ] done | `status: done`, `result.stages.*.seconds / peak_vram_mb`, `result.uploads.*.mb_per_s` | `error` field has the stage's message + log tail |
| - [ ] scene | `GET /projects/{id}/scene`; download `splat.url` in a browser | |

**Record for the report** (from `GET /tasks/{id}` → `result`): per-stage `seconds` and `peak_vram_mb`, upload MB and
MB/s through the tunnel, total time. These numbers have **not** been measured yet: Kaggle → Cloudflare throughput
is unknown, and local tests run at ~40 MB/s.

## 5. Failure drills (10 minutes, worth doing once)
- [ ] **Cancel**: submit a second task, then `POST /tasks/{id}/cancel` during training. Expect the worker log to show
      `stopping pipeline: cancelled by user` within ~60 s, the task `cancelled`, and `nvidia-smi` showing no python
      process left on the GPU.
- [ ] **Session death**: submit, then stop the Kaggle session (Power off) during training. After 5 min the task goes
      back to `pending` with `stopped responding … re-queued`. Start the notebook again and the task is picked up
      as attempt 2.
- [ ] **Tunnel restart**: restart `cloudflared` during the upload phase (new URL). The worker cannot follow a new
      URL: the task re-queues after the lease expires. Re-run cells 5–6 with the new URL. Uploads of the new attempt
      start fresh (a new attempt reruns the pipeline). A **named tunnel** (stable URL) avoids this. See README §5b.

## Known gaps to confirm on Kaggle
- GPU kill: on Linux the worker kills the pipeline's whole process group (`start_new_session`). Locally it was
  tested on Windows (`taskkill /T`), where the stage grandchild did die. Confirm with `nvidia-smi` in the cancel drill.
- `scene/images` and `poses/*/images` are symlinks on Kaggle. The worker tars with `dereference=True`, which was tested
  locally with A's real stage 3. Windows fell back to copying, so the symlink branch itself only runs on Kaggle.

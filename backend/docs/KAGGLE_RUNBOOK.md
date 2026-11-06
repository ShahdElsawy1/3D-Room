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

## 6. Week 2 (M7): diffusion benchmark on the T4s
Tested locally only with a fake model. These steps measure the real model and fill deliverable 4 (VRAM and timing
table). Do them in the same session as the Week 1 run: start the worker with `--keep-workspace` so the frames stay in
`/kaggle/working/workspace/<project_id>/`. ~1–2 h of GPU quota (download included).

Prerequisites: accept the FLUX 3 Image Klein license on huggingface.co with your account, create a read token, add it
as Kaggle Secret **`HF_TOKEN`**. Keep the HF cache out of `/kaggle/working` (20 GB output limit); the default
`~/.cache/huggingface` is fine.

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
E = "/kaggle/working/repo/backend/editing"
P = "/kaggle/working/workspace/<project_id>"          # <- the project of the Week 1 run
!pip install -q -r $E/requirements-kaggle.txt
!python $E/bench_t4.py --out /kaggle/working/bench_t4.json     # ~2 min: options A/B/C/D, linear + attention
!ls $P/frames/images | head -40                                # pick 3 consecutive frames with a big object
```
Until Person A's masks exist, draw a box around the object (look at the frame with `PIL.Image.open(...)`):
```python
!python $E/make_box_masks.py --project-dir $P --edit-id bench --box 300,180,620,420 \
        --frames frame_000010.jpg,frame_000011.jpg,frame_000012.jpg
R = f"python {E}/run_edit.py"
A = f"--project-dir {P} --edit-id bench --prompt 'make the sofa dark green leather'"
!{R} preview {A} --model klein --precision bf16      --seeds 0,1,2    # option A
!{R} preview {A} --model klein --precision bf16-fp32 --seeds 0        # option B
!{R} preview {A} --model klein --precision fp8-bf16  --seeds 0        # option D
!{R} frames  {A} --model klein --precision bf16 --chosen-seed 0       # reference-conditioned frames
!cat {P}/edits/bench/edit_report.json
```
| Check | Expected | If not |
|---|---|---|
| bench_t4 rows | A and B slow, C fast, `nan_or_inf` false for A/B/D | an `error` in a row = that dtype path has no kernel on Turing: note it, it rules the option out |
| preview | `candidates/seed_*.png`, report `seconds`, `peak_vram_mb` per GPU | OOM with `split` → `--placement offload`; error mentioning `.transformer` or layerwise casting → option B/D unsupported for this pipeline, note it |
| frames | `edited/*.png`, `changed_outside_px: 0` for every frame | |
| `degenerate: true` | never for A/B/D | black/flat output = NaNs: report it |

Option C (fp16) only after the team decides it: add `--precision fp16 --allow-fp16`. Qwen comparison: same
commands with `--model qwen-turbo` (bf16 does not fit one T4, so try `--placement offload` and `--precision fp8-bf16`).
Save `bench_t4.json` and `edit_report.json` (and a few `candidates/` + `edited/` PNGs) and send them to Person B.

## 7. Report tables (M9)
After the runs, on your machine (backend running):
```powershell
python tools/make_report.py --backend-url http://localhost:8000 --email <you> --password <pw> --project <project_id> `
       --bench bench_t4.json --edit-report edit_report.json --out docs/MEASUREMENTS.md
```
It prints Week 1 runtime/VRAM per stage, tunnel upload MB/s, stored sizes, the diffusion table (model, dtype option,
s/frame, peak VRAM per GPU, NaN check, outside-mask leak, colour drift), the T4 benchmark and rebuild times. Cells
without a measurement say "not measured"; nothing is estimated.

# Measurements

## Week 1: runtime and VRAM per stage

| stage | seconds | peak VRAM (MB) | details |
|---|---|---|---|
| 1 extract frames | 139.2 | 3.0 | frames_kept=61 |
| 2a poses: VGGT | 97.0 | 10,333.0 | frames=61, registered_frames=61, points3D=300,000, chunks=3, model_load_s=46.81 |
| 4 train 3DGS | 1,061.1 | 3,105.0 | iters=7,000, num_gaussians=1,346,044 |
| 5 export splat | 10.3 | 3.0 | num_gaussians=1,346,044, ply_mb=302.95, splat_mb=41.08 |
| s6_prepare_week2 | 7.2 | 595.0 |  |
| **total** | 1,315.0 | 10,333.0 |  |


Pipeline 4,975.9 s, upload 1,353.9 s (worker `4926f4178028`).


## Week 1: upload Kaggle → backend (through the tunnel)

| artifact | MB | seconds | MB/s |
|---|---|---|---|
| frames | 5.3 | 23.5 | 0.2 |
| splat | 41.1 | 127.8 | 0.3 |
| cameras | 0.0 | 2.8 | 0.0 |
| scene_meta | 0.0 | 1.8 | 0.0 |
| scene | 39.3 | 68.6 | 0.6 |
| poses | 69.7 | 147.2 | 0.5 |
| metrics | 1.5 | 3.1 | 0.5 |
| manifest | 0.0 | 1.1 | 0.0 |
| gs | 1,000.2 | 978.0 | 1.0 |
| **total** | 1,157.1 | 1,353.9 | 0.8 |


## Stored artifact sizes

| artifact | version | MB on the backend |
|---|---|---|
| cameras | original | 0.0 |
| frames | original | 5.2 |
| gs | original | 1,000.1 |
| manifest | original | 0.0 |
| metrics | original | 1.5 |
| poses | original | 69.5 |
| scene | original | 39.1 |
| scene_meta | original | 0.0 |
| splat | original | 41.1 |
| video | original | 389.6 |
| cameras | 737d0a1c69d74da8b545c40828ba5c58 | 0.0 |
| candidates | 737d0a1c69d74da8b545c40828ba5c58 | 2.0 |
| edit_report | 737d0a1c69d74da8b545c40828ba5c58 | 0.0 |
| edited | 737d0a1c69d74da8b545c40828ba5c58 | 35.1 |
| masks | 737d0a1c69d74da8b545c40828ba5c58 | 0.1 |
| metrics | 737d0a1c69d74da8b545c40828ba5c58 | 1.2 |
| scene_meta | 737d0a1c69d74da8b545c40828ba5c58 | 0.0 |
| splat | 737d0a1c69d74da8b545c40828ba5c58 | 41.2 |


## Week 2: diffusion editing (per run)

| run | model | option: dtype | placement | crop px | load s | frames | s / frame | peak VRAM GPU0 (MB) | GPU1 (MB) | NaN/flat frames | raw leak PSNR (dB) | ΔE vs reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edit_apply 39c132 | klein | A: bf16 | split | 512 | 196.3 | 59 | 114.5 | 12,165 | 8,570 | 0 | 27.9 | 11.3 |
| edit_preview a2f6b1 | klein | A: bf16 | split | 768 | 213.1 | 3 | 189.4 | 13,159 | 8,570 | 0 | 34.3 | not measured |


## Week 2: T4 precision micro-benchmark

_not measured_


## Week 2: rebuild time (poses reused)

| task | mode | stage seconds | total seconds |
|---|---|---|---|
| afa04d | scratch | 1,047.3 | 1,567.6 |

# VRAM and runtime per stage

| stage | wall time (s) | peak VRAM, nvidia-smi (MiB) | peak VRAM, torch alloc (MiB) | notes |
|---|---|---|---|---|
| 1. Frame extraction (ffmpeg + blur filter) | 123.7 | 3 | 0 | frames_kept=61 |
| 2a. Poses: VGGT (chunked, fp16) | 118.2 | 11363 | 9595 | registered_frames=61, chunks=3, dtype=torch.float16 |
| 2b. Poses: GLOMAP (features + seq. matching + mapper) | 37.6 | 543 | 0 | registered_frames=61 |
| 4. 3DGS training | 2465.7 | 3653 | 0 | num_gaussians=1488141, iters=30000 |
| 5. Export (.ply -> .splat) | 2.9 | 3 | 0 | num_gaussians=1059412, splat_mb=32.33 |

## VGGT peak VRAM vs frames per chunk

dtype `torch.float16`, input [518, 294]

| frames | seconds | peak alloc (MiB) |
|---|---|---|
| 8 | 7.05 | 8499.4 |
| 16 | 3.46 | 8726.2 |
| 32 | 9.2 | 9162.3 |
| 48 | 17.59 | 9594.6 |

## 3DGS run

- trainer: gsplat, iterations: 30000
- training wall time: 2463.7 s
- final #Gaussians: 1488141
- trainer-reported peak memory: 2.4676756858825684 GB
- held-out metrics: {'psnr': 26.776355743408203, 'ssim': 0.8870153427124023, 'lpips': 0.06398005038499832}

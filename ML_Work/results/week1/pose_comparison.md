# VGGT vs GLOMAP

Frames fed to both methods: **61**

| method | registered | runtime (s) | peak VRAM (MiB) | points3D | path length |
|---|---|---|---|---|---|
| vggt | 61 | 118.23 | 11363.0 | 300,000 | 3.212 |
| glomap | 61 | 37.61 | 543.0 | 11,286 | 33.608 |

Trajectory agreement (Sim3-aligned, 61 common frames): RMSE = 0.70% of scene extent.


## Draft justification (edit before submitting)

We chose GLOMAP for the project because both register all frames and runtimes are comparable, so the more accurate classical baseline wins. VGGT: 61 frames, 118s, 11363 MiB peak VRAM. GLOMAP: 61 frames, 38s, 543 MiB peak VRAM.
[compare] scene/ <- glomap

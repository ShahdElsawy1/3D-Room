| setting | model | quantization | dtype | offload | steps | load time | s/frame (mean) | s/frame (median) | peak VRAM (nvidia-smi, GPU delta) | peak VRAM generating (torch) | VRAM after load (torch) | frames | max pixel change outside the mask |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4bit_bfloat16_model_s40 | Qwen/Qwen-Image-2.1 | 4bit | bfloat16 | model | 40 | 300 s | 427.2 s | 427.2 s | 12041 MiB | 11554 MiB | 10261 MiB | None | 0 |

# Week 2 pipeline (Person A) - click -> mask -> track -> rebuild -> hand-off

```
 Notebook 1 (Week 1, once, ~1 h GPU)                 Notebook 2 (Week 2, interactive)
 video -> frames -> poses -> splat ----zip----> unpack -> self-test -> click -> mask -> track -> (B edits) -> rebuild
        demo_week2_inputs.zip                                                        |                          |
        = frames, scene, export, week2/, checkpoint                      tracking report (IoU table)   finetune vs scratch table
                                                                                          \_____________________/
                                                                                       tools/make_handoff.py -> handoff_<project>.zip
```
Why two notebooks: Week 1 is done once and costs the most; Week 2 is trial-and-error on objects/clicks and uses different models
(SAM 2.1). The saved `week2/inputs.json` is the only link. B's worker runs the same three stage scripts as tasks.

## Stages (CLI = the worker's: `--project-dir <root>/<id> --edit-id E [--backend-url --task-id]`)
| stage | does | models |
|---|---|---|
| `stages/s6_prepare_week2.py` (end of Week 1) | `week2/inputs.json`, sparse depth per frame, depth rendered from the splat | gsplat (GPU) |
| `stages/w2_segment.py` | clicks (+/-) -> mask on one frame (`--frame/--clicks`), or fallback marking (`--add-frames`) | SAM 2.1 image |
| `stages/w2_track.py` | `sam2` video propagation, `lift` (3D reprojection + visibility + SAM edge refine), `both` = fused | SAM 2.1 video + image |
| `stages/w2_rebuild.py` | `finetune` (from the checkpoint, fixed poses) or `scratch` (stage 4 on the edited scene) -> `edits/E/export/` | gsplat |
| `tools/w2_eval_tracking.py` | IoU / presence accuracy of each tracker on hand-labelled frames | SAM 2.1 (GT from clicks) |
| `tools/w2_make_test_edit.py` | fake edit (recolour) to test the rebuild without B | - |
| `tools/w2_preview.py` | contact sheet of mask overlays | - |
| `tools/w2_selftest.py` | 2-minute GPU check of every model/API used above | all |
| `tools/make_handoff.py` | zip for B and C + generated `SUMMARY.md`, `README_FOR_B.md`, `README_FOR_C.md` | - |

## How tracking decides (`both`)
1. SAM 2 propagates the click mask (and any user-marked frames) forwards and backwards.
2. The click mask + the splat's depth give the object's 3D points; every frame's camera projects them; points BEHIND the splat surface
   seen there are hidden (occlusion); the visible share says whether the object is in the frame at all.
3. Per frame: SAM 2 mask accepted when it overlaps the geometric prediction (IoU >= 0.3) or no geometry exists; otherwise the lifted mask
   (edges refined by SAM 2 from box + points) is used; if geometry says "not visible" the frame is `present: false`.
   Thresholds (`--min-conf 0.6 --min-visible 0.15 --iou-agree 0.3`) are arguments; `metrics/track_debug.json` lists each frame's decision.
Not implemented (PDF: technique guideline only): DINOv2 re-identification, splat-based masks.

## Rebuild
Poses and intrinsics are reused. `scene/images` of the rebuild = symlinks to the originals + edited frames as JPEG q100/4:4:4 (gsplat reads by
file name; the change outside the mask is measured and reported). `finetune` keeps the trained Gaussians and trains ~3k steps at 0.25x
learning rates with densification, no opacity reset; `scratch` is stage 4 again. Both write `edits/E/rebuilds/<mode>/report.json`
(time, VRAM, #Gaussians, held-out PSNR/SSIM); `edits/E/export/` holds the last one run. The original `scene/ gs/ export/` are never modified.

## What was verified where
* CPU unit tests (47): mask contract, 3D lift / occlusion / out-of-view, frames<->scene undistortion, tracking decisions with injected
  models, rebuild scene building + wiring with a faked trainer, evaluation metrics, hand-off zip, pycolmap backend end to end.
* NOT run without a GPU (first Kaggle run decides): SAM 2 calls (`common/sam_wrap.py`), the fine-tune loop (`common/gs_finetune.py`, checked
  against gsplat 1.5.3's source), the stage scripts' model parts. `python tools/w2_selftest.py` runs all of them in ~2 minutes and prints the
  exact failing call if any. Known fallbacks: NaN masks -> `--sam-dtype float32`; no `fused_ssim` -> L1-only loss; no depth -> tracking
  uses SAM 2 only.

## Order of work
Notebook 1 -> download `demo_week2_inputs.zip` -> add it as input of Notebook 2 -> self-test -> click -> track -> evaluate -> (B's edited frames or
the fake edit) -> rebuild both modes -> `make_handoff` -> send the zip to B and C.

## Diffusion edit (`stages/w2_edit.py`, added later; nothing above changed)
Reads `edits/E/masks`, writes `edits/E/edited/<stem>.png` + `edited/index.json` - the format `w2_rebuild.py` already reads.
```
 --phase reference : seeds on the frame with the largest uncut mask  -> edits/E/preview_reference.png   (pick a seed)
 --phase rest      : every other present frame, its own mask, the same prompt, the chosen reference edit as image 2
 --phase all       : both in one run (reference seed = --ref-seed, default the first)
```
Model: Qwen-Image-2.1 through diffusers (`QwenImage21Pipeline`, one `image=[frame, reference_edit]` call; the pipeline has no mask
argument). The mask is used twice: a red outline ring painted just outside it on the input (`--hint outline|none`), and the compositing
that keeps every pixel outside the mask grown by `--grow` px (default 2) bit-identical to the original (proved per run in
`metrics/w2_edit.json -> outside_grown_mask_max_abs_diff`; `--grow 0 --feather 0` = the exact mask). Frames are reflect-padded to a
multiple of 16 for the model and cropped back; outputs have the size of `frames/images`.
Memory: `--quant 4bit|8bit|none` (bitsandbytes on transformer + text encoder), `--offload none|model|sequential`, `--dtype bfloat16`
(default, as the brief says), `--gpu 0`. Every run writes `metrics/w2_edit_<tag>.json`; `tools/w2_edit_table.py` collects them into
the VRAM / seconds-per-frame / quantization table. `--stride N` edits every Nth frame only (keyframes), `--resume` continues a run.
Install: `bash scripts/setup_kaggle.sh diffusion` (transformers>=5.17, diffusers, bitsandbytes). License: Qwen Research (non-commercial).
Verified on CPU (15 tests, fake model): sizes, the outside-the-mask guarantee read back from disk, reference choice, stage wiring,
resume/stride, hand-over to `w2_rebuild`. NOT run without a GPU: the model itself (`common/diffusion_edit.py -> QwenEditor`) -
smoke test: `python stages/w2_edit.py --project-dir P --edit-id E --prompt "..." --n-seeds 1 --limit 2`.
Not implemented: FLUX 3 Klein (same `Editor.edit(images, prompt, seed, size_hw)` interface, so a second class is all it needs).

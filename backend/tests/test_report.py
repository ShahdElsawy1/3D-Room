"""M9: report tables are built only from measured numbers; missing ones say so."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import make_report  # noqa: E402


def test_tables_from_files(tmp_path, capsys):
    m = tmp_path / "metrics"
    m.mkdir()
    (m / "s1_extract_frames.json").write_text(json.dumps({"seconds": 132.19, "peak_vram_mb": 3.0, "ok": True,
                                                          "frames_kept": 61, "stage": "/kaggle/x"}))
    (m / "s4_train_3dgs.json").write_text(json.dumps({"seconds": 2550.5, "peak_vram_mb": 3609.0, "iters": 30000}))
    bench = tmp_path / "b.json"
    bench.write_text(json.dumps({"gpu": "Tesla T4", "capability": "7.5", "torch": "2.x", "rows": [
        {"option": "A", "path": "bf16", "linear_ms": 40.0, "linear_tflops": 7.1, "attention_ms": 90.0,
         "attention_tflops": 5.0, "nan_or_inf": False, "error": None}], "overflow_probe_fp16": "inf"}))
    rep = tmp_path / "e1" / "edit_report.json"
    rep.parent.mkdir()
    rep.write_text(json.dumps({"frames": {"model": "klein", "option": "A", "precision": "bf16", "placement": "split",
                                          "long_side": 768, "load_seconds": 95.0,
                                          "summary": {"frames": 40, "seconds_per_frame": 61.2,
                                                      "peak_vram_mb": {"cuda:0": 13900, "cuda:1": 9100},
                                                      "degenerate_frames": 0, "mean_raw_leak_psnr": 31.5,
                                                      "mean_delta_e_vs_reference": 3.2}}}))
    task = tmp_path / "t.json"
    task.write_text(json.dumps({"id": "abcdef12", "type": "reconstruct", "status": "done", "params": {},
                                "result": {"pipeline_seconds": 2700, "upload_seconds": 30, "host": "kaggle",
                                           "uploads": {"splat": {"mb": 32.2, "seconds": 8.0, "mb_per_s": 4.0}}}}))
    assert make_report.main(["--metrics-dir", str(m), "--bench", str(bench), "--edit-report", str(rep),
                             "--task-json", str(task)]) == 0
    md = capsys.readouterr().out
    assert "| 1 extract frames | 132.2 | 3.0 | frames_kept=61 |" in md
    assert "| **total** | 2,682.7 | 3,609.0 |  |" in md
    assert "/kaggle" not in md                                   # only numbers are taken from A's files
    assert "| splat | 32.2 | 8.0 | 4.0 |" in md
    assert "| e1/frames | klein | A: bf16 | split | 768 | 95.0 | 40 | 61.2 | 13,900 | 9,100 | 0 | 31.5 | 3.2 |" in md
    assert "fp16=inf" in md and "| A | bf16 | 40.0 | 7.1 |" in md
    assert md.count("_not measured_") == 1                       # rebuild: nothing measured -> says so


def test_nothing_measured(capsys):
    assert make_report.main([]) == 0
    md = capsys.readouterr().out
    assert md.count("_not measured_") == 4

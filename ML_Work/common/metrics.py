"""Runtime + peak-VRAM measurement for every stage.

Stages such as GLOMAP and the gsplat trainer run as *subprocesses*, so
torch.cuda.max_memory_allocated() inside our process cannot see them. We therefore
sample nvidia-smi from a background thread (process-agnostic, whole-GPU) and, when
torch is importable and CUDA is in use, also record torch's own peak numbers.
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

from .config import write_json


def _smi_used_mb():
    """Per-GPU used memory in MiB, or [] when nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return []
        return [float(x) for x in out.stdout.strip().splitlines() if x.strip()]
    except Exception:
        return []


class GpuSampler(threading.Thread):
    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_evt = threading.Event()
        self.baseline = _smi_used_mb()
        self.peak = list(self.baseline)

    def run(self):
        while not self._stop_evt.is_set():
            cur = _smi_used_mb()
            for i, v in enumerate(cur):
                if i >= len(self.peak):
                    self.peak.append(v)
                else:
                    self.peak[i] = max(self.peak[i], v)
            time.sleep(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2)


class StageMetrics:
    """with StageMetrics(cfg.metrics_dir, "extract_frames") as m: ...; m.extra[...] = ..."""

    def __init__(self, metrics_dir, stage: str):
        self.path = Path(metrics_dir) / f"{stage}.json"
        self.stage = stage
        self.extra: dict = {}
        self._sampler = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._sampler = GpuSampler()
        self._sampler.start()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        seconds = time.perf_counter() - self._t0
        self._sampler.stop()
        base, peak = self._sampler.baseline, self._sampler.peak
        rec = {
            "stage": self.stage,
            "seconds": round(seconds, 2),
            "ok": exc_type is None,
            "gpu_baseline_mb": base,
            "gpu_peak_mb": peak,
            # VRAM attributable to this stage = peak - baseline (per GPU, max over GPUs)
            "peak_vram_mb": round(max([p - b for p, b in zip(peak, base)] or [0.0]), 1),
        }
        try:
            import torch
            if torch.cuda.is_available():
                rec["torch_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
                rec["torch_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        except Exception:
            pass
        rec.update(self.extra)
        if exc_type is not None:
            rec["error"] = f"{exc_type.__name__}: {exc}"
        write_json(self.path, rec)
        return False  # never swallow exceptions

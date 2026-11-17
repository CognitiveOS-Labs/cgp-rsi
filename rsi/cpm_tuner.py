"""cpm tune integration (Controller Pattern).

`cpm tune <name>` reads the `brain.training` block of an installed patch's
manifest and invokes the designated MCP tool through the system daemon. On
the GPU training host this module shells out to cpm; in unit tests a mock is
injected instead.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CpmTuneResult:
    loss: float | None
    lora_weight_norm: float | None
    adapter_dir: str
    latency_ms: float | None
    raw: str


class CpmTuner:
    """Production tuner: delegates to `cpm tune <package>`."""

    def __init__(self, cpm_bin: str = "cpm", patch_name: str = "base-model"):
        self.cpm_bin = cpm_bin
        self.patch_name = patch_name

    def tune(self, base_pkg: str, samples_file: Path, cfg) -> dict[str, Any]:
        import time

        start = time.monotonic()
        env = dict(os.environ)
        env["RSI_SAMPLES_FILE"] = str(samples_file)
        cmd = [
            self.cpm_bin,
            "tune",
            self.patch_name,
            "--epochs",
            str(cfg.epochs),
            "--yes",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        raw = proc.stdout + proc.stderr
        latency_ms = (time.monotonic() - start) * 1000.0
        if proc.returncode != 0:
            raise RuntimeError(f"cpm tune failed (exit {proc.returncode}):\n{raw[-2000:]}")
        metrics_path = Path(os.environ.get("RSI_TUNER_METRICS", "runs/last-tune.json"))
        loss = norm = None
        if metrics_path.exists():
            try:
                m = json.loads(metrics_path.read_text())
                loss = m.get("loss")
                norm = m.get("lora_weight_norm")
            except json.JSONDecodeError:
                pass
        return {
            "adapter_dir": os.environ.get("RSI_ADAPTER_DIR", "adapters"),
            "loss": loss,
            "lora_weight_norm": norm,
            "latency_ms": latency_ms,
            "raw": raw[-4000:],
        }

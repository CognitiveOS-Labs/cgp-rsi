"""Configuration for the RSI controller.

All knobs are plain dataclasses so runs can be serialized to YAML/JSON for
reproducibility. Defaults follow the research protocol in the research repo
(PROMOTION_EPSILON, PROMOTION_DELTA, REPLAY_ANCHOR_FRACTION=0.2).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class GateConfig:
    """Zero-trust promotion gate thresholds."""

    # Minimum margin by which M_{t+1} must beat M_t on the target benchmark.
    epsilon: float = 0.02
    # Maximum allowed regression on the general benchmark.
    delta: float = 0.01


@dataclass
class TuneConfig:
    """LoRA hyperparameters for `cpm tune` (passed through to the controller)."""

    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 2e-4
    epochs: int = 2
    min_samples: int = 50


@dataclass
class RsiConfig:
    """Top-level pipeline configuration."""

    generations: int = 6  # t = 0..5 -> M_0 ... M_5
    replay_anchor_fraction: float = 0.2
    target_domain: str = "python-algorithmic-synthesis"
    gate: GateConfig = field(default_factory=GateConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)
    promotion_epsilon: float = 0.02  # mirrored from gate for the metrics header
    metrics_path: Path = Path("runs/metrics.jsonl")
    artifacts_root: Path = Path("runs")
    seed: int = 1337

    def to_json(self) -> str:
        d = asdict(self)
        d["metrics_path"] = str(self.metrics_path)
        d["artifacts_root"] = str(self.artifacts_root)
        return json.dumps(d, indent=2, sort_keys=True)


def load_config(path: str | Path) -> RsiConfig:
    raw = json.loads(Path(path).read_text())
    gate = GateConfig(**raw.pop("gate", {}))
    tune = TuneConfig(**raw.pop("tune", {}))
    raw["metrics_path"] = Path(raw.get("metrics_path", "runs/metrics.jsonl"))
    raw["artifacts_root"] = Path(raw.get("artifacts_root", "runs"))
    return RsiConfig(gate=gate, tune=tune, **raw)

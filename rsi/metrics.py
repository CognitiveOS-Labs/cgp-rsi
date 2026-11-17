"""Per-generation metrics logging (JSONL) and summary rendering."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class GenerationMetrics:
    generation: int
    timestamp: float
    synthetic_pass_rate: float
    samples_generated: int
    samples_verified: int
    train_loss: float | None
    lora_weight_norm: float | None
    b_target_new: float
    b_target_base: float
    b_general_new: float
    b_general_base: float
    target_margin: float
    general_delta: float
    promoted: bool
    failure_mode: str | None
    inference_latency_ms: float | None
    rollback: bool = False


class MetricsLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, m: GenerationMetrics) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(m), sort_keys=True) + "\n")

    def read_all(self) -> list[GenerationMetrics]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            out.append(GenerationMetrics(**raw))
        return out


def render_table(rows: list[GenerationMetrics]) -> str:
    header = (
        "| gen | pass% | loss | \\|dW\\| | B_target | d(target) | B_general | d(general) | promoted |\n"
        "|-----|-------|------|---------|----------|-----------|-----------|------------|----------|"
    )
    lines = [header]
    for m in rows:
        loss = f"{m.train_loss:.4f}" if m.train_loss is not None else "—"
        norm = f"{m.lora_weight_norm:.4f}" if m.lora_weight_norm is not None else "—"
        lines.append(
            f"| {m.generation} | {100 * m.synthetic_pass_rate:.1f} | {loss} | {norm} | "
            f"{m.b_target_new:.4f} | {m.target_margin:+.4f} | {m.b_general_new:.4f} | "
            f"{m.general_delta:+.4f} | {'yes' if m.promoted else 'NO (rollback)'} |"
        )
    return "\n".join(lines)


def now() -> float:
    return time.time()

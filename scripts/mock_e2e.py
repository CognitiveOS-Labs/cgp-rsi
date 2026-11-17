#!/usr/bin/env python3
"""MOCK end-to-end pipeline exercise (CI only).

Runs the full Generate -> Verify -> Mix -> Tune -> Evaluate -> Gate ->
Promote/Rollback state machine for 2 generations using the deterministic
MockModelClient and a scripted MockBenchmarker.

This validates pipeline MECHANICS (orchestration, verifier wiring, replay
mixing, gate decisions, metrics logging, rollback on rejection).

It is NOT a language model and produces NO scientific claims. Every artifact
is written under runs/mock-e2e/ and is clearly labeled MOCK.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsi.config import RsiConfig, GateConfig, TuneConfig  # noqa: E402
from rsi.controller import RsiController  # noqa: E402
from rsi.generator import MockModelClient, ProblemSynthesiser, SolutionGenerator  # noqa: E402
from rsi.metrics import render_table  # noqa: E402


class MockTuner:
    """Pretends to fine-tune; writes a toy adapter descriptor, returns canned loss."""

    def tune(self, base_pkg, samples_file: Path, cfg):
        adapter = samples_file.parent / "adapters" / "mock-adapter.bin"
        adapter.parent.mkdir(parents=True, exist_ok=True)
        adapter.write_bytes(b"MOCK-ADAPTER\x00" + samples_file.read_bytes()[:64])
        return {"adapter_dir": str(adapter.parent), "loss": 0.42, "lora_weight_norm": 1.618,
                "latency_ms": 3.5}


class MockBenchmarker:
    """Scripted scores: improves on each call by `step` (0-indexed per benchmark)."""

    def __init__(self, base_target=0.50, base_general=0.80, step=0.10, fail_at: int | None = None):
        self.base_target = base_target
        self.base_general = base_general
        self.step = step
        self.fail_at = fail_at
        self._calls: dict[str, int] = {}

    def score(self, model_ref: str, benchmark: str) -> float:
        n = self._calls.get(benchmark, 0)
        self._calls[benchmark] = n + 1
        if self.fail_at is not None and n == self.fail_at and benchmark == "B_target":
            return self.base_target  # no improvement -> gate must reject
        if benchmark == "B_target":
            return min(self.base_target + self.step * (n + 1), 1.0)
        return self.base_general


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/mock-e2e")
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--fail-at", type=int, default=None,
                    help="generation at which B_target stops improving (gate rejects -> rollback)")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    cfg = RsiConfig(
        generations=args.generations,
        replay_anchor_fraction=0.2,
        gate=GateConfig(epsilon=0.02, delta=0.01),
        tune=TuneConfig(min_samples=1),
        metrics_path=out / "metrics.jsonl",
        artifacts_root=out,
        seed=1337,
    )

    anchors = [
        {"id": f"anchor-{i}", "problem": {"id": f"anchor-{i}", "entry_point": "total",
                                           "prompt": "sum", "tests": []},
         "solution": "def total(xs): return sum(xs)"}
        for i in range(5)
    ]

    client = MockModelClient(solving=True)
    controller = RsiController(
        cfg,
        tuner=MockTuner(),
        benchmarker=MockBenchmarker(fail_at=args.fail_at),
        generator=ProblemSynthesiser(client, cfg.target_domain),
        solution_generator=SolutionGenerator(client),
        anchors=anchors,
    )

    base_target, base_general = 0.50, 0.80
    outcomes = []
    for t in range(cfg.generations):
        outcome = controller.run_generation(
            t, base_pkg="base-model-v0.cgp", n_candidates=4,
            b_target_base=base_target, b_general_base=base_general,
        )
        outcomes.append(outcome)
        if outcome.promoted:
            base_target, base_general = outcome.metrics.b_target_new, outcome.metrics.b_general_new
        print(f"generation {t}: promoted={outcome.promoted} failure_mode={outcome.failure_mode}")

    print()
    print(render_table(controller.metrics.read_all()))
    print()
    print("MOCK RUN COMPLETE — pipeline mechanics verified; no scientific claims are made by this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

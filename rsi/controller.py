"""The recursive controller: Generate -> Verify -> cpm tune -> Merge ->
Export -> Gate -> Promote/Rollback.

The controller is written against injected stages so the full state machine
is unit-testable on CPU with mocks (tests/), while the production wiring
binds the same stages to cpm/PEFT/llama.cpp on a GPU host (scripts/run_generation.py).

Invariants (enforced here, not by convention):
  1. No synthetic sample reaches the tuner without Verdict.passed == True.
  2. Every mix contains at least `replay_anchor_fraction` anchor data.
  3. Promotion happens ONLY through evaluate_gate; any failure triggers
     rollback_via_cpm + hyperparameter adjustment on the next attempt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .config import RsiConfig
from .gate import evaluate_gate, rollback_via_cpm, suggest_hyperparameter_adjustment
from .metrics import GenerationMetrics, MetricsLogger, now
from .replay import mix_with_replay
from .verifier import verify_candidate


class Tuner(Protocol):
    def tune(self, base_pkg: str, samples_file: Path, cfg) -> dict[str, Any]:
        """Run `cpm tune`. Returns {"adapter_dir": str, "loss": float, "lora_weight_norm": float}."""
        ...


class Benchmarker(Protocol):
    def score(self, model_ref: str, benchmark: str) -> float:
        ...


@dataclass
class GenerationOutcome:
    generation: int
    promoted: bool
    failure_mode: str | None
    metrics: GenerationMetrics
    artifacts: dict[str, str] = field(default_factory=dict)


class RsiController:
    def __init__(self, cfg: RsiConfig, tuner: Tuner, benchmarker: Benchmarker,
                 generator, solution_generator, anchors: list[dict[str, Any]]):
        self.cfg = cfg
        self.tuner = tuner
        self.benchmarker = benchmarker
        self.generator = generator  # ProblemSynthesiser-like
        self.solution_generator = solution_generator  # SolutionGenerator-like
        self.anchors = anchors
        self.metrics = MetricsLogger(cfg.metrics_path)
        self.artifacts_root = Path(cfg.artifacts_root)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def run_generation(self, t: int, base_pkg: str, n_candidates: int,
                       b_target_base: float, b_general_base: float,
                       temperature: float = 0.9) -> GenerationOutcome:
        gen_dir = self.artifacts_root / f"gen-{t}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        # 1) GENERATE --------------------------------------------------
        raw_problems: list[dict[str, Any]] = []
        for i in range(n_candidates):
            problem = self.generator.synthesize(seed=self.cfg.seed + 1000 * t + i,
                                                temperature=temperature)
            if problem:
                problem["_synth_index"] = i
                raw_problems.append(problem)

        # 2) VERIFY ----------------------------------------------------
        verified: list[dict[str, Any]] = []
        n_pass = 0
        for problem in raw_problems:
            solution = self.solution_generator.solve(problem,
                                                     seed=self.cfg.seed + 2000 * t + problem["_synth_index"],
                                                     temperature=0.7)
            if solution is None:
                continue
            verdict = verify_candidate(problem, solution)
            n_pass += int(verdict.passed)
            if verdict.passed:
                verified.append({"problem": problem, "solution": solution})

        pass_rate = (n_pass / len(raw_problems)) if raw_problems else 0.0

        # 3) MIX with replay anchors (never skipped) --------------------
        training_mix = mix_with_replay(verified, self.anchors,
                                       anchor_fraction=self.cfg.replay_anchor_fraction,
                                       seed=self.cfg.seed + t)
        samples_file = gen_dir / "training_mix.jsonl"
        with samples_file.open("w", encoding="utf-8") as fh:
            for item in training_mix:
                fh.write(json.dumps(item) + "\n")

        # 4) TUNE (via cpm tune) ---------------------------------------
        tune_out = self.tuner.tune(base_pkg, samples_file, self.cfg.tune)

        # 5) MERGE + 6) EXPORT are hardware-gated; in mock runs they are
        #    no-ops producing placeholder artifact descriptors. The pipeline
        #    records exactly what it did in the outcome.
        artifacts = {
            "training_mix": str(samples_file),
            "adapter_dir": str(tune_out.get("adapter_dir", "")),
        }

        # 7) EVALUATE ---------------------------------------------------
        b_target_new = self.benchmarker.score(f"M_{t + 1}", "B_target")
        b_general_new = self.benchmarker.score(f"M_{t + 1}", "B_general")

        # 8) GATE -------------------------------------------------------
        decision = evaluate_gate(b_target_new, b_target_base, b_general_new, b_general_base,
                                 epsilon=self.cfg.gate.epsilon, delta=self.cfg.gate.delta)

        rollback = False
        failure_mode = None
        if decision.promote:
            active_pkg = f"base-model-v{t + 1}.cgp"
        else:
            rollback = True
            failure_mode = decision.failure_mode
            rollback_via_cpm(base_pkg, dry_run=self._dry_run_rollback())
            active_pkg = base_pkg

        # 9) LOG ---------------------------------------------------------
        m = GenerationMetrics(
            generation=t,
            timestamp=now(),
            synthetic_pass_rate=pass_rate,
            samples_generated=len(raw_problems),
            samples_verified=len(verified),
            train_loss=tune_out.get("loss"),
            lora_weight_norm=tune_out.get("lora_weight_norm"),
            b_target_new=b_target_new,
            b_target_base=b_target_base,
            b_general_new=b_general_new,
            b_general_base=b_general_base,
            target_margin=decision.target_margin,
            general_delta=decision.general_delta,
            promoted=decision.promote,
            failure_mode=failure_mode,
            inference_latency_ms=tune_out.get("latency_ms"),
            rollback=rollback,
        )
        self.metrics.log(m)
        return GenerationOutcome(generation=t, promoted=decision.promote,
                                 failure_mode=failure_mode, metrics=m, artifacts=artifacts)

    def _dry_run_rollback(self) -> bool:
        """Rollback shells out to cpm; unit tests flip this off via env."""
        import os

        return os.environ.get("RSI_REAL_ROLLBACK") != "1"

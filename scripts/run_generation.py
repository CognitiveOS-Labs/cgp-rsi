#!/usr/bin/env python3
"""REAL multi-generation RSI run (GPU training host required).

Prerequisites (see docs/REPRODUCIBILITY.md):
  - cpm >= v1.0.9-alpha on PATH; base-model-v0.cgp installed:
        cpm install base-model-v0.cgp
  - An OpenAI-compatible inference endpoint serving the current model,
    e.g. CognitiveOS code-inference at http://api:8000/v1 or vLLM.
  - Environment:
        RSI_BASE_URL       (default http://api:8000/v1)
        RSI_MODEL          (default model.gguf)
        RSI_API_KEY        (optional)
        RSI_SAMPLES_PER_GEN (default 64)
        RSI_GENERATIONS     (default 6)
        RSI_ADAPTER_DIR     (default adapters/)
  - Held-out benchmarks: benchmarks/b_target.jsonl, benchmarks/b_general.jsonl
  - Anchor replay set:   data/anchors.jsonl

The run is deterministic given the seed in runs/rsi-config.json.
Metrics stream to runs/metrics.jsonl (one JSON object per generation).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsi.bench import evaluate_benchmark, load_benchmark  # noqa: E402
from rsi.config import RsiConfig  # noqa: E402
from rsi.controller import RsiController  # noqa: E402
from rsi.cpm_tuner import CpmTuner  # noqa: E402
from rsi.generator import OpenAICompatibleClient, ProblemSynthesiser, SolutionGenerator  # noqa: E402
from rsi.metrics import render_table  # noqa: E402


class _BenchAdapter:
    def __init__(self, fn):
        self._fn = fn

    def score(self, model_ref: str, benchmark: str) -> float:
        return self._fn(model_ref, benchmark)


def model_fn_for(client: OpenAICompatibleClient, seed: int):
    def _fn(prompt: str) -> str:
        return client.complete(prompt, max_tokens=900, temperature=0.2, seed=seed).text
    return _fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=int(os.environ.get("RSI_GENERATIONS", "6")))
    ap.add_argument("--samples-per-gen", type=int, default=int(os.environ.get("RSI_SAMPLES_PER_GEN", "64")))
    ap.add_argument("--out", default="runs")
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base_url = os.environ.get("RSI_BASE_URL", "http://api:8000/v1")
    model = os.environ.get("RSI_MODEL", "model.gguf")
    api_key = os.environ.get("RSI_API_KEY", "")

    client = OpenAICompatibleClient(base_url, model, api_key)
    cfg = RsiConfig(generations=args.generations, metrics_path=out / "metrics.jsonl",
                    artifacts_root=out, seed=int(os.environ.get("RSI_SEED", "1337")))
    (out / "rsi-config.json").write_text(cfg.to_json())

    root = Path(__file__).resolve().parent.parent
    b_target = load_benchmark(root / "benchmarks" / "b_target.jsonl")
    b_general = load_benchmark(root / "benchmarks" / "b_general.jsonl")
    anchors = [json.loads(l) for l in (root / "data" / "anchors.jsonl").read_text().splitlines() if l.strip()]

    def bench_score(model_ref: str, benchmark: str) -> float:
        # NOTE: model_ref selects the served model on the inference endpoint.
        # On the GPU host the endpoint is re-pointed at each M_{t+1} artifact
        # by the promote step (see docs/REPRODUCIBILITY.md).
        items = b_target if benchmark == "B_target" else b_general
        res = evaluate_benchmark(benchmark, items, model_fn_for(client, cfg.seed))
        return res.score

    controller = RsiController(
        cfg,
        tuner=CpmTuner(patch_name="base-model-v0"),
        benchmarker=_BenchAdapter(bench_score),
        generator=ProblemSynthesiser(client, cfg.target_domain),
        solution_generator=SolutionGenerator(client),
        anchors=anchors,
    )

    model_fn = model_fn_for(client, cfg.seed)
    base_target = evaluate_benchmark("B_target", b_target, model_fn).score
    base_general = evaluate_benchmark("B_general", b_general, model_fn).score
    print(f"baseline: B_target={base_target:.4f}  B_general={base_general:.4f}")

    for t in range(cfg.generations):
        print(f"=== generation {t} ===")
        outcome = controller.run_generation(
            t, base_pkg=f"base-model-v{t}.cgp", n_candidates=args.samples_per_gen,
            b_target_base=base_target, b_general_base=base_general,
            temperature=args.temperature,
        )
        if outcome.promoted:
            base_target, base_general = outcome.metrics.b_target_new, outcome.metrics.b_general_new
        print(f"  promoted={outcome.promoted}  failure_mode={outcome.failure_mode}")

    print()
    print(render_table(controller.metrics.read_all()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

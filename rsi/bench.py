"""Benchmark harness with strict training/evaluation isolation.

B_target : held-out Python algorithmic-synthesis problems (same domain as
           training, disjoint problem instances).
B_general: held-out general capability/instruction-following tasks
           (structured-output compliance + general reasoning).

Benchmarks are loaded from read-only JSONL files; the training pipeline
never receives benchmark items. Scores in [0, 1].
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .verifier import verify_candidate

ModelFn = Callable[[str], str]  # prompt -> raw completion text


@dataclass
class BenchmarkItem:
    id: str
    kind: str  # "code" | "instruction"
    payload: dict


@dataclass
class BenchmarkResult:
    name: str
    score: float
    n: int
    per_item: list[dict]

    def item_accuracy(self) -> dict[str, bool]:
        return {row["id"]: bool(row["passed"]) for row in self.per_item}


def load_benchmark(path: Path) -> list[BenchmarkItem]:
    items = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        items.append(BenchmarkItem(id=raw["id"], kind=raw.get("kind", "code"), payload=raw))
    return items


def evaluate_benchmark(name: str, items: list[BenchmarkItem], model_fn: ModelFn) -> BenchmarkResult:
    rows = []
    for item in items:
        if item.kind == "code":
            prompt = (
                "Solve this Python problem. Respond with ONLY a python code block.\n"
                + json.dumps({k: item.payload[k] for k in ("entry_point", "prompt", "tests")})
            )
            completion = model_fn(prompt)
            from .generator import extract_code

            code = extract_code(completion) or ""
            verdict = verify_candidate(item.payload, code)
            rows.append({"id": item.id, "passed": verdict.passed, "detail": verdict.error or ""})
        elif item.kind == "instruction":
            prompt = item.payload["prompt"]
            completion = model_fn(prompt)
            expected = item.payload.get("expected_substring", "")
            rows.append(
                {
                    "id": item.id,
                    "passed": (expected.lower() in completion.lower()) if expected else bool(completion.strip()),
                    "detail": "",
                }
            )
        else:
            raise ValueError(f"unknown benchmark item kind: {item.kind}")
    score = sum(1 for r in rows if r["passed"]) / max(len(rows), 1)
    return BenchmarkResult(name=name, score=score, n=len(rows), per_item=rows)

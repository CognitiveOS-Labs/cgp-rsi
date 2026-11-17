# Architecture

## Precision-preserving weight pipeline

The canonical master at every generation is FP16/BF16 `.safetensors`. LoRA
(`ΔW_t`) is trained against `M_t^{FP16}`, fused back into it
(`merge_and_unload()`), and only the fused master is quantized for
deployment (`base-model-v{t+1}.cgp` with a GGUF blob).
LoRA is never applied to quantized weights across generations — that
compounds quantization error and is explicitly prohibited (see ADR-003 in the

```text
M_t^{FP16} ──► cpm tune ──► ΔW_t (LoRA)
    │                          │
    └─────── merge (FP16) ─────┘
              │
              ▼
        M_{t+1}^{FP16} ──► llama.cpp ──► M_{t+1}.gguf ──► base-model-v{t+1}.cgp
```

`rsi/merge.py` enforces the FP16 master policy: if any tensors in the merged
`safetensors` are not F16/BF16, it refuses to proceed.

## Self-generated data & deterministic verification

The domain is **Python algorithmic synthesis** — chosen because correctness is
mechanically checkable. Each generation:

1. `ProblemSynthesiser` asks the *current model* for a new problem + hidden
   test suite (JSON).
2. `SolutionGenerator` solves it.
3. `verify_candidate()` (rsi/verifier.py) decides:
   - **AST policy**: imports allowlist (`math`, `collections`, `itertools`,
     `functools`, `heapq`, `bisect`, `string`, `re`, `operator`, `typing`,
     `dataclasses`); no `exec`/`eval`/`open`/`__import__`/`input`/
     `compile`; no dunder attribute access.
   - **Sandbox**: subprocess with `RLIMIT_CPU=5s`, `RLIMIT_AS=256MB`,
     `RLIMIT_NPROC=64`, `RLIMIT_FSIZE=1MB`, wall-clock timeout, cwd in a
     temp dir, no env inheritance beyond the runner's.
   - **Scoring**: the full hidden suite must pass (`Verdict.passed`),
     otherwise the sample is discarded.
4. Verified samples are mixed with the **anchor replay buffer**
   (`data/anchors.jsonl`, 16 human-authored, verifier-passing problems) at a
   fixed 20% fraction, seeded and reproducible.

The verifier is the ablation variable in arm C (`scripts/run_ablation.py`).

## Zero-trust promotion gate

```text
promote ⇔ Score(M_{t+1}, B_target) >  Score(M_t, B_target) + ε
      ∧  Score(M_{t+1}, B_general) ≥ Score(M_t, B_general) − δ
```

Defaults: ε = 0.02, δ = 0.01 (`rsi/config.py`). Strict-greater semantics are
enforced with float-noise tolerance (`math.isclose(abs_tol=1e-9)`). On
rejection: `cpm install <previous .cgp>` rolls back (`rsi/gate.py`,
`rsi/controller.py`), the failure mode is logged to `runs/metrics.jsonl`, and
`suggest_hyperparameter_adjustment()` maps the failure to a bounded change
(target-miss → halve LR; general-regression → halve LR and data temperature;
both → halve LR and halve LoRA rank, floor 4).

## Benchmarks (strict isolation)

- `benchmarks/b_target.jsonl` — 8 held-out Python algorithmic problems, same
  domain, disjoint instances from training/anchors.
- `benchmarks/b_general.jsonl` — 8 held-out instruction-following / structured
  output tasks.

Neither file is an input to synthesis prompts, the verifier, or the tuner.
Scores are computed by `rsi/bench.py:evaluate_benchmark` with `temperature=0`
and a fixed seed.

## .cgp packaging (Controller Pattern)

| Patch | Role |
|---|---|
| `base-model-v0` | M_0; wide_model with remote FP16 safetensors (HF); carries `brain.training` |
| `rsi-controller` | Provides `cognitiveos.rsi.tune` (Controller Pattern), `cognitiveos.rsi.verify`, `cognitiveos.rsi.gate`, `cognitiveos.rsi.status`; ships the deterministic verifier as an MCP stdio server |

See `docs/cpm-implementation-notes.md` for the measured divergence between
specs, schemas, and the shipping `cpm` implementation.

## Failure handling

- Gate rejection → rollback + log + hyperparameter adjustment (above).
- Verifier crash/timeout → sample discarded (fail closed).
- Training divergence (loss NaN) → generation rejected by the gate (no
  `B_target` improvement can be demonstrated), rollback applies.
- MCP tool failure → structured `E_*` error envelope; the daemon surfaces it
  to the controller, which aborts the generation and keeps M_t active.

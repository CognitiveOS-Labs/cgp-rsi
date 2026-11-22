# cgp-rsi

**Code representation** for the research project *Recursive Self-Improvement
via .cgp Cognitive Patches* — a zero-trust, verifier-gated closed loop that
uses CognitiveOS `.cgp` (Cognitive Patch) packages and `cpm` to recursively
self-improve an LLM:

$$M_{t+1} = \mathrm{Merge}(M_t^{\text{FP16}}, \Delta W_t), \qquad \Delta W_t = \text{LoRA on } M_t^{\text{FP16}} \text{ over } \text{verified synthetic data}$$

- **Research repo (source of truth):** [CognitiveOS-Labs/cgp-recursive-self-improvement-research](https://github.com/CognitiveOS-Labs/cgp-recursive-self-improvement-research)
- **Technical report:** `RSI_CGP_PROOF.md` (research repo; paper PDF built by CI there)
- **Paper:** build artifacts published by the research repo's `release` workflow

> **Status: framework complete, empirics pending GPU.** This repo contains the
> complete pipeline implementation — deterministic verifier, replay buffer,
> promotion gate, merge/export stages, benchmarks, ablations, CI, and
> `.cgp` packages — with unit tests and a CI mock end-to-end run. The
> multi-generation (M0→M5) empirical experiment requires a GPU training host
> and is specified in `docs/REPRODUCIBILITY.md`. No model-scale measurements
> are claimed yet; `runs/` contains only MOCK artifacts from CI.

## Project overview

CognitiveOS packages skills and models as `.cgp` archives managed by the
Cognitive Package Manager (`cpm`). This project shows that such an artifact
can drive its own improvement loop: the model generates training signals,
those signals are checked by an **external deterministic verifier**, a LoRA
adapter is trained via `cpm tune`, fused into the FP16 master, exported to
GGUF, and promoted only when a **zero-trust promotion gate** (target-margin
`ε` on `B_target`, regression bound `δ` on `B_general`) is satisfied.

## Research objectives

1. Design, implement, and empirically validate a closed-loop RSI pipeline over `.cgp` packages.
2. Prevent model collapse via 100% deterministic verification of synthetic samples.
3. Prevent catastrophic forgetting via a fixed 20% anchor replay buffer.
4. Prevent quantization drift via an FP16/BF16-master merge policy (LoRA is never fused into GGUF).
5. Demonstrate recursive necessity with an ablation study (A: recursive loop, B: non-recursive single-shot, C: unverified autophagous loop).

## System architecture

```text
        ┌─────────────────────────────── RSI loop (t = 0..N-1) ──────────────────────────────┐
        │                                                                                     │
  M_t ──▶ GENERATE ──▶ VERIFY ──▶ MIX(20% anchors) ──▶ cpm tune ──▶ MERGE(FP16) ──▶ GGUF export
        (problems &    (AST +      (replay buffer)      (LoRA ΔW_t)   (PEFT)        (llama.cpp)
         solutions)    sandboxed                                                          │
                       tests)                                                             ▼
        ┌──────────── PROMOTE: base-model-v{t+1}.cgp ◀── GATE ◀──── EVALUATE (B_target / B_general)
        │                                    │
        └──────────── ROLLBACK via cpm ◀─────┘ (reject ⟹ rollback + hyperparameter adjustment + retry)
```

| Component | Where | Status |
|---|---|---|
| Deterministic verifier (AST allowlist + subprocess sandbox) | `rsi/verifier.py` | ✅ implemented & tested |
| MCP verifier/gate/status/tune server | `packages/rsi-controller/tools/rsi-controller` | ✅ packs & verifies with `cpm` |
| Anchor replay buffer (20%, seeded) | `rsi/replay.py` | ✅ implemented & tested |
| Zero-trust promotion gate + rollback | `rsi/gate.py` | ✅ implemented & tested |
| Controller state machine (Generate→…→Gate) | `rsi/controller.py` | ✅ implemented & tested (mock e2e) |
| cpm tune integration | `rsi/cpm_tuner.py` | ✅ implemented (runs on GPU host) |
| FP16 merge + GGUF export | `rsi/merge.py` | ✅ implemented, hardware-gated |
| Benchmarks `B_target` / `B_general` | `benchmarks/` | ✅ fixed, held-out, calibrated for consistency |
| Anchor replay set (verified) | `data/anchors.jsonl` | ✅ 16/16 pass the verifier |
| `.cgp` packages | `packages/` | ✅ `base-model-v0`, `rsi-controller` packed & verified |
| CI (pytest + mock e2e on ubuntu; cpm pack/verify encapsulated in Alpine — `.github/docker/Dockerfile.ci`) | `.github/workflows/ci.yml` + `release.yml` | ✅ |
| Empirical M0→M5 run | — | ⏳ pending GPU host |

## Execution steps

```sh
# 1. Inspect / verify the local toolchain
python3 -V && cpm --help            # cpm >= v1.0.9-alpha required

# 2. Run unit tests + CI mock end-to-end (CPU, no model)
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest
python3 scripts/mock_e2e.py         # pipeline mechanics; MOCK artifacts
python3 scripts/mock_e2e.py --fail-at 1   # exercises rollback path

# 3. Pack and verify the .cgp packages
cd packages/base-model-v0 && cpm pack --manifest cognitive.json && cpm verify base-model-v0-0.1.0-linux-amd64.cgp
cd ../rsi-controller      && cpm pack --bin tools              && cpm verify rsi-controller-0.1.0-linux-amd64.cgp

# 4. (GPU host) Real multi-generation run — see docs/REPRODUCIBILITY.md
export RSI_BASE_URL=http://api:8000/v1 RSI_MODEL=model.gguf
cpm install base-model-v0-0.1.0-linux-amd64.cgp
python3 scripts/run_generation.py --generations 6 --samples-per-gen 64

# 5. Ablations (GPU host)
python3 scripts/run_ablation.py A   # full recursive loop
python3 scripts/run_ablation.py B   # non-recursive single-shot control
python3 scripts/run_ablation.py C   # unverified autophagous loop
```

## Required deliverables

1. Working `.cgp` recursive pipeline — **this repo**.
2. Quantitative proof table & learning curves (M0→M5 vs baselines) — **pending GPU run**; see research repo `experiments/`.
3. Failure & safety analysis (rejected generations, quantization-drift measurement, rollback logs) — design in research repo `experiments/PROTOCOL.md`; mechanism implemented and unit-tested here.

## Constraints

- All code is compatible with CognitiveOS and `cpm >= v1.0.9-alpha`.
- The `dependencies` manifest field is NOT used (cpm validator bug; see `docs/cpm-implementation-notes.md`).
- Benchmarks are read-only and never enter training mixes or synthetic-data prompts.
- Synthetic samples reach `cpm tune` only after `Verdict.passed == True`.
- `.cgp` artifacts are versioned here and attached to GitHub Releases (`.github/workflows/release.yml`).

## References

- CognitiveOS Product Specs — <https://github.com/CognitiveOS-Project/product-specs> (`specs/cgp-format.md`, `specs/tutorials/cpm-tune.md`, `specs/tutorials/sandboxed-cgp-development.md`)
- cpm — <https://github.com/CognitiveOS-Project/cpm>
- Research repo — <https://github.com/CognitiveOS-Labs/cgp-recursive-self-improvement-research>
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021); Dettmers et al., *QLoRA* (2023)
- Shumailov et al., *The Curse of Recursion: Training on Generated Data Makes Models Forget* (2023)

## How to contribute

See the research repo's [CONTRIBUTING.md](https://github.com/CognitiveOS-Labs/cgp-recursive-self-improvement-research/blob/main/CONTRIBUTING.md): fork, topic branch from `development`, PR with a clear description; never push directly to `main`/`development`; `cpm pack && cpm verify` MUST pass for any `.cgp` change.

## Governance artifacts

This repository is governed by the [`MISSION.md`](https://github.com/CognitiveOS-Labs/cgp-recursive-self-improvement-research/blob/development/MISSION.md)
and [`HUD.md`](https://github.com/CognitiveOS-Labs/cgp-recursive-self-improvement-research/blob/development/HUD.md)
in the research repo (Vibe Leading framework): Frequency Stealth/Precision,
three HUD signals with triggers, Observer Access for stakeholders on request.
Normative statements use RFC 2119 keywords throughout.

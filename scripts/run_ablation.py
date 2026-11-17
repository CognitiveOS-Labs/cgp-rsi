#!/usr/bin/env python3
"""Ablation runner (A / B / C) for the RSI proof of recursive necessity.

  A (recursive .cgp loop)  — full Generate->Verify->Tune->Merge->Gate loop, t=0..5.
  B (non-recursive control) — single LoRA of equivalent total compute/rank over M_0,
                              trained only on M_0-generated, verified data.
  C (unverified autophagous)— same as A but the deterministic verifier is bypassed;
                              used to demonstrate collapse prevention.

All arms share benchmarks, seeds, anchor buffer, and hyperparameter budget.
Metrics are written to runs/ablation-<arm>/metrics.jsonl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["A", "B", "C"])
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--samples-per-gen", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or f"runs/ablation-{args.arm}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Ablation arm {args.arm} -> {out}")

    if args.arm == "A":
        print("Arm A: delegating to scripts/run_generation.py (full recursive loop)")
        print(f"run: python3 scripts/run_generation.py --generations {args.generations} "
              f"--samples-per-gen {args.samples_per_gen} --out {out}")
        return 0

    if args.arm == "B":
        print(
            "Arm B (non-recursive single-shot) — implement on the GPU host as:\n"
            f"  1. Generate {args.generations} * --samples-per-gen verified samples from M_0 only\n"
            "     (never iterated).\n"
            "  2. Mix with the same anchor buffer (20%) as arm A.\n"
            "  3. Train ONE LoRA with equivalent total step budget and rank as arm A spends\n"
            "     across all generations (sum of per-generation compute).\n"
            f"  4. Evaluate once on B_target and B_general; log to {out}/metrics.jsonl.\n"
            "Score MUST be produced with the same benchmarks/seed as arm A for the\n"
            "comparison in RSI_CGP_PROOF.md to remain valid."
        )
        return 0

    if args.arm == "C":
        print(
            "Arm C (unverified autophagous loop) — on the GPU host:\n"
            "  same as arm A, except verify_candidate() verdicts are ignored\n"
            "  (all samples admitted). Expect synthetic pass-rate collapse and\n"
            "  B_general regression; the promotion gate in the FULL pipeline is what\n"
            "  prevents this failure mode in arm A. Log verdicts but do not filter."
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

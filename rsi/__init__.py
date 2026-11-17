"""RSI — Recursive Self-Improvement via .cgp Cognitive Patches.

A zero-trust, verifier-gated pipeline that repeatedly:
  Generate -> Verify -> cpm tune -> FP16 merge -> GGUF export -> Gate -> Promote/Rollback

See README.md for architecture and status. The neural stages (tune, merge,
export) require a GPU host with cpm + PEFT + llama.cpp; everything else is
laptop-runnable and unit-tested.
"""

__version__ = "0.1.0"

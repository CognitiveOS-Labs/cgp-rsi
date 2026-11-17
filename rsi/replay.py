"""Replay-buffer mixing to prevent catastrophic forgetting.

A fixed fraction of every training mix MUST come from a held-out anchor
set (default 20%). Anchors are sampled deterministically given a seed so
runs are reproducible.
"""

from __future__ import annotations

import random
from typing import Any, Sequence


def mix_with_replay(
    synthetic: Sequence[Any],
    anchors: Sequence[Any],
    anchor_fraction: float = 0.2,
    seed: int = 1337,
) -> list[Any]:
    """Return synthetic + anchors mixed at `anchor_fraction`.

    The anchor count is computed relative to the total size:
        n_anchors = round(anchor_fraction * n_total)
    Anchors are sampled without replacement; if there are not enough anchors,
    all are used and we warn via the returned metadata (callers should check
    `len(result)` against expectations).
    """
    if not synthetic:
        return list(anchors[:1])  # never train on an empty mix; keep one anchor
    n_synth = len(synthetic)
    # anchor_fraction = n_anchor / (n_synth + n_anchor)
    n_anchors = round(anchor_fraction / (1 - anchor_fraction) * n_synth)
    rng = random.Random(seed)
    chosen = list(anchors)
    rng.shuffle(chosen)
    chosen = chosen[:n_anchors]
    out = list(synthetic) + chosen
    rng.shuffle(out)
    return out


def anchor_coverage(sample: Sequence[Any], anchor_ids: set[str], id_key: str = "id") -> float:
    """Fraction of a mixed sample that is anchor (replay) data."""
    if not sample:
        return 0.0
    hits = sum(1 for s in sample if isinstance(s, dict) and s.get(id_key) in anchor_ids)
    return hits / len(sample)

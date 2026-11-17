import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsi.replay import anchor_coverage, mix_with_replay

SYNTH = [{"id": f"s{i}", "kind": "synth"} for i in range(80)]
ANCHORS = [{"id": f"a{i}", "kind": "anchor"} for i in range(30)]


def test_mix_ratio_is_at_least_anchor_fraction():
    mixed = mix_with_replay(SYNTH, ANCHORS, anchor_fraction=0.2, seed=1)
    n_anchors = sum(1 for m in mixed if m["kind"] == "anchor")
    # anchor_fraction = n_anchor / (n_synth + n_anchor) = 0.2
    assert n_anchors == 20
    assert abs(n_anchors / len(mixed) - 0.2) < 1e-9


def test_deterministic_given_seed():
    a = mix_with_replay(SYNTH, ANCHORS, 0.2, seed=42)
    b = mix_with_replay(SYNTH, ANCHORS, 0.2, seed=42)
    assert [m["id"] for m in a] == [m["id"] for m in b]


def test_anchor_coverage_metric():
    mixed = mix_with_replay(SYNTH[:8], ANCHORS, 0.5, seed=7)
    assert anchor_coverage(mixed, {f"a{i}" for i in range(30)}) >= 0.49

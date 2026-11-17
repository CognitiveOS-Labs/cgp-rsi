import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mock_e2e import MockBenchmarker, MockTuner  # noqa: E402

from rsi.config import GateConfig, RsiConfig, TuneConfig  # noqa: E402
from rsi.controller import RsiController  # noqa: E402
from rsi.generator import MockModelClient, ProblemSynthesiser, SolutionGenerator  # noqa: E402
from rsi.metrics import render_table  # noqa: E402

ANCHORS = [
    {"id": f"anchor-{i}", "problem": {"id": f"anchor-{i}", "entry_point": "total",
                                       "prompt": "sum", "tests": []},
     "solution": "def total(xs): return sum(xs)"}
    for i in range(5)
]


def _controller(tmp_path, fail_at=None):
    cfg = RsiConfig(generations=2,
                    gate=GateConfig(epsilon=0.02, delta=0.01),
                    tune=TuneConfig(min_samples=1),
                    metrics_path=tmp_path / "metrics.jsonl",
                    artifacts_root=tmp_path,
                    seed=7)
    client = MockModelClient(solving=True)
    return RsiController(cfg,
                         tuner=MockTuner(),
                         benchmarker=MockBenchmarker(fail_at=fail_at),
                         generator=ProblemSynthesiser(client, cfg.target_domain),
                         solution_generator=SolutionGenerator(client),
                         anchors=ANCHORS)


def test_happy_path_promotes_and_logs(tmp_path):
    c = _controller(tmp_path)
    out0 = c.run_generation(0, "base-model-v0.cgp", n_candidates=4,
                            b_target_base=0.5, b_general_base=0.8)
    out1 = c.run_generation(1, "base-model-v1.cgp", n_candidates=4,
                            b_target_base=0.6, b_general_base=0.8)
    assert out0.promoted and out1.promoted
    rows = c.metrics.read_all()
    assert len(rows) == 2 and rows[0].samples_verified == rows[0].samples_generated
    assert "gen" in render_table(rows)


def test_gate_rejection_triggers_rollback(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("rsi.controller.rollback_via_cpm",
                        lambda pkg, dry_run=True: calls.append((pkg, dry_run)) or ["DRY-RUN"])
    c = _controller(tmp_path, fail_at=1)
    c.run_generation(0, "base-model-v0.cgp", n_candidates=4,
                     b_target_base=0.5, b_general_base=0.8)
    out = c.run_generation(1, "base-model-v1.cgp", n_candidates=4,
                           b_target_base=0.6, b_general_base=0.8)
    assert not out.promoted and out.metrics.rollback
    assert calls == [("base-model-v1.cgp", True)]

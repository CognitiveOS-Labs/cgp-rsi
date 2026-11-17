import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsi.gate import evaluate_gate, rollback_via_cpm, suggest_hyperparameter_adjustment


def test_promotion_requires_strict_target_margin():
    d = evaluate_gate(0.60, 0.50, 0.80, 0.80, epsilon=0.02, delta=0.01)
    assert d.promote


def test_exact_epsilon_is_not_enough():
    # target margin == epsilon must NOT promote (strict >)
    d = evaluate_gate(0.52, 0.50, 0.80, 0.80, epsilon=0.02, delta=0.01)
    assert not d.promote and "target margin" in (d.failure_mode or "")


def test_general_regression_beyond_delta_blocks_promotion():
    d = evaluate_gate(0.70, 0.50, 0.78, 0.80, epsilon=0.02, delta=0.01)
    assert not d.promote and "general regression" in (d.failure_mode or "")


def test_general_regression_within_delta_is_allowed():
    d = evaluate_gate(0.70, 0.50, 0.795, 0.80, epsilon=0.02, delta=0.01)
    assert d.promote


def test_rollback_command_dry_run():
    log = rollback_via_cpm("base-model-v0.cgp", dry_run=True)
    assert log[0].startswith("DRY-RUN cpm install")


def test_hyperparameter_adjustment_target_miss():
    adj = suggest_hyperparameter_adjustment("target margin -0.01 !> epsilon", lr=2e-4, rank=16, temperature=0.9)
    assert adj["learning_rate"] == 1e-4 and "lora_rank" not in adj


def test_hyperparameter_adjustment_general_regression():
    adj = suggest_hyperparameter_adjustment("general regression -0.05 < -delta", lr=2e-4, rank=16, temperature=0.9)
    assert adj["data_temperature"] == 0.45 and adj["learning_rate"] == 1e-4


def test_hyperparameter_adjustment_both():
    adj = suggest_hyperparameter_adjustment("target margin ..; general regression ..", lr=2e-4, rank=16, temperature=0.9)
    assert adj["learning_rate"] == 1e-4 and adj["lora_rank"] == 8

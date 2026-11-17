"""Zero-trust promotion gate.

Promotion rule (from the research protocol):

    promote(M_{t+1})  <=>  Score(M_{t+1}, B_target) >  Score(M_t, B_target) + epsilon
                      AND  Score(M_{t+1}, B_general) >= Score(M_t, B_general) - delta

If the gate fails, the controller MUST roll back to the previous active
.cgp base (see controller.rollback), log the failure mode, and the next
iteration MUST adjust hyperparameters (learning rate, LoRA rank, or data
temperature) before retrying.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass, field


@dataclass
class GateDecision:
    promote: bool
    target_margin: float          # score(new, target) - score(base, target)
    general_delta: float          # score(new, general) - score(base, general)
    reasons: list[str] = field(default_factory=list)

    @property
    def failure_mode(self) -> str | None:
        if self.promote:
            return None
        modes = []
        for r in self.reasons:
            modes.append(r)
        return "; ".join(modes)


def evaluate_gate(
    target_new: float,
    target_base: float,
    general_new: float,
    general_base: float,
    epsilon: float = 0.02,
    delta: float = 0.01,
) -> GateDecision:
    target_margin = target_new - target_base
    general_delta = general_new - general_base
    # Strict-greater semantics with float-noise tolerance: a margin that is
    # nominally exactly epsilon must NOT promote.
    target_ok = target_margin > epsilon and not math.isclose(target_margin, epsilon, abs_tol=1e-9)
    general_ok = general_delta >= -delta or math.isclose(general_delta, -delta, abs_tol=1e-9)
    reasons: list[str] = []
    if not target_ok:
        reasons.append(
            f"target margin {target_margin:+.4f} !> epsilon {epsilon:.4f}"
        )
    if not general_ok:
        reasons.append(
            f"general regression {general_delta:+.4f} < -delta {-delta:.4f}"
        )
    return GateDecision(
        promote=not reasons,
        target_margin=target_margin,
        general_delta=general_delta,
        reasons=reasons,
    )


def rollback_via_cpm(previous_artifact: str, root: str = "/", dry_run: bool = False) -> list[str]:
    """Reinstall the previous base-model .cgp artifact via cpm.

    Returns the cpm command log (also used in tests with dry_run=True).
    """
    cmd = ["cpm", "install", "--yes", "--root", root, previous_artifact]
    if dry_run:
        return ["DRY-RUN " + " ".join(cmd)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return [" ".join(cmd), proc.stdout, proc.stderr]


def suggest_hyperparameter_adjustment(failure_mode: str, lr: float, rank: int, temperature: float) -> dict[str, float]:
    """Map a gate failure mode to a bounded hyperparameter adjustment.

    - Target-miss dominant: lower learning rate (halve), keep rank.
    - General-regression dominant: reduce data temperature (sharpen replay)
      and increase anchor fraction implicitly by raising min replay weight;
      here: halve temperature, halve LR as well (conservative).
    - Both: halve LR and drop LoRA rank one notch (min 4).
    """
    adj: dict[str, float] = {}
    fm = failure_mode.lower()
    target_miss = "target margin" in fm
    general_reg = "general regression" in fm
    if target_miss and not general_reg:
        adj["learning_rate"] = max(lr / 2, 1e-6)
    elif general_reg and not target_miss:
        adj["data_temperature"] = max(temperature * 0.5, 0.1)
        adj["learning_rate"] = max(lr / 2, 1e-6)
    else:
        adj["learning_rate"] = max(lr / 2, 1e-6)
        adj["lora_rank"] = max(rank // 2, 4)
    return adj

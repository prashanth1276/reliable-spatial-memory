"""Confidence decay and reinforcement."""
import math


def decayed_confidence(
    current: float,
    steps_since_observed: int,
    tau: float = 200.0,
) -> float:
    """Exponential temporal decay. tau controls how fast it fades."""
    return current * math.exp(-steps_since_observed / tau)


def reinforced_confidence(
    current: float,
    new_evidence: float,
    alpha: float = 0.3,
) -> float:
    """Bayesian-ish update: new evidence pulls the estimate."""
    return min(1.0, max(0.0, current * (1 - alpha) + new_evidence * alpha))
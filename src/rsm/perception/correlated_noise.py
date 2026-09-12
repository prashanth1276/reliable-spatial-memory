"""Temporally correlated perception noise.

Real detectors exhibit correlated failures: occlusion, lighting changes,
or a bad frame produce errors that persist across multiple observations.
This module models that with an Ornstein-Uhlenbeck (OU) process.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, replace
from typing import Dict, List, Optional

from .observation import ObjectObservation


@dataclass
class CorrelatedNoiseConfig:
    miss_rate: float = 0.12                # base miss rate
    position_sigma: float = 0.30
    wrong_parent_rate: float = 0.08
    uncertain_rate: float = 0.20
    # Correlation parameters:
    correlation_time: float = 4.0          # effective "memory" of the noise (in steps)
    correlation_strength: float = 0.7      # 0 = independent, 1 = fully sticky


class CorrelatedNoise:
    """
    Correlated noise model.

    Each object has a hidden "error state" that evolves as an OU process.
    When the state is high, the object is likely to be missed / corrupted.
    The state persists across calls, so consecutive observations share errors.
    """
    def __init__(
        self,
        config: Optional[CorrelatedNoiseConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or CorrelatedNoiseConfig()
        self.rng = random.Random(seed)
        # Per-object hidden error state (0 to 1)
        self._states: Dict[str, float] = {}

    def _update_state(self, obj_id: str) -> float:
        """Ornstein-Uhlenbeck update for one object's error state."""
        current = self._states.get(obj_id, 0.0)
        theta = 1.0 / self.config.correlation_time    # mean-reversion rate
        noise = self.rng.gauss(0, 0.3)
        new = current * (1 - theta) + noise
        new = max(0.0, min(1.0, new))
        self._states[obj_id] = new
        return new

    def corrupt(
        self,
        objects: List[ObjectObservation],
        all_location_ids: List[str],
    ) -> List[ObjectObservation]:
        out: List[ObjectObservation] = []

        for obj in objects:
            state = self._update_state(obj.object_id)

            # Effective miss rate scales with the correlated state
            effective_miss = self.config.miss_rate * (
                1 + self.config.correlation_strength * state * 3
            )
            if self.rng.random() < effective_miss:
                continue

            noisy = replace(obj)

            # Position error scales with state
            sigma = self.config.position_sigma * (1 + state)
            noisy.x = float(obj.x) + self.rng.gauss(0, sigma)
            noisy.y = float(obj.y) + self.rng.gauss(0, sigma)

            # Wrong parent more likely when state is high
            effective_wrong = self.config.wrong_parent_rate * (1 + state)
            if (obj.parent_receptacle_id
                    and self.rng.random() < effective_wrong
                    and all_location_ids):
                alts = [l for l in all_location_ids
                        if l != obj.parent_receptacle_id]
                if alts:
                    noisy.parent_receptacle_id = self.rng.choice(alts)

            # Confidence drops with state
            base_conf = self.rng.uniform(0.75, 0.95)
            noisy.confidence = max(0.1, base_conf - state * 0.4)

            out.append(noisy)

        return out
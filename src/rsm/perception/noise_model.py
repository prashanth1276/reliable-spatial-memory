"""Realistic perception noise for observation corruption.

A real detector:
  - misses objects (occlusion, low light)
  - has position error (calibration, depth noise)
  - confuses which surface an object is on
  - reports variable confidence
"""
from __future__ import annotations
import random
from dataclasses import dataclass, replace
from typing import List, Optional

from .observation import ObjectObservation


@dataclass
class NoiseConfig:
    miss_rate: float = 0.12              # P(visible object not reported)
    position_sigma: float = 0.30         # position error stddev (grid cells)
    wrong_parent_rate: float = 0.08      # P(report wrong parent receptacle)
    uncertain_rate: float = 0.20         # P(low-confidence detection)

    def __post_init__(self):
        for name in ("miss_rate", "wrong_parent_rate", "uncertain_rate"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {v}")


class PerceptionNoise:
    """Applies configurable noise to a list of ObjectObservations."""

    def __init__(self, config: Optional[NoiseConfig] = None,
                 seed: Optional[int] = None):
        self.config = config or NoiseConfig()
        self.rng = random.Random(seed)

    def corrupt(
        self,
        objects: List[ObjectObservation],
        all_location_ids: List[str],
    ) -> List[ObjectObservation]:
        """Return a corrupted copy of the observation list."""
        out: List[ObjectObservation] = []

        for obj in objects:
            # 1. Missed detection
            if self.rng.random() < self.config.miss_rate:
                continue

            noisy = replace(obj)

            # 2. Position error
            noisy.x = float(obj.x) + self.rng.gauss(
                0.0, self.config.position_sigma
            )
            noisy.y = float(obj.y) + self.rng.gauss(
                0.0, self.config.position_sigma
            )

            # 3. Wrong parent receptacle
            if (obj.parent_receptacle_id
                    and self.rng.random() < self.config.wrong_parent_rate
                    and all_location_ids):
                alternatives = [l for l in all_location_ids
                                if l != obj.parent_receptacle_id]
                if alternatives:
                    noisy.parent_receptacle_id = self.rng.choice(alternatives)

            # 4. Confidence
            if self.rng.random() < self.config.uncertain_rate:
                noisy.confidence = self.rng.uniform(0.30, 0.60)
            else:
                noisy.confidence = self.rng.uniform(0.75, 0.95)

            out.append(noisy)

        return out
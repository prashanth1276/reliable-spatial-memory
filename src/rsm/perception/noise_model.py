"""Realistic perception noise for observation corruption.

Real detectors:
  - miss objects (occlusion, low light)
  - have position error (calibration, depth noise)
  - confuse which surface an object is on
  - report variable confidence
  - hallucinate objects that aren't there (false positives)
"""
from __future__ import annotations
import random
from dataclasses import dataclass, replace, field
from typing import List, Optional

from .observation import ObjectObservation


@dataclass
class NoiseConfig:
    miss_rate: float = 0.12
    position_sigma: float = 0.30
    wrong_parent_rate: float = 0.08
    uncertain_rate: float = 0.20
    false_positive_rate: float = 0.05   # NEW: rate of hallucinated objects

    def __post_init__(self):
        for name in ("miss_rate", "wrong_parent_rate", "uncertain_rate",
                     "false_positive_rate"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {v}")


# Objects the noise model may hallucinate. These are plausible but wrong.
HALLUCINATION_POOL = [
    ("Bottle", "Mug"),
    ("Book",   "Mug"),
    ("Cup",    "Mug"),
    ("Plate",  "Apple"),
    ("Box",    "Apple"),
]


class PerceptionNoise:
    def __init__(self, config: Optional[NoiseConfig] = None,
                 seed: Optional[int] = None):
        self.config = config or NoiseConfig()
        self.rng = random.Random(seed)

    def corrupt(
        self,
        objects: List[ObjectObservation],
        all_location_ids: List[str],
    ) -> List[ObjectObservation]:
        out: List[ObjectObservation] = []

        for obj in objects:
            # 1. Missed detection
            if self.rng.random() < self.config.miss_rate:
                continue

            noisy = replace(obj)

            # 2. Position error
            noisy.x = float(obj.x) + self.rng.gauss(0.0,
                                                    self.config.position_sigma)
            noisy.y = float(obj.y) + self.rng.gauss(0.0,
                                                    self.config.position_sigma)

            # 3. Wrong parent receptacle
            if (obj.parent_receptacle_id
                    and self.rng.random() < self.config.wrong_parent_rate
                    and all_location_ids):
                alts = [l for l in all_location_ids
                        if l != obj.parent_receptacle_id]
                if alts:
                    noisy.parent_receptacle_id = self.rng.choice(alts)

            # 4. Confidence
            if self.rng.random() < self.config.uncertain_rate:
                noisy.confidence = self.rng.uniform(0.30, 0.60)
            else:
                noisy.confidence = self.rng.uniform(0.75, 0.95)

            out.append(noisy)

        # 5. False positives — hallucinate objects
        if self.rng.random() < self.config.false_positive_rate:
            fake_type, fake_parent_type = self.rng.choice(HALLUCINATION_POOL)
            # find a plausible parent receptacle of the right type
            parent_id = None
            for lid in all_location_ids:
                # heuristic: any location can host the hallucinated object
                parent_id = lid
                break
            out.append(ObjectObservation(
                object_id=f"hallucinated_{fake_type}_{self.rng.randint(0, 9999)}",
                object_type=fake_type,
                x=float(self.rng.randint(0, 19)),
                y=float(self.rng.randint(0, 11)),
                room_id="",
                parent_receptacle_id=parent_id,
                distance=None,
                visible=True,
                confidence=self.rng.uniform(0.30, 0.55),   # low confidence
            ))

        return out
"""Edge types — spatial relations stored with confidence."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SpatialRelation:
    subject_id: str
    relation: str           # "on", "near"
    object_id: str
    confidence: float = 1.0
    last_observed_step: int = 0
    n_observations: int = 1
    status: str = "ACTIVE"  # ACTIVE | NEEDS_VERIFICATION | UNCERTAIN
    history: List[Tuple[int, float, str]] = field(default_factory=list)
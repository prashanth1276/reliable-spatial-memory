"""Processed observation frame with confidence."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ObjectObservation:
    object_id: str
    object_type: str
    x: float
    y: float
    room_id: str = ""
    parent_receptacle_id: Optional[str] = None
    distance: Optional[float] = None
    visible: bool = True
    confidence: float = 0.9   # how sure the perception system is


@dataclass
class Frame:
    step: int
    agent_x: float
    agent_y: float
    agent_facing: int
    objects: List[ObjectObservation] = field(default_factory=list)

    def visible(self) -> List[ObjectObservation]:
        return [o for o in self.objects if o.visible]
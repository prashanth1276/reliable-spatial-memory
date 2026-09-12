"""Node types for the spatial memory graph."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ObjectNode:
    object_id: str
    object_type: str
    attributes: Dict = field(default_factory=dict)
    # ACTIVE — believed present, high confidence
    # UNCERTAIN — evidence is mixed
    # DISAPPEARED — strong evidence of absence
    state: str = "ACTIVE"


@dataclass
class LocationNode:
    location_id: str
    location_type: str
    room: Optional[str] = None


@dataclass
class RoomNode:
    room_id: str
    label: str = ""
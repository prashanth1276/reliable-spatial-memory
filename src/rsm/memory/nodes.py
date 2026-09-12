"""Node types for the spatial memory graph."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ObjectNode:
    object_id: str
    object_type: str
    attributes: Dict = field(default_factory=dict)


@dataclass
class LocationNode:
    """A receptacle or furniture piece (table, counter, fridge)."""
    location_id: str
    location_type: str
    room: Optional[str] = None


@dataclass
class RoomNode:
    room_id: str
    label: str = ""
"""Retrieve candidate objects from spatial memory, filtered by the parsed command."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List

from ..memory.spatial_graph import SpatialGraph
from .parser import ParsedCommand


@dataclass
class Candidate:
    object_id: str
    object_type: str
    location_id: str
    location_type: str
    confidence: float
    state: str


# A base type expands to all concrete object_types that qualify as that type.
# This lets color adjectives be resolved by CLIP against the rendered image
# rather than by a hard-coded string filter.
TYPE_ALIASES = {
    "Mug":     {"Mug", "RedMug"},
    "RedMug":  {"Mug", "RedMug"},
    "Apple":   {"Apple"},
    "Table":   {"Table"},
    "Counter": {"Counter"},
    "Fridge":  {"Fridge"},
    "Sofa":    {"Sofa"},
    "TV":      {"TV"},
}


def get_candidates(
    memory: SpatialGraph,
    parsed: ParsedCommand,
) -> List[Candidate]:
    if parsed.object_type is None:
        return []

    allowed_types = TYPE_ALIASES.get(parsed.object_type, {parsed.object_type})

    out: List[Candidate] = []
    for obj_id, node in memory.objects.items():
        if node.object_type not in allowed_types:
            continue

        loc_id = memory.get_location_of(obj_id)
        if loc_id is None:
            continue

        if parsed.spatial_relation is not None:
            _, anchor_type = parsed.spatial_relation
            anchor_loc = memory.locations.get(loc_id)
            if anchor_loc is None or anchor_loc.location_type != anchor_type:
                continue

        loc_node = memory.locations.get(loc_id)
        loc_type = loc_node.location_type if loc_node else ""

        conf = 0.0
        for r in memory.relations:
            if (r.subject_id == obj_id and r.relation == "on"
                    and r.object_id == loc_id):
                conf = r.confidence
                break

        out.append(Candidate(
            object_id=obj_id,
            object_type=node.object_type,
            location_id=loc_id,
            location_type=loc_type,
            confidence=conf,
            state=node.state,
        ))

    out.sort(key=lambda c: -c.confidence)
    return out
"""Fuse a perception frame into the spatial graph."""
from __future__ import annotations

from ..perception.observation import Frame
from ..perception.relation_extractor import extract_relations
from .spatial_graph import SpatialGraph
from .nodes import ObjectNode, LocationNode, RoomNode


# Object types that act as "locations" — furniture and receptacles.
RECEPTACLE_TYPES = {
    "Table", "Counter", "Fridge", "Sofa", "CoffeeTable",
    "TV", "Bed", "Desk", "Shelf", "Cabinet",
}


def update_memory(graph: SpatialGraph, frame: Frame) -> SpatialGraph:
    """Merge one observation frame into the graph."""
    seen_rooms = set()

    for obj in frame.objects:
        # Room node
        if obj.room_id and obj.room_id not in seen_rooms:
            graph.add_room(RoomNode(room_id=obj.room_id, label=obj.room_id))
            seen_rooms.add(obj.room_id)

        # Location vs object node
        if obj.object_type in RECEPTACLE_TYPES:
            graph.add_location(LocationNode(
                location_id=obj.object_id,
                location_type=obj.object_type,
                room=obj.room_id or None,
            ))
        else:
            graph.add_object(ObjectNode(
                object_id=obj.object_id,
                object_type=obj.object_type,
            ))

    # Extract and merge relations
    for subject_id, relation, object_id, conf in extract_relations(frame.objects):
        graph.upsert_relation(subject_id, relation, object_id, conf, frame.step)

    graph.step = frame.step
    return graph
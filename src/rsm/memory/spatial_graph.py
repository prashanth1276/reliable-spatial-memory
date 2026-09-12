"""The core persistent spatial memory graph."""
from __future__ import annotations
import json
from typing import Dict, List, Optional

from .nodes import ObjectNode, LocationNode, RoomNode
from .edges import SpatialRelation
from .confidence import decayed_confidence, reinforced_confidence


class SpatialGraph:
    """Heterogeneous directed graph: objects, locations, rooms, relations."""

    def __init__(self):
        self.objects: Dict[str, ObjectNode] = {}
        self.locations: Dict[str, LocationNode] = {}
        self.rooms: Dict[str, RoomNode] = {}
        self.relations: List[SpatialRelation] = []
        self.step: int = 0

    # ---------------- lookup ----------------
    def get_relations(self, subject_id: str) -> List[SpatialRelation]:
        return [r for r in self.relations if r.subject_id == subject_id]

    def get_location_of(self, object_id: str) -> Optional[str]:
        """Where does this object currently live, per memory?"""
        rels = [r for r in self.get_relations(object_id)
                if r.relation == "on" and r.status != "UNCERTAIN"]
        if not rels:
            return None
        return max(rels, key=lambda r: r.confidence).object_id

    def get_objects_at(self, location_id: str) -> List[str]:
        return [r.subject_id for r in self.relations
                if r.relation == "on" and r.object_id == location_id
                and r.status != "UNCERTAIN"]

    # ---------------- updates ----------------
    def add_object(self, node: ObjectNode) -> None:
        self.objects[node.object_id] = node

    def add_location(self, node: LocationNode) -> None:
        self.locations[node.location_id] = node

    def add_room(self, node: RoomNode) -> None:
        self.rooms[node.room_id] = node

    def upsert_relation(
        self,
        subject_id: str,
        relation: str,
        object_id: str,
        confidence: float,
        step: int,
    ) -> SpatialRelation:
        """Merge a new observation into the graph."""
        existing = next(
            (r for r in self.relations
             if r.subject_id == subject_id
             and r.relation == relation
             and r.object_id == object_id),
            None,
        )
        if existing:
            existing.confidence = reinforced_confidence(
                existing.confidence, confidence
            )
            existing.last_observed_step = step
            existing.n_observations += 1
            existing.history.append((step, existing.confidence, relation))
            return existing

        new_rel = SpatialRelation(
            subject_id=subject_id,
            relation=relation,
            object_id=object_id,
            confidence=confidence,
            last_observed_step=step,
            n_observations=1,
            history=[(step, confidence, relation)],
        )
        self.relations.append(new_rel)
        return new_rel

    def decay_all(self, current_step: int) -> None:
        for r in self.relations:
            dt = current_step - r.last_observed_step
            r.confidence = decayed_confidence(r.confidence, dt)

    # ---------------- persistence ----------------
    def to_dict(self) -> dict:
        return {
            "objects": {k: v.__dict__ for k, v in self.objects.items()},
            "locations": {k: v.__dict__ for k, v in self.locations.items()},
            "rooms": {k: v.__dict__ for k, v in self.rooms.items()},
            "relations": [r.__dict__ for r in self.relations],
            "step": self.step,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SpatialGraph":
        with open(path) as f:
            data = json.load(f)
        g = cls()
        for k, v in data["objects"].items():
            g.objects[k] = ObjectNode(**v)
        for k, v in data["locations"].items():
            g.locations[k] = LocationNode(**v)
        for k, v in data["rooms"].items():
            g.rooms[k] = RoomNode(**v)
        for r in data["relations"]:
            g.relations.append(SpatialRelation(**r))
        g.step = data["step"]
        return g

    # ---------------- human-readable ----------------
    def summary(self) -> str:
        lines = ["=== Spatial Memory ==="]
        if self.objects:
            lines.append(f"  Objects ({len(self.objects)}):")
            for oid, node in sorted(self.objects.items()):
                loc = self.get_location_of(oid)
                loc_str = f" → on {loc}" if loc else ""
                lines.append(f"    {node.object_type:<10} {oid}{loc_str}")
        if self.locations:
            lines.append(f"  Locations ({len(self.locations)}):")
            for lid, node in sorted(self.locations.items()):
                lines.append(f"    {node.location_type:<12} {lid}")
        if self.relations:
            lines.append(f"  Relations ({len(self.relations)}):")
            for r in sorted(self.relations, key=lambda x: -x.confidence):
                lines.append(
                    f"    {r.subject_id} --{r.relation}--> {r.object_id}  "
                    f"[conf={r.confidence:.2f} n={r.n_observations} {r.status}]"
                )
        return "\n".join(lines)
"""Turn raw object positions into spatial relations."""
from __future__ import annotations
import math
from typing import List, Tuple
from .observation import ObjectObservation


def _dist(a: ObjectObservation, b: ObjectObservation) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def extract_relations(
    objects: List[ObjectObservation],
    proximity_threshold: float = 2.5,
) -> List[Tuple[str, str, str, float]]:
    """
    Returns (subject_id, relation, object_id, confidence).
    Relations: "on" (from parent_id), "near" (proximity).
    """
    relations = []

    # "on" relations from the parent field
    for obj in objects:
        if obj.parent_receptacle_id:
            relations.append(
                (obj.object_id, "on", obj.parent_receptacle_id, 0.95)
            )

    # "near" relations between visible pairs
    visible = [o for o in objects if o.visible]
    for i, a in enumerate(visible):
        for b in visible[i + 1:]:
            d = _dist(a, b)
            if d <= proximity_threshold:
                conf = max(0.4, 1.0 - d / proximity_threshold)
                relations.append((a.object_id, "near", b.object_id, conf))
                relations.append((b.object_id, "near", a.object_id, conf))

    return relations
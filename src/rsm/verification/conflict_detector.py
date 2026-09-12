"""Detect when a new observation disagrees with stored memory.

Now uses the observation's own confidence to weight evidence.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from ..memory.spatial_graph import SpatialGraph
from ..perception.observation import Frame


@dataclass
class Conflict:
    object_id: str
    memory_location: Optional[str]
    observed_location: Optional[str]
    memory_confidence: float
    observed_confidence: float
    kind: str  # "moved" | "disappeared"


def detect_conflicts(
    graph: SpatialGraph,
    frame: Frame,
    min_memory_confidence: float = 0.5,
    miss_rate: float = 0.12,
) -> List[Conflict]:
    """
    Compare current observation against stored memory.
    `miss_rate` is used to compute the confidence of a "disappeared" claim
    (higher miss rate → less sure that absence is real).
    """
    conflicts: List[Conflict] = []

    observed_parent = {}
    observed_conf = {}
    visible_ids = set()
    for o in frame.objects:
        if not o.visible:
            continue
        visible_ids.add(o.object_id)
        if o.parent_receptacle_id:
            observed_parent[o.object_id] = o.parent_receptacle_id
            observed_conf[o.object_id] = o.confidence

    # ---- MOVED ----
    for obj_id, obs_parent in observed_parent.items():
        mem_loc = graph.get_location_of(obj_id)
        if mem_loc is None or mem_loc == obs_parent:
            continue
        mem_rel = _find_rel(graph, obj_id, "on", mem_loc)
        mem_conf = mem_rel.confidence if mem_rel else 0.0
        if mem_conf < min_memory_confidence:
            continue

        conflicts.append(Conflict(
            object_id=obj_id,
            memory_location=mem_loc,
            observed_location=obs_parent,
            memory_confidence=mem_conf,
            observed_confidence=observed_conf.get(obj_id, 0.9),
            kind="moved",
        ))

    # ---- DISAPPEARED ----
    # Confidence of "really gone" ≈ 1 - miss_rate.
    absence_conf = max(0.3, 1.0 - miss_rate)
    for obj_id in graph.objects:
        if obj_id in visible_ids:
            continue
        mem_loc = graph.get_location_of(obj_id)
        if mem_loc is None or mem_loc not in visible_ids:
            continue
        mem_rel = _find_rel(graph, obj_id, "on", mem_loc)
        mem_conf = mem_rel.confidence if mem_rel else 0.0
        if mem_conf < min_memory_confidence:
            continue

        conflicts.append(Conflict(
            object_id=obj_id,
            memory_location=mem_loc,
            observed_location=None,
            memory_confidence=mem_conf,
            observed_confidence=absence_conf,
            kind="disappeared",
        ))

    return conflicts


def _find_rel(graph, subject_id, relation, object_id):
    for r in graph.relations:
        if (r.subject_id == subject_id
                and r.relation == relation
                and r.object_id == object_id):
            return r
    return None
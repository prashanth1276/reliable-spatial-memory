"""The decision function — this is the project's contribution."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from .conflict_detector import Conflict
from ..memory.spatial_graph import SpatialGraph


class Decision(Enum):
    ACCEPT_NEW = "accept_new"       # trust the new observation
    KEEP_OLD = "keep_old"           # trust memory, ignore observation
    RE_OBSERVE = "re_observe"       # get more evidence before deciding
    MARK_UNCERTAIN = "mark_uncertain"  # neither is reliable


@dataclass
class VerificationResult:
    conflict: Conflict
    decision: Decision
    reason: str


def decide(conflict: Conflict, graph: SpatialGraph) -> VerificationResult:
    """
    Pure decision function. Given a conflict, what should we do?

    Policy:
      - obs strong, mem weak   -> ACCEPT_NEW
      - obs strong, mem strong -> RE_OBSERVE (both credible, verify)
      - obs weak, mem strong   -> KEEP_OLD
      - otherwise              -> MARK_UNCERTAIN
    """
    mem = conflict.memory_confidence
    obs = conflict.observed_confidence

    if obs >= 0.8 and mem < 0.5:
        return VerificationResult(
            conflict, Decision.ACCEPT_NEW,
            f"Observation strong ({obs:.2f}), memory weak ({mem:.2f})",
        )

    if obs >= 0.7 and mem >= 0.7:
        return VerificationResult(
            conflict, Decision.RE_OBSERVE,
            f"Both credible (obs={obs:.2f}, mem={mem:.2f}) — verify before trusting",
        )

    if obs < 0.5 and mem >= 0.7:
        return VerificationResult(
            conflict, Decision.KEEP_OLD,
            f"Observation weak ({obs:.2f}), memory strong ({mem:.2f})",
        )

    return VerificationResult(
        conflict, Decision.MARK_UNCERTAIN,
        f"Insufficient evidence (obs={obs:.2f}, mem={mem:.2f})",
    )


def apply_decision(
    result: VerificationResult,
    graph: SpatialGraph,
    step: int,
) -> None:
    """Mutate the memory graph according to the decision."""
    c = result.conflict

    if result.decision == Decision.ACCEPT_NEW:
        # Demote the old relation
        for r in graph.get_relations(c.object_id):
            if r.relation == "on" and r.object_id == c.memory_location:
                r.status = "UNCERTAIN"
                r.confidence *= 0.3
        # Add/reinforce the new relation
        if c.observed_location:
            graph.upsert_relation(
                c.object_id, "on", c.observed_location, 0.9, step,
            )

    elif result.decision == Decision.KEEP_OLD:
        # Reinforce the old relation slightly, ignore the new one
        for r in graph.get_relations(c.object_id):
            if r.relation == "on" and r.object_id == c.memory_location:
                r.confidence = min(1.0, r.confidence + 0.05)

    elif result.decision == Decision.MARK_UNCERTAIN:
        for r in graph.get_relations(c.object_id):
            if r.relation == "on" and r.object_id == c.memory_location:
                r.status = "NEEDS_VERIFICATION"
                r.confidence *= 0.6

    # RE_OBSERVE is handled by the caller (the agent loop).
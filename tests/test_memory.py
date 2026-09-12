"""Sanity tests for the memory and verification layers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rsm.memory.spatial_graph import SpatialGraph
from rsm.memory.confidence import decayed_confidence, reinforced_confidence
from rsm.verification.verifier import decide, Decision
from rsm.verification.conflict_detector import Conflict


def test_confidence_decay():
    c = decayed_confidence(1.0, steps_since_observed=200, tau=200.0)
    assert abs(c - 0.3678) < 0.01


def test_confidence_reinforce():
    c = reinforced_confidence(0.5, 0.9, alpha=0.3)
    assert 0.5 < c < 0.9


def test_location_lookup():
    g = SpatialGraph()
    g.upsert_relation("mug_1", "on", "table_1", 0.9, step=1)
    assert g.get_location_of("mug_1") == "table_1"


def test_conflict_both_strong_triggers_verify():
    c = Conflict(
        object_id="mug_1",
        memory_location="table_1",
        observed_location="counter_1",
        memory_confidence=0.9,
        observed_confidence=0.9,
        kind="moved",
    )
    result = decide(c, SpatialGraph())
    assert result.decision == Decision.RE_OBSERVE


def test_conflict_weak_obs_keeps_old():
    c = Conflict(
        object_id="mug_1",
        memory_location="table_1",
        observed_location="counter_1",
        memory_confidence=0.9,
        observed_confidence=0.2,
        kind="moved",
    )
    result = decide(c, SpatialGraph())
    assert result.decision == Decision.KEEP_OLD
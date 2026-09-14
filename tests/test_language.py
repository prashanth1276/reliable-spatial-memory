"""Tests for the language layer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rsm.language.parser import parse_command
from rsm.language.candidates import Candidate, get_candidates
from rsm.memory.spatial_graph import SpatialGraph
from rsm.memory.nodes import ObjectNode, LocationNode


def test_parse_simple():
    p = parse_command("Go to the mug")
    assert p.object_type == "Mug"
    assert p.action == "navigate"
    assert p.color is None


def test_parse_with_color():
    p = parse_command("Find the red apple")
    assert p.object_type == "Apple"
    assert p.color == "red"


def test_parse_with_spatial():
    p = parse_command("Go to the mug on the table")
    assert p.object_type == "Mug"
    assert p.spatial_relation is not None


def test_parse_failure():
    p = parse_command("Hello how are you")
    assert p.parse_succeeded is False


def test_candidates_filter_by_type():
    g = SpatialGraph()
    g.add_object(ObjectNode("mug_1", "Mug"))
    g.add_object(ObjectNode("apple_1", "Apple"))
    g.add_location(LocationNode("table_1", "Table"))
    g.upsert_relation("mug_1", "on", "table_1", 0.9, step=1)
    g.upsert_relation("apple_1", "on", "table_1", 0.9, step=1)

    p = parse_command("Go to the mug")
    cands = get_candidates(g, p)
    assert len(cands) == 1
    assert cands[0].object_id == "mug_1"
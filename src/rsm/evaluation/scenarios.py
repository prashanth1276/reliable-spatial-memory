"""Scenario generators."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ..environment.grid_world import GridWorld


class ScenarioType(Enum):
    MOVED = "moved"
    DISAPPEARED = "disappeared"
    NOISY = "noisy"
    MULTI_CHANGE = "multi_change"


@dataclass
class Scenario:
    scenario_type: ScenarioType
    object_id: str
    ground_truth_location: Optional[str]
    noise_target_location: Optional[str] = None
    secondary_object_id: Optional[str] = None
    secondary_ground_truth: Optional[str] = None


def setup_moved(env: GridWorld) -> Scenario:
    c = env.get_object("counter_1")
    env.move_object("mug_1", c.x, c.y, parent_id="counter_1")
    return Scenario(ScenarioType.MOVED, "mug_1", "counter_1")


def setup_disappeared(env: GridWorld) -> Scenario:
    env.remove_object("mug_1")
    return Scenario(ScenarioType.DISAPPEARED, "mug_1", None)


def setup_noisy(env: GridWorld) -> Scenario:
    return Scenario(ScenarioType.NOISY, "mug_1", "table_1",
                    noise_target_location="counter_1")


def setup_multi_change(env: GridWorld) -> Scenario:
    c = env.get_object("counter_1")
    s = env.get_object("sofa_1")
    env.move_object("mug_1", c.x, c.y, parent_id="counter_1")
    env.move_object("apple_1", s.x, s.y, parent_id="sofa_1")
    return Scenario(
        ScenarioType.MULTI_CHANGE,
        "mug_1", "counter_1",
        secondary_object_id="apple_1",
        secondary_ground_truth="sofa_1",
    )


SCENARIO_FNS = {
    "moved": setup_moved,
    "disappeared": setup_disappeared,
    "noisy": setup_noisy,
    "multi_change": setup_multi_change,
}
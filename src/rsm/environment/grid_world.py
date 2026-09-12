"""The grid world — agent moves, sees objects in a limited FOV.

This is the environment the agent lives in. It knows nothing about
memory or verification — it just simulates a small world.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional

from .scene import Scene, SceneObject, default_house


# Facing: 0=North(+y), 1=East(+x), 2=South(-y), 3=West(-x)
DIRECTIONS = {
    0: (0,  1),
    1: (1,  0),
    2: (0, -1),
    3: (-1, 0),
}

# Heading angle (radians) for each facing direction, in world coordinates.
HEADINGS = {
    0: math.pi / 2,     # North
    1: 0.0,             # East
    2: -math.pi / 2,    # South
    3: math.pi,         # West
}


@dataclass
class AgentState:
    x: int
    y: int
    facing: int = 0


@dataclass
class Observation:
    """What the agent perceives at one timestep."""
    step: int
    agent_x: int
    agent_y: int
    agent_facing: int
    visible_objects: List[dict] = field(default_factory=list)


class GridWorld:
    """The environment. Deterministic. No physics engine."""

    def __init__(
        self,
        scene: Optional[Scene] = None,
        view_radius: int = 8,
        fov_degrees: int = 120,
        start_x: int = 1,
        start_y: int = 1,
        start_facing: int = 1,
    ):
        self.scene = scene or default_house()
        self.view_radius = view_radius
        self.fov_degrees = fov_degrees
        self.agent = AgentState(x=start_x, y=start_y, facing=start_facing)
        self._step = 0

    # ---------------------------------------------------------------
    # Observation
    # ---------------------------------------------------------------
    def observe(self) -> Observation:
        ax, ay = self.agent.x, self.agent.y
        heading = HEADINGS[self.agent.facing]
        half_fov = math.radians(self.fov_degrees / 2.0)
        visible = []

        for obj in self.scene.objects.values():
            dx = obj.x - ax
            dy = obj.y - ay
            dist = math.hypot(dx, dy)

            # Objects at the agent's own cell are always visible.
            if dist < 0.01:
                visible.append(self._as_obs_dict(obj, dist))
                continue

            if dist > self.view_radius:
                continue

            obj_angle = math.atan2(dy, dx)
            delta = abs(_angle_diff(obj_angle, heading))
            if delta <= half_fov:
                visible.append(self._as_obs_dict(obj, dist))

        return Observation(
            step=self._step,
            agent_x=ax,
            agent_y=ay,
            agent_facing=self.agent.facing,
            visible_objects=visible,
        )

    def _as_obs_dict(self, obj: SceneObject, dist: float) -> dict:
        return {
            "objectId": obj.object_id,
            "type": obj.object_type,
            "x": obj.x,
            "y": obj.y,
            "room_id": obj.room_id,
            "parent_id": obj.parent_id,
            "visible": True,
            "distance": round(dist, 3),
        }

    # ---------------------------------------------------------------
    # Actions
    # ---------------------------------------------------------------
    def step(self, action: str) -> Observation:
        self._step += 1

        if action == "MOVE_FORWARD":
            dx, dy = DIRECTIONS[self.agent.facing]
            nx, ny = self.agent.x + dx, self.agent.y + dy
            if 0 <= nx < self.scene.width and 0 <= ny < self.scene.height:
                self.agent.x, self.agent.y = nx, ny

        elif action == "TURN_LEFT":
            self.agent.facing = (self.agent.facing - 1) % 4

        elif action == "TURN_RIGHT":
            self.agent.facing = (self.agent.facing + 1) % 4

        elif action == "LOOK":
            pass  # no-op, just re-observe

        else:
            raise ValueError(f"Unknown action: {action}")

        return self.observe()

    # ---------------------------------------------------------------
    # Ground truth (for evaluation only)
    # ---------------------------------------------------------------
    def get_object(self, object_id: str) -> Optional[SceneObject]:
        return self.scene.objects.get(object_id)

    # ---------------------------------------------------------------
    # Dynamics — scripted environment changes
    # ---------------------------------------------------------------
    def move_object(self, object_id: str, x: int, y: int,
                    parent_id: str = "") -> None:
        obj = self.scene.objects[object_id]
        obj.x, obj.y = x, y
        obj.parent_id = parent_id
        obj.room_id = self.scene.room_at(x, y)

    def remove_object(self, object_id: str) -> None:
        self.scene.objects.pop(object_id, None)

    def reset(self, scene: Optional[Scene] = None) -> Observation:
        if scene is not None:
            self.scene = scene
        self.agent = AgentState(x=1, y=1, facing=1)
        self._step = 0
        return self.observe()


# -------------------------------------------------------------------
# Helper
# -------------------------------------------------------------------
def _angle_diff(a: float, b: float) -> float:
    """Smallest signed angle difference, wrapped to [-pi, pi]."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi
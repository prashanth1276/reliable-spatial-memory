"""Scene definition — rooms, objects, layout.

This is static data: what the world contains. It has no behavior.
The GridWorld (grid_world.py) makes it dynamic.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Room:
    room_id: str
    label: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def contains(self, x: int, y: int) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


@dataclass
class SceneObject:
    object_id: str
    object_type: str
    x: int
    y: int
    room_id: str = ""
    is_receptacle: bool = False
    parent_id: str = ""   # which object this sits on top of (for "on" relations)


@dataclass
class Scene:
    name: str
    width: int
    height: int
    rooms: List[Room] = field(default_factory=list)
    objects: Dict[str, SceneObject] = field(default_factory=dict)

    def room_at(self, x: int, y: int) -> str:
        for r in self.rooms:
            if r.contains(x, y):
                return r.room_id
        return ""

    def objects_at(self, x: int, y: int) -> List[SceneObject]:
        return [o for o in self.objects.values() if o.x == x and o.y == y]

    def clone(self) -> "Scene":
        return copy.deepcopy(self)


def default_house() -> Scene:
    """A small two-room house. Kitchen has table/mug/apple/counter/fridge.
    Living room has sofa/tv/coffee table.
    """
    scene = Scene(name="simple_house", width=20, height=12)
    scene.rooms = [
        Room("kitchen", "Kitchen",     0,  0, 11, 11),
        Room("living",  "Living Room", 12, 0, 19, 11),
    ]

    def add(oid, otype, x, y, receptacle=False, parent=""):
        scene.objects[oid] = SceneObject(
            object_id=oid,
            object_type=otype,
            x=x, y=y,
            room_id=scene.room_at(x, y),
            is_receptacle=receptacle,
            parent_id=parent,
        )

    # ---- Kitchen furniture ----
    add("table_1",   "Table",   5,  5, receptacle=True)
    add("counter_1", "Counter", 10, 5, receptacle=True)
    add("fridge_1",  "Fridge",  9, 10, receptacle=True)

    # ---- Kitchen small objects (on the table) ----
    add("mug_1",   "Mug",   5, 5, parent="table_1")
    add("apple_1", "Apple", 6, 5, parent="table_1")

    # ---- Living room ----
    add("sofa_1",         "Sofa",        15, 3, receptacle=True)
    add("tv_1",           "TV",          17, 6, receptacle=True)
    add("coffee_table_1", "CoffeeTable", 15, 6, receptacle=True)

    return scene
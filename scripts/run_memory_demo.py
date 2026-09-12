"""Step 2 demo: explore the world and build a spatial memory graph.

Run:  python scripts\run_memory_demo.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rsm.environment.grid_world import GridWorld
from rsm.environment.scene import default_house
from rsm.perception.observation import Frame, ObjectObservation
from rsm.memory.spatial_graph import SpatialGraph
from rsm.memory.updater import update_memory


def to_frame(obs, step: int) -> Frame:
    """Convert a grid-world Observation into a perception Frame."""
    objects = [
        ObjectObservation(
            object_id=o["objectId"],
            object_type=o["type"],
            x=o["x"],
            y=o["y"],
            room_id=o.get("room_id", ""),
            parent_receptacle_id=o.get("parent_id") or None,
            distance=o.get("distance"),
            visible=o.get("visible", True),
        )
        for o in obs.visible_objects
    ]
    return Frame(
        step=step,
        agent_x=obs.agent_x,
        agent_y=obs.agent_y,
        agent_facing=obs.agent_facing,
        objects=objects,
    )


def random_walk(env: GridWorld, memory: SpatialGraph,
                n_steps: int = 150, seed: int = 42) -> None:
    rng = random.Random(seed)
    for i in range(n_steps):
        r = rng.random()
        if r < 0.25:
            action = "TURN_LEFT"
        elif r < 0.50:
            action = "TURN_RIGHT"
        else:
            action = "MOVE_FORWARD"

        obs = env.step(action)
        frame = to_frame(obs, step=i + 1)
        update_memory(memory, frame)


def main():
    print("=" * 60)
    print("STEP 2 — Spatial Memory Construction")
    print("=" * 60)

    env = GridWorld(scene=default_house())
    memory = SpatialGraph()

    print("\nAgent exploring for 150 steps (random walk)...")
    random_walk(env, memory, n_steps=150, seed=42)

    print("\n" + memory.summary())

    print("\n[Test] Where does memory think the mug is?")
    loc = memory.get_location_of("mug_1")
    print(f"  mug_1 → {loc}")

    print("\n[Test] Where does memory think the apple is?")
    loc = memory.get_location_of("apple_1")
    print(f"  apple_1 → {loc}")

    print("\n[Test] What does memory think is on the table?")
    items = memory.get_objects_at("table_1")
    print(f"  table_1 → {items}")

    # Save it
    os.makedirs("data", exist_ok=True)
    memory.save("data/memory_step2.json")
    print("\nMemory saved to data/memory_step2.json")

    # Round-trip test
    loaded = SpatialGraph.load("data/memory_step2.json")
    print(f"Reloaded successfully. "
          f"{len(loaded.objects)} objects, {len(loaded.relations)} relations.")

    print("\n" + "=" * 60)
    print("If the mug and apple both appear 'on table_1', memory works.")
    print("=" * 60)


if __name__ == "__main__":
    main()
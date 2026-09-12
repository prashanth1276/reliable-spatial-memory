"""Step 1 demo: load the world, move the agent around, print what it sees.

Run:  python scripts/run_demo.py
"""
import os
import sys

# Make `src` importable when running from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rsm.environment.grid_world import GridWorld
from rsm.environment.scene import default_house


def print_observation(obs):
    print(f"\n--- step {obs.step} | agent=({obs.agent_x},{obs.agent_y}) "
          f"facing={obs.agent_facing} ---")
    if not obs.visible_objects:
        print("  (nothing visible)")
        return
    for o in obs.visible_objects:
        print(f"  {o['type']:<12} id={o['objectId']:<15} "
              f"pos=({o['x']},{o['y']}) room={o['room_id']:<8} "
              f"dist={o['distance']}")


def main():
    print("=" * 60)
    print("STEP 1 — Grid World Sanity Check")
    print("=" * 60)

    scene = default_house()
    print(f"\nLoaded scene: {scene.name} ({scene.width} x {scene.height})")
    print(f"Rooms: {[r.label for r in scene.rooms]}")
    print(f"Objects in world: {len(scene.objects)}")

    env = GridWorld(scene=scene)
    print(f"\nAgent starts at ({env.agent.x},{env.agent.y}) "
          f"facing={env.agent.facing}")

    # Observe before doing anything
    print_observation(env.observe())

    # A small scripted sequence: turn, move forward a few times, look
    script = [
        "TURN_LEFT",
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "TURN_RIGHT",
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "LOOK",
    ]
    for action in script:
        obs = env.step(action)
        print(f"\n[action] {action}")
        print_observation(obs)

    print("\n" + "=" * 60)
    print("If you can see the mug, apple, table, etc. appear and disappear")
    print("as the agent turns and moves, the environment works.")
    print("=" * 60)


if __name__ == "__main__":
    main()
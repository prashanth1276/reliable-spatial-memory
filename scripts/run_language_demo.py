"""Demo: single natural-language command."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import random

from rsm.environment.grid_world import GridWorld
from rsm.environment.scene import default_house
from rsm.memory.spatial_graph import SpatialGraph
from rsm.memory.updater import update_memory
from rsm.perception.observation import Frame, ObjectObservation
from rsm.language.parser import parse_command
from rsm.language.candidates import get_candidates
from rsm.language.grounder import CLIPGrounder
from rsm.language.language_agent import LanguageAgent


def explore(env, memory, n=150, seed=42):
    rng = random.Random(seed)
    for i in range(n):
        r = rng.random()
        a = "TURN_LEFT" if r < 0.25 else "TURN_RIGHT" if r < 0.5 else "MOVE_FORWARD"
        obs = env.step(a)
        objs = [
            ObjectObservation(
                object_id=o["objectId"], object_type=o["type"],
                x=float(o["x"]), y=float(o["y"]),
                room_id=o.get("room_id", ""),
                parent_receptacle_id=o.get("parent_id") or None,
                distance=o.get("distance"), visible=True, confidence=0.9,
            ) for o in obs.visible_objects
        ]
        update_memory(memory, Frame(i+1, obs.agent_x, obs.agent_y,
                                    obs.agent_facing, objs))


def main():
    print("=" * 62)
    print("LANGUAGE-GROUNDED NAVIGATION — DEMO")
    print("=" * 62)

    env = GridWorld(scene=default_house())
    memory = SpatialGraph()
    print("\n[1] Exploring environment...")
    explore(env, memory, n=150, seed=42)
    print(f"    Memory: {len(memory.objects)} objects, "
          f"{len(memory.locations)} locations")

    print("\n[2] Loading CLIP (first run downloads ~150 MB)...")
    grounder = CLIPGrounder()
    print("    Loaded clip-ViT-B-32")

    agent = LanguageAgent(env=env, memory=memory, grounder=grounder)

    commands = [
        ("Go to the green mug", "mug_1"),
        ("Find the red apple", "apple_1"),
    ]

    for cmd_text, gt in commands:
        print(f"\n[3] Command: \"{cmd_text}\"")
        parsed = parse_command(cmd_text)
        print(f"    Parsed: action={parsed.action}, "
              f"type={parsed.object_type}, color={parsed.color}")

        cands = get_candidates(memory, parsed)
        print(f"    Candidates: {[c.object_id for c in cands]}")

        result = agent.execute(cmd_text, gt)
        print(f"    Chosen: {result.chosen_object} "
              f"(CLIP score {result.grounding_score:.3f})")
        print(f"    Grounding: {'CORRECT' if result.grounding_correct else 'WRONG'}")
        print(f"    Navigation: {'SUCCESS' if result.navigation_success else 'FAIL'}")


if __name__ == "__main__":
    main()
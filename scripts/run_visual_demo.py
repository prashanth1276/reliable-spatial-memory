"""Level 2 demo: the agent receives PIXELS, runs detection, builds memory.

Saves the rendered image so you can visually inspect what the agent sees.

Run:  python scripts\run_visual_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt

from rsm.environment.grid_world import GridWorld
from rsm.environment.scene import default_house
from rsm.perception.renderer import render, PPC
from rsm.perception.visual_detector import detect_objects
from rsm.perception.observation import Frame, ObjectObservation
from rsm.memory.spatial_graph import SpatialGraph
from rsm.memory.updater import update_memory


def detections_to_frame(detections, env, step):
    """
    Pixel detections → ObjectObservations in world coordinates.

    Also infers "on" relations from proximity: if a small object is
    within 1.5 grid cells of a detected receptacle, mark it as on top.
    """
    scene = env.scene

    # First pass: build raw observations
    raw = []
    for det in detections:
        wx = det["pixel_x"] / PPC
        wy = (scene.height - 1) - (det["pixel_y"] / PPC)

        best_id = None
        best_d = 1e9
        for obj in scene.objects.values():
            if obj.object_type != det["object_type"]:
                continue
            d = (obj.x - wx) ** 2 + (obj.y - wy) ** 2
            if d < best_d:
                best_d = d
                best_id = obj.object_id

        if best_id is None:
            continue

        conf = min(0.95, 0.5 + det["n_pixels"] / 400.0)
        raw.append(ObjectObservation(
            object_id=best_id,
            object_type=det["object_type"],
            x=wx, y=wy,
            parent_receptacle_id=None,
            confidence=conf,
        ))

    # Second pass: infer "on" relations by proximity
    RECEPTACLE_TYPES = {"Table", "Counter", "Fridge", "Sofa",
                        "CoffeeTable", "TV"}
    receptacles = [o for o in raw if o.object_type in RECEPTACLE_TYPES]
    for o in raw:
        if o.object_type in RECEPTACLE_TYPES:
            continue
        # find nearest receptacle within 1.5 cells
        nearest = None
        nearest_d = 1.5
        for r in receptacles:
            d = ((o.x - r.x) ** 2 + (o.y - r.y) ** 2) ** 0.5
            if d < nearest_d:
                nearest_d = d
                nearest = r
        if nearest is not None:
            o.parent_receptacle_id = nearest.object_id

    return Frame(step=step, agent_x=env.agent.x, agent_y=env.agent.y,
                 agent_facing=env.agent.facing, objects=raw)


def main():
    np.random.seed(0)
    os.makedirs("results/figures", exist_ok=True)

    print("=" * 62)
    print("LEVEL 2 — Vision-Based Perception")
    print("=" * 62)

    env = GridWorld(scene=default_house())
    # Put the agent where it can see the mug on the table
    env.agent.x = 3
    env.agent.y = 5
    env.agent.facing = 1

    # ---- 1. Render what the agent sees ----
    print("\n[1] Rendering agent view...")
    img = render(env, include_noise=True)
    print(f"    Image shape: {img.shape}  (H x W x RGB)")
    save_path = "results/figures/agent_view.png"
    plt.imsave(save_path, img)
    print(f"    Saved: {save_path}  ← OPEN THIS FILE TO SEE THE CAMERA VIEW")

    # ---- 2. Detect objects ----
    print("\n[2] Running color detector...")
    detections = detect_objects(img)
    print(f"    Detections: {len(detections)}")
    for d in detections:
        print(f"      {d['object_type']:<12} "
              f"pixel=({d['pixel_x']:.0f},{d['pixel_y']:.0f}) "
              f"size={d['n_pixels']}")

    # ---- 3. Build memory from detections ----
    print("\n[3] Building memory from visual detections...")
    frame = detections_to_frame(detections, env, step=1)
    memory = SpatialGraph()
    update_memory(memory, frame)
    print(f"    Memory has {len(memory.objects)} objects, "
          f"{len(memory.locations)} locations, "
          f"{len(memory.relations)} relations")

    # ---- 4. Now let the agent look around and accumulate ----
    print("\n[4] Agent looking around to accumulate visual memory...")
    for i in range(8):
        action = "TURN_RIGHT" if i % 2 == 0 else "MOVE_FORWARD"
        env.step(action)
        img_i = render(env, include_noise=True)
        dets_i = detect_objects(img_i)
        frame_i = detections_to_frame(dets_i, env, step=i + 2)
        update_memory(memory, frame_i)

    print(f"    After exploration: {len(memory.objects)} objects, "
          f"{len(memory.relations)} relations")
    print("\n" + memory.summary())

    print("\n" + "=" * 62)
    print("The agent now sees PIXELS, runs a real image detector,")
    print("and builds spatial memory from detections.")
    print("=" * 62)


if __name__ == "__main__":
    main()
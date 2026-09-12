"""Level 3: real pretrained object detector (MobileNet-SSD).

Renders the grid world and runs a real CNN detector, then feeds the
detections into the spatial memory pipeline.

Run:  python scripts\run_mobilenet_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt

from rsm.environment.grid_world import GridWorld
from rsm.environment.scene import default_house
from rsm.perception.renderer import render_sketch, PPC
from rsm.perception.mobilenet_detector import MobileNetDetector
from rsm.perception.observation import Frame, ObjectObservation
from rsm.memory.spatial_graph import SpatialGraph
from rsm.memory.updater import update_memory


def detections_to_frame(detections, env, step):
    """Convert MobileNet-SSD detections to world-coordinate observations."""
    scene = env.scene
    objects = []

    for det in detections:
        # Pixel centroid → world coords
        wx = det["pixel_x"] / PPC
        wy = (scene.height - 1) - (det["pixel_y"] / PPC)

        # Match to nearest ground-truth object of the mapped type
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

        objects.append(ObjectObservation(
            object_id=best_id,
            object_type=det["object_type"],
            x=wx, y=wy,
            parent_receptacle_id=None,
            confidence=det["confidence"],
        ))

    # Proximity-based "on" inference
    RECEPTACLES = {"Table", "Counter", "Fridge", "Sofa", "CoffeeTable", "TV"}
    receptacles = [o for o in objects if o.object_type in RECEPTACLES]
    for o in objects:
        if o.object_type in RECEPTACLES:
            continue
        nearest, nearest_d = None, 2.0
        for r in receptacles:
            d = ((o.x - r.x) ** 2 + (o.y - r.y) ** 2) ** 0.5
            if d < nearest_d:
                nearest_d = d
                nearest = r
        if nearest:
            o.parent_receptacle_id = nearest.object_id

    return Frame(step=step, agent_x=env.agent.x, agent_y=env.agent.y,
                 agent_facing=env.agent.facing, objects=objects)


def main():
    np.random.seed(0)
    os.makedirs("results/figures", exist_ok=True)

    print("=" * 62)
    print("LEVEL 3 — MobileNet-SSD Detection")
    print("=" * 62)

    env = GridWorld(scene=default_house())
    env.agent.x, env.agent.y, env.agent.facing = 2, 5, 1

    print("\n[1] Rendering sketch-style view...")
    img = render_sketch(env, include_noise=True)
    plt.imsave("results/figures/mobilenet_view.png", img)
    print("    Saved: results/figures/mobilenet_view.png")

    print("\n[2] Loading MobileNet-SSD...")
    detector = MobileNetDetector()
    print("    Model loaded")

    print("\n[3] Running detection...")
    dets = detector.detect(img)
    print(f"    Raw detections: {len(dets)}")
    for d in dets:
        print(f"      {d['class_name']:<15} → {d['object_type']:<10} "
              f"conf={d['confidence']:.2f} "
              f"pixel=({d['pixel_x']:.0f},{d['pixel_y']:.0f})")

    print("\n[4] Building memory from detections...")
    frame = detections_to_frame(dets, env, step=1)
    memory = SpatialGraph()
    update_memory(memory, frame)
    print(f"    Memory: {len(memory.objects)} objects, "
          f"{len(memory.locations)} locations, "
          f"{len(memory.relations)} relations")
    print("\n" + memory.summary())


if __name__ == "__main__":
    main()
"""Renders the grid world as a top-down RGB image — the agent's camera view."""
from __future__ import annotations
import math
import numpy as np

from ..environment.grid_world import GridWorld, HEADINGS


# ---- Object palette (RGB) — chosen to be maximally distinct ----
OBJECT_COLORS = {
    "Table":       (139,  69,  19),    # brown
    "Counter":     ( 80,  80,  80),    # dark gray
    "Fridge":      ( 30, 144, 255),    # bright blue
    "Sofa":        (139,   0,   0),    # dark red
    "TV":          ( 10,  10,  10),    # near-black
    "CoffeeTable": (255, 165,   0),    # orange
    "Mug":         ( 50, 205,  50),    # lime green
    "Apple":       (255,   0,   0),    # pure red
}

FLOOR_KITCHEN = (200, 200, 200)
FLOOR_LIVING  = (230, 220, 200)
FLOOR_UNKNOWN = (100, 100, 100)

PPC = 20
OBJECT_RADIUS = 8
NOISE_SIGMA = 3.0    # reduced from 8.0 — a real camera is not this noisy


def render(env: GridWorld, include_noise: bool = True) -> np.ndarray:
    """
    Render only the objects in the agent's FOV.
    Returns an (H, W, 3) uint8 RGB array.
    """
    scene = env.scene
    H = scene.height * PPC
    W = scene.width * PPC
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # 1. Paint floor tiles
    for cx in range(scene.width):
        for cy in range(scene.height):
            room = scene.room_at(cx, cy)
            color = (FLOOR_KITCHEN if room == "kitchen"
                     else FLOOR_LIVING if room == "living"
                     else FLOOR_UNKNOWN)
            y0 = (scene.height - 1 - cy) * PPC
            x0 = cx * PPC
            img[y0:y0 + PPC, x0:x0 + PPC] = color

    # 2. Draw visible objects — receptacles first (big), then small objects
    obs = env.observe()
    visible = sorted(
        obs.visible_objects,
        key=lambda o: 0 if o["type"] in _RECEPTACLE_TYPES else 1,
    )

    for o in visible:
        obj_type = o["type"]
        color = OBJECT_COLORS.get(obj_type)
        if color is None:
            continue

        px = o["x"] * PPC + PPC // 2
        py = (scene.height - 1 - o["y"]) * PPC + PPC // 2

        if obj_type in _RECEPTACLE_TYPES:
            radius = PPC // 2 - 1     # almost fill the cell (~9px)
        else:
            radius = 5                # small dot on the surface

        _draw_circle(img, px, py, radius, color)

    # 3. Sensor noise
    if include_noise:
        noise = np.random.normal(0, NOISE_SIGMA, img.shape)
        img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)

    return img


_RECEPTACLE_TYPES = {
    "Table", "Counter", "Fridge", "Sofa", "CoffeeTable", "TV",
}


def _draw_circle(img, cx, cy, r, color):
    H, W = img.shape[:2]
    y0, y1 = max(0, cy - r), min(H, cy + r + 1)
    x0, x1 = max(0, cx - r), min(W, cx + r + 1)
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            if (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r:
                img[yy, xx] = color
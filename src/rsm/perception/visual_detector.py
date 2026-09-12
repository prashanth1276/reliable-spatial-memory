"""A tiny classical-CV detector: color threshold + connected components.

Not a neural network. But it's a real image → detections pipeline:
  input:  (H, W, 3) uint8 image
  output: list of {object_type, pixel_x, pixel_y, n_pixels}
"""
from __future__ import annotations
from typing import List
from collections import deque
import numpy as np

from .renderer import OBJECT_COLORS


def detect_objects(
    img: np.ndarray,
    color_tolerance: int = 15,
    min_blob_size: int = 40,
) -> List[dict]:
    detections: List[dict] = []
    for obj_type, color in OBJECT_COLORS.items():
        target = np.array(color, dtype=int)
        diff = np.abs(img.astype(int) - target)
        mask = (diff < color_tolerance).all(axis=2)
        if not mask.any():
            continue
        components = _connected_components(mask)
        for comp in components:
            if len(comp) < min_blob_size:
                continue
            ys = [p[0] for p in comp]
            xs = [p[1] for p in comp]
            detections.append({
                "object_type": obj_type,
                "pixel_x": sum(xs) / len(xs),
                "pixel_y": sum(ys) / len(ys),
                "n_pixels": len(comp),
            })
    return detections


def _connected_components(mask: np.ndarray) -> List[List[tuple]]:
    """4-connectivity BFS labeling. Pure Python + numpy, no scipy."""
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []

    for sy in range(H):
        for sx in range(W):
            if not mask[sy, sx] or visited[sy, sx]:
                continue
            q = deque([(sy, sx)])
            visited[sy, sx] = True
            comp = []
            while q:
                y, x = q.popleft()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < H and 0 <= nx < W
                            and mask[ny, nx] and not visited[ny, nx]):
                        visited[ny, nx] = True
                        q.append((ny, nx))
            components.append(comp)

    return components
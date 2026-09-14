"""CLIP-based grounding: match command text to candidate objects."""
from __future__ import annotations
import os
import numpy as np
from typing import List, Optional, Tuple

import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:
    from sentence_transformers import SentenceTransformer
    _CLIP_OK = True
except ImportError:
    _CLIP_OK = False

from ..environment.grid_world import GridWorld
from .candidates import Candidate


class CLIPGrounder:
    """
    Grounds a natural-language command to a specific candidate object.

    Strategy: for each candidate, crop its region from the rendered frame
    and compute CLIP similarity with the command text. Pick highest.
    """
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        if not _CLIP_OK:
            raise ImportError(
                "sentence-transformers not installed. Run: "
                "pip install sentence-transformers"
            )
        self.model = SentenceTransformer(model_name)

    def score(
        self,
        command: str,
        candidates: List[Candidate],
        env: GridWorld,
    ) -> List[Tuple[Candidate, float]]:
        """
        Returns list of (candidate, similarity_score), sorted descending.
        """
        if not candidates:
            return []

        from ..perception.renderer import PPC
        from ..perception.renderer import OBJECT_COLORS, _RECEPTACLE_TYPES, _draw_circle
        import numpy as np

        # ---- Global render: draw ALL objects, ignoring FOV ----
        # CLIP needs to see every candidate to score it meaningfully.
        scene = env.scene
        H = scene.height * PPC
        W = scene.width * PPC
        frame = np.full((H, W, 3), 230, dtype=np.uint8)  # light grey floor

        for obj in scene.objects.values():
            color = OBJECT_COLORS.get(obj.object_type)
            if color is None:
                continue
            px = int(obj.x * PPC + PPC // 2)
            py = int((scene.height - 1 - obj.y) * PPC + PPC // 2)
            if obj.object_type in _RECEPTACLE_TYPES:
                r = PPC // 2 - 1
            else:
                r = 5
            _draw_circle(frame, px, py, r, color)

        text_emb = self.model.encode(command, convert_to_numpy=True,
                                      normalize_embeddings=True)

        results: List[Tuple[Candidate, float]] = []
        for c in candidates:
            loc = env.get_object(c.location_id)
            if loc is None:
                continue
            # Crop a region around the location
            cx = int(loc.x * PPC)
            cy = int((env.scene.height - 1 - loc.y) * PPC)
            r = 24
            y0, y1 = max(0, cy - r), min(H, cy + r)
            x0, x1 = max(0, cx - r), min(W, cx + r)
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue

            # Resize to CLIP's expected input (or let ST handle it)
            img_emb = self.model.encode(
                _to_pil(crop), convert_to_numpy=True,
                normalize_embeddings=True,
            )
            sim = float(np.dot(text_emb, img_emb))
            results.append((c, sim))

        results.sort(key=lambda x: -x[1])
        return results

    def ground(
        self,
        command: str,
        candidates: List[Candidate],
        env: GridWorld,
    ) -> Optional[Candidate]:
        scored = self.score(command, candidates, env)
        return scored[0][0] if scored else None


def _to_pil(arr: np.ndarray):
    from PIL import Image
    return Image.fromarray(arr.astype(np.uint8))
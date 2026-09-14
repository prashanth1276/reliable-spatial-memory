"""Language-guided agent: parse → retrieve → ground → navigate → verify."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from ..environment.grid_world import GridWorld
from ..memory.spatial_graph import SpatialGraph
from ..memory.updater import update_memory
from ..perception.observation import Frame, ObjectObservation
from ..perception.noise_model import PerceptionNoise, NoiseConfig
from ..agent.navigator import navigate_to_observe
from ..verification.conflict_detector import detect_conflicts, Conflict
from ..verification.verifier import decide, apply_decision, Decision, VerificationResult
from .parser import parse_command, ParsedCommand
from .candidates import get_candidates, Candidate
from .grounder import CLIPGrounder


@dataclass
class LanguageResult:
    command: str
    parsed: ParsedCommand
    n_candidates: int
    chosen_object: Optional[str]
    grounding_score: float
    grounding_correct: bool
    navigation_success: bool
    verification_fired: bool
    reobservations: int


class LanguageAgent:
    """Wraps Project 1's memory + verifier with language grounding."""

    def __init__(
        self,
        env: GridWorld,
        memory: SpatialGraph,
        grounder: Optional[CLIPGrounder] = None,
        noise: Optional[PerceptionNoise] = None,
        use_memory_verification: bool = True,
    ):
        self.env = env
        self.memory = memory
        self.grounder = grounder
        self.noise = noise
        self.use_memory_verification = use_memory_verification

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def execute(
        self,
        command: str,
        ground_truth_object_id: Optional[str],
        frame_builder=None,
        location_ids: Optional[List[str]] = None,
    ) -> LanguageResult:
        parsed = parse_command(command)
        candidates = get_candidates(self.memory, parsed)

        if not candidates:
            return LanguageResult(
                command=command, parsed=parsed, n_candidates=0,
                chosen_object=None, grounding_score=0.0,
                grounding_correct=(ground_truth_object_id is None),
                navigation_success=False,
                verification_fired=False, reobservations=0,
            )

        chosen, score = self._ground(command, candidates)
        if chosen is None:
            return LanguageResult(
                command=command, parsed=parsed,
                n_candidates=len(candidates),
                chosen_object=None, grounding_score=0.0,
                grounding_correct=(ground_truth_object_id is None),
                navigation_success=False,
                verification_fired=False, reobservations=0,
            )

        grounding_correct = (chosen.object_id == ground_truth_object_id)

        # ---- Attempt loop: navigate → observe → verify → retry ----
        max_attempts = 3 if self.use_memory_verification else 1
        verification_fired = False
        reobs_total = 0
        nav_success = False

        for attempt in range(max_attempts):
            loc_id = self.memory.get_location_of(chosen.object_id)
            if loc_id is None:
                break
            loc_obj = self.env.get_object(loc_id)
            if loc_obj is None:
                break

            # Navigate to where memory says the target is
            navigate_to_observe(self.env, loc_obj.x, loc_obj.y)

            # Check whether target is visible from here
            obs = self.env.observe()
            visible_ids = {o["objectId"] for o in obs.visible_objects}
            if chosen.object_id in visible_ids:
                nav_success = True
                break

            # Not visible → verify from this position (scan 360°)
            if self.use_memory_verification and attempt < max_attempts - 1:
                fired, reobs = self._verify_target(chosen)
                verification_fired = verification_fired or fired
                reobs_total += reobs
            else:
                break

        return LanguageResult(
            command=command, parsed=parsed, n_candidates=len(candidates),
            chosen_object=chosen.object_id, grounding_score=score,
            grounding_correct=grounding_correct,
            navigation_success=nav_success,
            verification_fired=verification_fired,
            reobservations=reobs_total,
        )

    # ------------------------------------------------------------------
    def _ground(self, command: str, candidates: List[Candidate]):
        if self.grounder is None:
            # Fallback: pick highest confidence candidate
            return candidates[0], 1.0
        scored = self.grounder.score(command, candidates, self.env)
        if not scored:
            return candidates[0], 0.0
        return scored[0]

    def _frame_from_obs(self, obs) -> Frame:
        """Convert a raw grid-world observation to a perception Frame."""
        objects = [
            ObjectObservation(
                object_id=o["objectId"], object_type=o["type"],
                x=float(o["x"]), y=float(o["y"]),
                room_id=o.get("room_id", ""),
                parent_receptacle_id=o.get("parent_id") or None,
                distance=o.get("distance"), visible=True, confidence=0.9,
            )
            for o in obs.visible_objects
        ]
        return Frame(
            step=self.memory.step + 1,
            agent_x=obs.agent_x, agent_y=obs.agent_y,
            agent_facing=obs.agent_facing,
            objects=objects,
        )

    def _verify_target(self, chosen: Candidate, n_samples_per_dir: int = 2):
        """
        Scan 360° from current position, vote on the target's location.
        Called AFTER the agent has navigated to the remembered location,
        so the target may be visible in a new place.
        """
        scans = []
        for _ in range(4):  # rotate through 4 orientations
            for _ in range(n_samples_per_dir):
                obs = self.env.observe()
                frame = self._frame_from_obs(obs)
                if self.noise is not None:
                    frame.objects = self.noise.corrupt(
                        frame.objects, list(self.env.scene.objects.keys()),
                    )
                scans.append(frame)
            self.env.step("TURN_RIGHT")

        n_scans = len(scans)
        votes: dict = {}
        for frame in scans:
            for o in frame.objects:
                if o.object_id == chosen.object_id and o.parent_receptacle_id:
                    votes[o.parent_receptacle_id] = (
                        votes.get(o.parent_receptacle_id, 0.0) + o.confidence
                    )
                    break

        mem_loc = self.memory.get_location_of(chosen.object_id)

        # ---- Target never seen → disappeared ----
        if not votes:
            if mem_loc is not None:
                conflict = Conflict(
                    object_id=chosen.object_id,
                    memory_location=mem_loc,
                    observed_location=None,
                    memory_confidence=0.9,
                    observed_confidence=0.7,
                    kind="disappeared",
                )
                apply_decision(
                    VerificationResult(conflict, Decision.ACCEPT_NEW,
                                       "scan-absent"),
                    self.memory, self.memory.step + 1,
                )
            return True, n_scans

        # ---- Target seen somewhere → compare to memory ----
        best = max(votes, key=votes.get)
        if best == mem_loc:
            return False, n_scans   # memory confirmed

        conflict = Conflict(
            object_id=chosen.object_id,
            memory_location=mem_loc,
            observed_location=best,
            memory_confidence=0.9,
            observed_confidence=votes[best] / n_scans,
            kind="moved",
        )
        apply_decision(
            VerificationResult(conflict, Decision.ACCEPT_NEW, "scan-moved"),
            self.memory, self.memory.step + 1,
        )
        return True, n_scans
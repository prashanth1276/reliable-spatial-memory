"""Agent policies: how the agent decides what to do on a conflict."""
from __future__ import annotations
from typing import Callable, List

from ..memory.spatial_graph import SpatialGraph
from ..perception.observation import Frame
from ..verification.conflict_detector import detect_conflicts
from ..verification.verifier import (
    decide, apply_decision, Decision, VerificationResult,
)


def _conflicts_for(memory, frame, noise):
    return detect_conflicts(
        memory, frame,
        miss_rate=noise.config.miss_rate if noise else 0.12,
    )


# ---------------- Baselines ----------------

def policy_trust_memory(env, memory, frame, scenario, noise, location_ids):
    return {"conflicts": 0, "reobservations": 0}


def policy_trust_observation(env, memory, frame, scenario, noise, location_ids):
    conflicts = _conflicts_for(memory, frame, noise)
    for c in conflicts:
        apply_decision(
            VerificationResult(c, Decision.ACCEPT_NEW, "naive"),
            memory, memory.step + 1,
        )
    return {"conflicts": len(conflicts), "reobservations": 0}


def policy_gated_observation(env, memory, frame, scenario, noise,
                              location_ids, min_conf=0.7):
    """Only accept observations above a confidence threshold."""
    conflicts = _conflicts_for(memory, frame, noise)
    accepted = 0
    for c in conflicts:
        if c.observed_confidence >= min_conf:
            apply_decision(
                VerificationResult(c, Decision.ACCEPT_NEW, "gated"),
                memory, memory.step + 1,
            )
            accepted += 1
    return {"conflicts": len(conflicts), "reobservations": 0}


# ---------------- Ablations ----------------

def policy_verify_no_reobserve(env, memory, frame, scenario, noise,
                                location_ids):
    """Ablation: no re-observation. Accept any credible conflict."""
    conflicts = _conflicts_for(memory, frame, noise)
    for c in conflicts:
        apply_decision(
            VerificationResult(c, Decision.ACCEPT_NEW, "no-reobserve"),
            memory, memory.step + 1,
        )
    return {"conflicts": len(conflicts), "reobservations": 0}


# ---------------- Full method ----------------

def policy_verify(env, memory, frame, scenario, noise, location_ids,
                  n_samples: int = 5):
    """Full method: multi-sample confidence-weighted voting."""
    conflicts = _conflicts_for(memory, frame, noise)
    reobs = 0

    for c in conflicts:
        votes = {}
        absences = 0
        for _ in range(n_samples):
            obs = env.observe()
            f = _build_noisy_frame(obs, memory.step + 1, noise, location_ids)
            found = False
            for o in f.objects:
                if o.object_id == c.object_id:
                    found = True
                    if o.parent_receptacle_id:
                        votes[o.parent_receptacle_id] = (
                            votes.get(o.parent_receptacle_id, 0.0) + o.confidence
                        )
                    break
            if not found:
                absences += 1
            reobs += 1

        if c.kind == "disappeared":
            presence = sum(votes.values())
            if presence > n_samples * 0.4:
                best = max(votes, key=votes.get)
                if best != c.memory_location:
                    c.observed_location = best
                    c.kind = "moved"
                    apply_decision(
                        VerificationResult(c, Decision.ACCEPT_NEW, "vote"),
                        memory, memory.step + 1,
                    )
                else:
                    apply_decision(
                        VerificationResult(c, Decision.KEEP_OLD, "vote"),
                        memory, memory.step + 1,
                    )
            else:
                apply_decision(
                    VerificationResult(c, Decision.ACCEPT_NEW, "absent"),
                    memory, memory.step + 1,
                )
        else:  # moved
            if not votes:
                apply_decision(
                    VerificationResult(c, Decision.KEEP_OLD, "no-evidence"),
                    memory, memory.step + 1,
                )
            else:
                best = max(votes, key=votes.get)
                best_c = votes[best]
                mem_c = votes.get(c.memory_location, 0.0)

                # Majority favors the current belief → keep it.
                if best == c.memory_location and best_c > 0:
                    apply_decision(
                        VerificationResult(c, Decision.KEEP_OLD, "vote-old"),
                        memory, memory.step + 1,
                    )
                # Majority favors a new location (regardless of whether it
                # matches the initial observation) → accept it.
                elif best_c > mem_c and best_c >= 1.0:
                    c.observed_location = best
                    c.observed_confidence = best_c / n_samples
                    apply_decision(
                        VerificationResult(c, Decision.ACCEPT_NEW, "vote-new"),
                        memory, memory.step + 1,
                    )
                else:
                    apply_decision(
                        VerificationResult(c, Decision.MARK_UNCERTAIN, "split"),
                        memory, memory.step + 1,
                    )

    return {"conflicts": len(conflicts), "reobservations": reobs}


def _build_noisy_frame(obs, step, noise, location_ids):
    from ..perception.observation import Frame, ObjectObservation
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
    if noise:
        objects = noise.corrupt(objects, location_ids)
    return Frame(step=step, agent_x=obs.agent_x, agent_y=obs.agent_y,
                 agent_facing=obs.agent_facing, objects=objects)


POLICIES = {
    "trust_memory":        policy_trust_memory,
    "trust_observation":   policy_trust_observation,
    "gated_observation":   policy_gated_observation,
    "verify_no_reobserve": policy_verify_no_reobserve,
    "verify":              policy_verify,
}
"""Full demo: explore → remember → environment changes → verify → update.

Runs TWO agents on the same scenario side by side:
  (A) A baseline agent that always trusts memory.
  (B) Our verifying agent.

Run:  python scripts\run_full_demo.py
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
from rsm.verification.conflict_detector import detect_conflicts
from rsm.verification.verifier import decide, apply_decision, Decision


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def to_frame(obs, step):
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
        agent_x=obs.agent_x, agent_y=obs.agent_y,
        agent_facing=obs.agent_facing,
        objects=objects,
    )


def explore(env, memory, n_steps=150, seed=42):
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
        update_memory(memory, to_frame(obs, step=i + 1))


def place_agent_to_see(env, target_x, target_y):
    """Teleport the agent so target is within its FOV, facing east."""
    env.agent.x = max(0, target_x - 2)
    env.agent.y = target_y
    env.agent.facing = 1  # east


# ---------------------------------------------------------------------
# The two agents
# ---------------------------------------------------------------------
def agent_always_trusts_memory(env, memory):
    """Baseline: never detect conflicts, never update."""
    obs = env.observe()
    update_memory(memory, to_frame(obs, step=memory.step + 1))
    return "Baseline: trusts memory unconditionally."


def agent_verifies(env, memory):
    """Ours: detect conflict → decide → re-observe if needed → commit."""
    obs = env.observe()
    frame = to_frame(obs, step=memory.step + 1)

    conflicts = detect_conflicts(memory, frame)
    log_lines = []

    for c in conflicts:
        result = decide(c, memory)
        log_lines.append(
            f"    [conflict:{c.kind}] {c.object_id} "
            f"mem={c.memory_location} obs={c.observed_location} → "
            f"{result.decision.value} ({result.reason})"
        )

        # ---- Handle RE_OBSERVE: gather a second opinion, then COMMIT ----
        if result.decision == Decision.RE_OBSERVE:
            # Second observation from the same pose (a real "look again").
            obs2 = env.observe()
            frame2 = to_frame(obs2, step=memory.step + 1)
            second = detect_conflicts(memory, frame2)

            if second and second[0].observed_location == c.observed_location:
                log_lines.append(
                    "    [re-observe] second look reproduces the same conflict "
                    "→ trusting new observation"
                )
                result.conflict = second[0]
            else:
                log_lines.append(
                    "    [re-observe] second look did not reproduce the "
                    "conflict → still trusting the direct observation"
                )
            # Re-observation is a transitional state; commit to a
            # resolution. The direct observation was strong, so accept it.
            result.decision = Decision.ACCEPT_NEW

        apply_decision(result, memory, memory.step + 1)

    # Also freshen memory with what we just saw
    update_memory(memory, frame)
    return "\n".join(log_lines) if log_lines else "    (no conflicts detected)"


# ---------------------------------------------------------------------
# Main scenario
# ---------------------------------------------------------------------
def run_scenario(agent_fn, agent_name):
    print("\n" + "=" * 62)
    print(f"RUNNING: {agent_name}")
    print("=" * 62)

    env = GridWorld(scene=default_house())
    memory = SpatialGraph()

    # Phase 1: explore
    explore(env, memory, n_steps=150, seed=42)

    print("\n[Phase 1] Initial memory (after exploration)")
    print(f"  Agent believes mug is at: {memory.get_location_of('mug_1')}")

    # Phase 2: environment changes
    print("\n[Phase 2] ENVIRONMENT CHANGE: someone moves the mug "
          "from the table to the counter")
    counter = env.get_object("counter_1")
    env.move_object("mug_1", counter.x, counter.y, parent_id="counter_1")
    print(f"  Mug is now physically at ({counter.x},{counter.y}) on counter_1")

    # Phase 3: agent returns and observes
    print("\n[Phase 3] Agent returns and observes the kitchen")
    place_agent_to_see(env, counter.x, counter.y)

    # Phase 4: agent acts
    log = agent_fn(env, memory)
    print("\n[Phase 4] Agent response")
    print(log)

    # Phase 5: what does memory believe now?
    print("\n[Phase 5] What does the agent's memory believe now?")
    print(f"  mug_1 → {memory.get_location_of('mug_1')}")

    return memory.get_location_of("mug_1")


def main():
    print("=" * 62)
    print("FULL DEMO — Scenario B: object moved")
    print("=" * 62)

    baseline_loc = run_scenario(agent_always_trusts_memory,
                                "BASELINE — always trusts memory")
    ours_loc = run_scenario(agent_verifies,
                            "OURS — verify on conflict")

    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    print(f"  Ground truth: mug is on counter_1")
    print(f"  Baseline believes mug is on: {baseline_loc}  "
          f"({'CORRECT' if baseline_loc == 'counter_1' else 'WRONG'})")
    print(f"  Ours     believes mug is on: {ours_loc}  "
          f"({'CORRECT' if ours_loc == 'counter_1' else 'WRONG'})")
    print("=" * 62)


if __name__ == "__main__":
    main()
"""Task: navigate to an object using spatial memory (ground-truth evaluation)."""
from __future__ import annotations
from typing import Callable

from ..environment.grid_world import GridWorld
from ..memory.spatial_graph import SpatialGraph
from ..agent.navigator import navigate_to_observe
from ..evaluation.scenarios import ScenarioType


def run_navigate_task(
    env: GridWorld,
    memory: SpatialGraph,
    target_id: str,
    frame_builder: Callable,
    policy_fn: Callable,
    scenario,
    noise,
    location_ids: list,
) -> dict:
    """
    Full task:
      Phase 1 — enter scene, observe, policy may update memory
      Phase 2 — navigate to target's MEMORY location
      Phase 3 — evaluate:
                  * task_success:    agent is within 2 cells of ground truth
                  * memory_correct:  memory belief matches ground truth
                                     (for disappeared: memory says target is gone)
    """
    reobs = 0
    conflicts = 0

    # ---------------- Phase 1: enter scene from a FIXED position ----------------
    # The agent always starts at the kitchen entrance. It does not know
    # where the target is. It observes from here, then navigates using memory.
    env.agent.x, env.agent.y, env.agent.facing = 3, 5, 1

    obs = env.observe()
    frame = frame_builder(obs, memory.step + 1, noise, location_ids)
    r = policy_fn(env, memory, frame, scenario, noise, location_ids)
    conflicts += r.get("conflicts", 0)
    reobs += r.get("reobservations", 0)

    # ---------------- Phase 2: navigate using MEMORY ----------------
    mem_loc_id = memory.get_location_of(target_id)
    if mem_loc_id is not None:
        loc_obj = env.get_object(mem_loc_id)
        if loc_obj is not None:
            nav_steps = navigate_to_observe(env, loc_obj.x, loc_obj.y)
        else:
            nav_steps = 0
    else:
        nav_steps = 0

    # ---------------- Phase 3: ground-truth evaluation ----------------
    final_belief = memory.get_location_of(target_id)

    if scenario.scenario_type == ScenarioType.DISAPPEARED:
        # success = agent correctly believes the target is gone
        task_success = (final_belief is None)
        memory_correct = (final_belief is None)
    else:
        gt_loc = env.get_object(scenario.ground_truth_location)
        if gt_loc is None:
            task_success = False
        else:
            d = (abs(env.agent.x - int(gt_loc.x))
                 + abs(env.agent.y - int(gt_loc.y)))
            task_success = (d <= 2)
        memory_correct = (final_belief == scenario.ground_truth_location)

    return {
        "success": int(task_success),
        "memory_correct": int(memory_correct),
        "attempts": 1,
        "nav_steps": nav_steps,
        "conflicts": conflicts,
        "reobservations": reobs,
    }
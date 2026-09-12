"""Simple greedy navigation on the grid."""
from __future__ import annotations

from ..environment.grid_world import GridWorld


def navigate_to_observe(
    env: GridWorld,
    target_x: int,
    target_y: int,
    max_steps: int = 120,
) -> int:
    """
    Move the agent to a pose that faces (target_x, target_y) from ~2 cells away.
    Returns the number of actions taken.
    """
    tx, ty = int(target_x), int(target_y)

    # Choose observation pose
    if tx - 2 >= 0:
        obs_x, obs_y, facing = tx - 2, ty, 1     # stand west, face east
    else:
        obs_x, obs_y, facing = tx + 2, ty, 3     # stand east, face west

    steps = 0

    # Greedy walk toward (obs_x, obs_y)
    while (env.agent.x, env.agent.y) != (obs_x, obs_y) and steps < max_steps:
        dx = obs_x - env.agent.x
        dy = obs_y - env.agent.y
        if dx != 0:
            desired = 1 if dx > 0 else 3
        else:
            desired = 0 if dy > 0 else 2

        if env.agent.facing != desired:
            diff = (desired - env.agent.facing) % 4
            env.step("TURN_RIGHT" if diff <= 2 else "TURN_LEFT")
        else:
            prev = (env.agent.x, env.agent.y)
            env.step("MOVE_FORWARD")
            if (env.agent.x, env.agent.y) == prev:
                env.step("TURN_RIGHT")
        steps += 1

    # Face target
    while env.agent.facing != facing and steps < max_steps:
        diff = (facing - env.agent.facing) % 4
        env.step("TURN_RIGHT" if diff <= 2 else "TURN_LEFT")
        steps += 1

    return steps
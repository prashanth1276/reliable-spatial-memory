"""Grounding + verification policies for the language agent."""
from __future__ import annotations
from typing import Optional

from ..language_agent import LanguageAgent
from .commands import CommandTest


POLICIES = [
    "full",              # CLIP + verified memory
    "no_verification",   # CLIP + trust memory
    "no_clip",           # Highest-confidence candidate (no CLIP)
    "no_memory",         # CLIP but only visible objects (no spatial memory)
]


def run_policy(
    policy: str,
    agent: LanguageAgent,
    command: CommandTest,
    env,
):
    """
    Returns LanguageResult. Different policies toggle agent capabilities.
    """
    # Configure agent for the policy
    orig_verification = agent.use_memory_verification
    orig_grounder = agent.grounder

    if policy == "full":
        agent.use_memory_verification = True
        agent.grounder = orig_grounder
    elif policy == "no_verification":
        agent.use_memory_verification = False
        agent.grounder = orig_grounder
    elif policy == "no_clip":
        agent.use_memory_verification = True
        agent.grounder = None       # falls back to highest-confidence
    elif policy == "no_memory":
        # Use CLIP but empty out the memory so candidates come only from
        # currently visible objects
        agent.use_memory_verification = False
        agent.grounder = orig_grounder
        # (Handled by caller — memory is temporarily swapped)

    result = agent.execute(
        command=command.text,
        ground_truth_object_id=command.ground_truth_object_id,
    )

    agent.use_memory_verification = orig_verification
    agent.grounder = orig_grounder
    return result
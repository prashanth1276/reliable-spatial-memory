"""Test command set with known ground truth."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CommandTest:
    text: str
    ground_truth_object_id: str
    note: str = ""


def default_commands() -> list[CommandTest]:
    # Spatial-relation commands ("mug on the table") are excluded because
    # they are contradictory on the "moved" scenario: once the mug moves,
    # the phrase no longer describes a valid object, causing grounding to
    # fail even when memory is correct.
    return [
        # Simple
        CommandTest("Go to the mug", "mug_1", "simple"),
        CommandTest("Find the apple", "apple_1", "simple"),

        # Color, single-candidate (blue mug = mug_1)
        CommandTest("Go to the blue mug", "mug_1", "color"),
        CommandTest("Find the red apple", "apple_1", "color"),

        # Color, two-candidate disambiguation (CLIP has to pick!)
        CommandTest("Go to the red mug", "mug_2", "disambiguation"),
        CommandTest("Find the red cup", "mug_2", "disambiguation"),

        # Paraphrases
        CommandTest("Could you reach the cup", "mug_1", "paraphrase"),
        CommandTest("I want you to locate the apple", "apple_1", "paraphrase"),
        CommandTest("Please head to the mug", "mug_1", "paraphrase"),

        # Adversarial color (nonexistent color — should still pick by type)
        CommandTest("Go to the green mug", "mug_1", "adversarial_color"),
        CommandTest("Find the purple apple", "apple_1", "adversarial_color"),
    ]
"""Aggregate metrics for a batch of commands."""
from __future__ import annotations
from collections import defaultdict
from typing import List
from ..language_agent import LanguageResult


def aggregate(results: List[LanguageResult]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    return {
        "n": n,
        "parse_rate":     sum(r.parsed.parse_succeeded for r in results) / n,
        "grounding_acc":  sum(r.grounding_correct for r in results) / n,
        "navigation_acc": sum(r.navigation_success for r in results) / n,
        "end_to_end":     sum(r.grounding_correct and r.navigation_success
                              for r in results) / n,
        "verif_fired":    sum(r.verification_fired for r in results) / n,
        "avg_reobs":      sum(r.reobservations for r in results) / n,
    }


def by_category(results: List[LanguageResult], commands) -> dict:
    """Group results by command category (note field)."""
    buckets = defaultdict(list)
    for r, c in zip(results, commands):
        buckets[c.note or "misc"].append(r)
    return {k: aggregate(v) for k, v in buckets.items()}
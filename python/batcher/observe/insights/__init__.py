"""Automatic findings for one run — what is wrong, the evidence, and what to do.

A rule is a function of `(profile, ops, total_ms)` returning zero or more `Insight`s, so the
registry below is the whole dispatch: adding a rule is one function and one line. Rules are
grouped by what they read — `resources` looks at the machine, `planning` at how work was
distributed, `dataflow` at row counts — because that grouping is also what decides whose
problem a finding is.
"""

from __future__ import annotations

from typing import Any

from .dataflow import exploding_join, late_filter, wide_scan
from .kinds import Insight
from .planning import bad_estimates, dominant_operator, long_tail, planning_dominates
from .resources import (
    idle_cpu,
    memory_headroom,
    memory_underused,
    spilled_operators,
)

__all__ = ["derive_insights"]

#: Every rule, in no significant order — the result is sorted by severity, not by registry
#: position, so a rule can be added anywhere.
_RULES = (
    spilled_operators,
    bad_estimates,
    dominant_operator,
    idle_cpu,
    memory_headroom,
    memory_underused,
    wide_scan,
    exploding_join,
    late_filter,
    long_tail,
    planning_dominates,
)


def derive_insights(profile: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Findings for one query's profile, most severe first; `[]` for a healthy run.

    Args:
        profile: A `QueryProfile.to_dict()` document, or None if the query never produced
            one (a metadata-answered fast path, or a streaming query).

    Returns:
        A list of insight dicts, ordered critical -> warning -> info.
    """
    if not profile:
        return []
    ops = [op for op in profile.get("ops", []) if op.get("measured")]
    if not ops:
        return []
    total_ms = float(profile.get("total_ms", 0.0))
    found: list[Insight] = []
    for rule in _RULES:
        found.extend(rule(profile, ops, total_ms))
    order = {"critical": 0, "warning": 1, "info": 2}
    found.sort(key=lambda i: order.get(i.severity, 3))
    return [insight.to_dict() for insight in found]

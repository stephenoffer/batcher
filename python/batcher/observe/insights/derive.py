"""`derive_insights` — run every insight rule over one profile and rank what they find.

A rule is a function of ``(profile, ops, total_ms)`` returning zero or more `Insight`s, so the
registry here is the whole dispatch: adding a rule is one function and one line. The result is
sorted by severity rather than by registry position, so a rule can be added anywhere.
"""

from __future__ import annotations

from typing import Any

from batcher.observe.insights.dataflow import exploding_join, late_filter, wide_scan
from batcher.observe.insights.devices import derated_host_link, device_bottleneck
from batcher.observe.insights.kinds import Insight
from batcher.observe.insights.planning import (
    bad_estimates,
    dominant_operator,
    long_tail,
    planning_dominates,
)
from batcher.observe.insights.resources import (
    idle_cpu,
    memory_headroom,
    memory_underused,
    paging_operators,
    preempted_operators,
    spilled_operators,
)
from batcher.observe.insights.stages import (
    gpu_starved,
    per_row_map,
    row_exploding_stage,
    udf_dominates,
)

__all__ = ["derive_insights"]

#: Every rule, in no significant order — the result is sorted by severity, not by registry
#: position, so a rule can be added anywhere.
_RULES = (
    paging_operators,
    preempted_operators,
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
    # ML-pipeline stage findings. They read the orchestrator's per-stage measurements
    # rather than the engine's, which are the only numbers a batch-inference pipeline has.
    gpu_starved,
    udf_dominates,
    row_exploding_stage,
    per_row_map,
    # Findings about the *devices* rather than the plan. They read the sampling window and the
    # live link geometry, not the profile, because none of what they detect changes what the
    # plan did — a clamped device and a derated slot produce a profile identical to a healthy
    # run's, with larger numbers in it.
    device_bottleneck,
    derated_host_link,
)


def derive_insights(profile: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Findings for one query's profile, most severe first; `[]` for a healthy run.

    On a **distributed** run the driver tree carries no measurements — the work happened on
    the workers, and their map sub-plan arrives as `worker_ops` in its own op-id space. Rules
    read that instead when the driver tree is empty. Without the fallback every rule went
    silent on exactly the runs that most need them: a spill or a starved GPU on a cluster is
    both more likely and more expensive than the same finding on one node.

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
        ops = [op for op in profile.get("worker_ops", []) if op.get("measured")]
    if not ops:
        return []
    total_ms = float(profile.get("total_ms", 0.0))
    found: list[Insight] = []
    for rule in _RULES:
        found.extend(rule(profile, ops, total_ms))
    order = {"critical": 0, "warning": 1, "info": 2}
    found.sort(key=lambda i: order.get(i.severity, 3))
    return [insight.to_dict() for insight in found]

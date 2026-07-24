"""Findings about the machine: memory, spill, and CPU the run did or did not get.

These share a shape — the plan may be fine and the environment wrong — and their advice
points at configuration rather than at the query."""

from __future__ import annotations

import pathlib
from typing import Any

from batcher.observe.insights.kinds import (
    _CPU_IDLE,
    _LOAD_CONTENDED,
    _MEMORY_IDLE,
    _MEMORY_TIGHT,
    _THROTTLED_HIGH,
    _TRIVIAL_MS,
    Insight,
    gib,
)


def cpu_contention() -> dict[str, float]:
    """Run-queue length per core and CFS throttling, when the platform reports them.

    Empty when unavailable — the caller treats a missing reading as "cannot tell", never as
    "not contended", because blaming the plan for a busy box sends the reader to tune a query
    that was already fine.
    """
    out: dict[str, float] = {}
    try:
        import os

        from batcher._internal.hardware import available_cpu_count

        load1, _, _ = os.getloadavg()
        # The cores this process may actually use (cgroup quota ∧ affinity), not the host's.
        # `os.cpu_count()` reports the whole machine, so a container limited to 4 of 64 cores
        # divided a saturating load of 4.0 by 64 and reported 0.06 — "idle" at exactly the
        # moment it was pegged and throttled, which is the reading this metric exists to catch.
        out["load_per_core"] = load1 / max(1, available_cpu_count())
    except (OSError, AttributeError):
        pass  # no getloadavg on this platform -> omit the metric rather than guess one
    try:
        stat = pathlib.Path("/sys/fs/cgroup/cpu.stat").read_text()
        fields = dict(line.split(maxsplit=1) for line in stat.splitlines() if " " in line)
        periods = float(fields.get("nr_periods", 0))
        throttled = float(fields.get("nr_throttled", 0))
        if periods > 0:
            out["throttled_share"] = throttled / periods
    except (OSError, ValueError):
        pass  # not running under a cgroup v2 CPU controller -> omit the throttling metric
    return out


__all__ = ["idle_cpu", "memory_headroom", "memory_underused", "spilled_operators", "stolen_cpu"]


def spilled_operators(
    _profile: dict[str, Any], ops: list[dict[str, Any]], _total_ms: float
) -> list[Insight]:
    """Spilling means the working set did not fit; name the volume and the operator."""
    spilled = [op for op in ops if op.get("spilled")]
    if not spilled:
        return []
    volume = sum(int(op.get("spill_bytes", 0)) for op in spilled)
    names = ", ".join(sorted({str(op.get("kind", "?")) for op in spilled}))
    return [
        Insight(
            severity="warning",
            rule="operator-spilled",
            title="Spilled to disk",
            evidence=(
                f"{len(spilled)} operator(s) spilled {gib(volume)} — {names}. Spilling keeps "
                "the query alive under a bounded envelope, but it trades memory for disk I/O."
            ),
            action=(
                "Raise memory.max_memory_bytes if headroom exists, or reduce the working set: "
                "filter and project earlier so less data reaches the breaker."
            ),
            op=names,
            detail={"spill_bytes": volume, "operators": len(spilled)},
        )
    ]


def idle_cpu(_profile: dict[str, Any], ops: list[dict[str, Any]], total_ms: float) -> list[Insight]:
    """Idle cores are only a finding when the run was long enough for it to matter."""
    if total_ms < _TRIVIAL_MS:
        return []
    weighted = [op for op in ops if float(op.get("cpu_util", 0.0)) > 0 and op.get("elapsed_ms")]
    if not weighted:
        return []
    held = sum(float(op["elapsed_ms"]) for op in weighted)
    util = sum(float(op["cpu_util"]) * float(op["elapsed_ms"]) for op in weighted) / held
    if util >= _CPU_IDLE:
        return []

    # Same symptom, opposite fixes: cores we never got vs cores we failed to ask for. Blaming
    # the plan for a noisy co-tenant sends the user to tune a query that was already fine.
    contention = cpu_contention()
    detail: dict[str, Any] = {"cpu_util": util, **contention}
    stolen = stolen_cpu(util, contention, detail)
    if stolen:
        return stolen
    load = contention.get("load_per_core")
    return [
        Insight(
            severity="info",
            rule="cpu-underutilized",
            title=f"CPU {util * 100:.0f}% utilized",
            evidence=(
                f"Wall-time-weighted CPU utilization was {util * 100:.0f}% across "
                f"{len(weighted)} measured operator(s); the engine targets >90%."
                + (f" The box was not contended (run queue {load:.1f}x cores)." if load else "")
            ),
            action=(
                "Expected for an I/O-bound scan or a GPU stage. Otherwise the query may be too "
                "small to parallelize, or bounded by one source file — check that the input "
                "splits into more than one morsel."
            ),
            detail=detail,
        )
    ]


def stolen_cpu(util: float, contention: dict[str, float], detail: dict[str, Any]) -> list[Insight]:
    """The cores were taken, not left idle — or `[]` when the machine was actually free.

    Split out of [`_idle_cpu`] because it answers a different question: *whose* fault the idle
    cores are. Both findings outrank the plain under-utilization advice, which would otherwise
    tell a user throttled by their own container limit to go re-partition their input.
    """
    throttled = contention.get("throttled_ratio")
    if throttled is not None and throttled >= _THROTTLED_HIGH:
        return [
            Insight(
                severity="warning",
                rule="cpu-throttled",
                title=f"CPU quota throttled {throttled * 100:.0f}% of periods",
                evidence=(
                    f"CPU utilization was {util * 100:.0f}%, but the cgroup CFS quota throttled "
                    f"{throttled * 100:.0f}% of scheduling periods — the cores were taken away, "
                    "not left idle by the plan."
                ),
                action=(
                    "Raise the container's CPU limit, or size the request to the quota. Tuning "
                    "the query will not help while the quota is binding."
                ),
                detail=detail,
            )
        ]
    load = contention.get("load_per_core")
    if load is not None and load >= _LOAD_CONTENDED:
        return [
            Insight(
                severity="warning",
                rule="cpu-contended",
                title=f"Box oversubscribed ({load:.1f}x cores)",
                evidence=(
                    f"CPU utilization was {util * 100:.0f}%, but the 1-minute run queue was "
                    f"{load:.1f}x the available cores — other work on this machine was competing "
                    "for them."
                ),
                action=(
                    "Re-measure on a quiet box before trusting this timing, or give the process "
                    "its own cpuset. The plan is not the bottleneck here."
                ),
                detail=detail,
            )
        ]
    return []


def memory_headroom(
    profile: dict[str, Any], ops: list[dict[str, Any]], _total_ms: float
) -> list[Insight]:
    """Near the memory ceiling now means spilling or failing on slightly more data."""
    budget = int(profile.get("memory_budget_bytes", 0))
    peak = max((int(op.get("peak_rss_bytes", 0)) for op in ops), default=0)
    if budget <= 0 or peak <= 0:
        return []
    share = peak / budget
    if share < _MEMORY_TIGHT:
        return []
    return [
        Insight(
            severity="warning" if share < 1.0 else "critical",
            rule="memory-headroom",
            title=f"Peak memory {share * 100:.0f}% of budget",
            evidence=f"Peak working set {gib(peak)} against a {gib(budget)} budget.",
            action=(
                "This shape has little headroom — a larger input will spill or fail. Raise "
                "memory.max_memory_bytes, or narrow the working set with an earlier projection."
            ),
            detail={"peak_bytes": peak, "budget_bytes": budget},
        )
    ]


def memory_underused(
    profile: dict[str, Any], ops: list[dict[str, Any]], total_ms: float
) -> list[Insight]:
    """Memory left on the table — and the case where that cost real time.

    The counterpart to [`_memory_headroom`]. An oversized budget is harmless on its own, so
    this stays quiet unless the run was long enough for the headroom to have bought something,
    and escalates only for the genuinely wrong outcome: spilling to disk while most of the
    envelope went unused, which means the *estimate* that chose to spill was wrong, not the
    memory limit.
    """
    budget = int(profile.get("memory_budget_bytes", 0))
    peak = max((int(op.get("peak_rss_bytes", 0)) for op in ops), default=0)
    if budget <= 0 or peak <= 0 or total_ms < _TRIVIAL_MS:
        return []
    share = peak / budget
    if share >= _MEMORY_IDLE:
        return []
    spilled = [op for op in ops if op.get("spilled")]
    if spilled:
        names = ", ".join(sorted({str(op.get("kind", "?")) for op in spilled}))
        return [
            Insight(
                severity="warning",
                rule="spilled-with-headroom",
                title=f"Spilled while using {share * 100:.0f}% of memory",
                evidence=(
                    f"{names} spilled to disk, yet peak working set was {gib(peak)} against a "
                    f"{gib(budget)} budget ({share * 100:.0f}%). The decision to spill is made "
                    "before execution from an estimate, so an over-estimated cardinality spills a "
                    "query that would have fit in memory."
                ),
                action=(
                    "Check the estimate against the actual row count for this operator. Caching "
                    "statistics for this shape, or raising memory.max_memory_bytes, avoids the "
                    "disk round-trip the run did not need."
                ),
                op=names,
                detail={"share": share, "peak_rss_bytes": peak, "memory_budget_bytes": budget},
            )
        ]
    return [
        Insight(
            severity="info",
            rule="memory-underused",
            title=f"Peak memory {share * 100:.0f}% of budget",
            evidence=(
                f"Peak working set {gib(peak)} against a {gib(budget)} budget; the engine "
                "targets >80% so that batches stay large and breakers stay in memory."
            ),
            action=(
                "Nothing is wrong if the query is simply small. If it is not, the budget is "
                "sized off free RAM at query start — a large page cache or a co-tenant shrinks "
                "it. Set memory.max_memory_bytes explicitly to claim the capacity."
            ),
            detail={"share": share, "peak_rss_bytes": peak, "memory_budget_bytes": budget},
        )
    ]

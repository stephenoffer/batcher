"""Findings about the machine: memory, spill, and CPU the run did or did not get.

These share a shape — the plan may be fine and the environment wrong — and their advice
points at configuration rather than at the query."""

from __future__ import annotations

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
from batcher.plan.feedback import CONTENDED_PREEMPTIONS_PER_CORE_SECOND, preemption_rate


def cpu_contention() -> dict[str, float]:
    """How much of the machine's CPU other work was taking, when the platform reports it.

    A thin pass-through to the shared probe in `_internal.hardware`, which is the one place
    that knows how to read these. It used to be a second implementation here, and the copy had
    drifted in two ways that each silently disabled a finding:

    * it emitted ``throttled_share`` while `stolen_cpu` reads ``throttled_ratio``, so the
      `cpu-throttled` insight could never fire. A container throttled to a fraction of its
      quota was told its query "may be too small to parallelize" — advice that sends the
      reader to tune a query that was already fine, which is precisely the misdiagnosis this
      pair of findings exists to prevent; and
    * it read only ``/sys/fs/cgroup/cpu.stat``, the mount root. A process in a *delegated*
      cgroup with no namespace — a Ray worker under a systemd slice, the common deployment —
      has its quota enforced at a leaf the root file knows nothing about, so throttling there
      read as zero.

    Empty when unavailable — the caller treats a missing reading as "cannot tell", never as
    "not contended", because blaming the plan for a busy box sends the reader to tune a query
    that was already fine.

    Returns:
        The measurable contention signals, each omitted when the platform cannot report it.
    """
    from batcher._internal.hardware import cpu_contention as probe

    return probe()


__all__ = [
    "idle_cpu",
    "memory_headroom",
    "memory_underused",
    "paging_operators",
    "preempted_operators",
    "spilled_operators",
    "stale_core_budget",
    "stolen_cpu",
]

#: How far the control plane's core count may differ from the engine's before it is reported.
#: One core is ordinary rounding between a CFS quota and a thread count; a *factor* is a
#: different machine. Expressed as a ratio so it means the same thing at 4 cores and at 128.
_CORE_BUDGET_DRIFT = 1.5


def stale_core_budget(
    _profile: dict[str, Any], _ops: list[dict[str, Any]], _total_ms: float
) -> list[Insight]:
    """The control plane and the data plane disagree about how many cores this machine has.

    Every hardware probe in `_internal.hardware` reads `/proc` and `/sys` from the *Python*
    process and memoizes the answer, because a running process cannot normally see its own
    machine change. One ordinary deployment breaks that assumption: **a cgroup applied after
    the interpreter started**. A Ray worker's CPU quota lands on the actor once it is placed,
    which is after `import batcher` — so anything already memoized describes a machine the
    engine is no longer running on.

    The engine detects its own hardware locally and reports it back, and the two figures were
    never compared. That is the whole failure: when they diverge, the control plane sizes a
    fan-out, a spill threshold and a per-task CPU share for a machine with (say) sixteen cores
    while the data plane runs four, and every symptom of it — throttling, a fan-out that does
    not help, timings that will not reproduce — points somewhere else. Nothing in a profile,
    a log line, or an `EXPLAIN` said the two planes were describing different hardware.

    Reported rather than corrected. The right correction depends on why they differ, and the
    engine's figure is not automatically the one to adopt: it governs how the data plane sized
    itself, but a plan already annotated against the other figure has been shipped. Saying so
    lets an operator fix the cause, which is nearly always placement.

    Args:
        _profile: The query profile; unused, since this describes the machine, not the query.
        _ops: The measured operators; unused for the same reason.
        _total_ms: The query's wall time; unused for the same reason.

    Returns:
        One finding when the two planes disagree by more than `_CORE_BUDGET_DRIFT`, else `[]` —
        including whenever the engine cannot report, which is a shrug and not agreement.
    """
    from batcher._internal.hardware import available_cpu_count, engine_hardware

    engine_cores = int(engine_hardware().get("logical_cores", 0) or 0)
    if engine_cores <= 0:
        return []  # not built, or built before these entry points existed: no comparison to make
    control_cores = available_cpu_count()
    ratio = max(engine_cores, control_cores) / max(1, min(engine_cores, control_cores))
    if ratio < _CORE_BUDGET_DRIFT:
        return []
    return [
        Insight(
            severity="warning",
            rule="core-budget-mismatch",
            title=f"Planned for {control_cores} cores, ran on {engine_cores}",
            evidence=(
                f"The control plane sized this query for {control_cores} usable core(s) while "
                f"the engine process detected {engine_cores}. The usual cause is a cgroup CPU "
                "quota applied to this process after the interpreter started — a Ray worker's "
                "limit lands when the actor is placed, which is after the hardware probes have "
                "already answered and been memoized."
            ),
            action=(
                "Apply the CPU limit before the worker starts, or set the quota on the pod "
                "rather than on the placed actor. Fan-out, spill thresholds and per-task CPU "
                "shares were all sized against the larger figure."
            ),
            detail={"control_plane_cores": control_cores, "engine_cores": engine_cores},
        )
    ]


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
    # Judged on the HOST-scoped ratio, not the slice-relative one. `os.getloadavg()` counts
    # runnable tasks across the whole machine and Linux publishes no per-cgroup equivalent, so
    # `load_per_core` — that host-wide numerator over *this process's* cores — is 16.0 for a
    # 4-core container on a half-idle 128-core host. This rule then told the user their box was
    # sixteen times oversubscribed and to go find a quieter machine, about a machine that was
    # 50% idle, on the single most common deployment shape there is.
    load = contention.get("host_load_per_core", contention.get("load_per_core"))
    if load is not None and load >= _LOAD_CONTENDED:
        share = contention.get("load_per_core")
        return [
            Insight(
                severity="warning",
                rule="cpu-contended",
                title=f"Box oversubscribed ({load:.1f}x cores)",
                evidence=(
                    f"CPU utilization was {util * 100:.0f}%, but the 1-minute run queue was "
                    f"{load:.1f}x this machine's cores — other work on it was competing for "
                    "them."
                    + (
                        f" Relative to this process's own core budget the queue is "
                        f"{share:.1f}x, so most of that work is not ours."
                        if share is not None and share > load * 1.5
                        else ""
                    )
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


def paging_operators(
    _profile: dict[str, Any], ops: list[dict[str, Any]], _total_ms: float
) -> list[Insight]:
    """The machine was paging against the query — the one finding that outranks every other.

    A major page fault is the kernel fetching back memory the process believed it already held.
    While that is happening, every other number on the plan line describes the symptom rather
    than the work: an operator's wall time is storage latency, its CPU utilization is low
    because its threads are blocked, and its measured "peak memory" is whatever survived being
    evicted. Reading those and tuning the query is the wrong move, and nothing else in a
    profile distinguishes this state from ordinary slowness.

    It is also the one condition under which more parallelism strictly hurts: extra threads
    each fault in their own working set and evict each other's.

    Args:
        _profile: The query profile (unused; the evidence is per-operator).
        ops: The measured operators.
        _total_ms: Total wall time (unused).

    Returns:
        One insight when any operator took disk-backed faults, else `[]`.
    """
    faulting = [op for op in ops if int(op.get("major_faults", 0)) > 0]
    if not faulting:
        return []
    total = sum(int(op.get("major_faults", 0)) for op in faulting)
    names = ", ".join(sorted({str(op.get("kind", "?")) for op in faulting}))
    return [
        Insight(
            severity="critical",
            rule="memory-paging",
            title=f"Machine paged {total:,} times during the query",
            evidence=(
                f"{len(faulting)} operator(s) took {total:,} disk-backed page faults — {names}. "
                "The kernel was fetching back memory the process already believed it held, so "
                "the timings above measure storage latency rather than the query's work."
            ),
            action=(
                "Lower memory.max_memory_bytes so the engine spills deliberately instead of "
                "being paged involuntarily, or give the process more RAM. Do not add "
                "parallelism: each extra worker faults in its own working set and evicts the "
                "others."
            ),
            op=names,
            detail={"major_faults": total, "operators": len(faulting)},
        )
    ]


def preempted_operators(
    _profile: dict[str, Any], ops: list[dict[str, Any]], total_ms: float
) -> list[Insight]:
    """The cores were repeatedly taken away mid-operator, per operator rather than per box.

    The machine-wide `cpu-contended` finding reads the run queue, which is an average over the
    whole box across the last minute. This reads involuntary context switches *per operator*,
    which is contention the query actually experienced, during the window it ran, attributed to
    the operator that suffered it. The two disagree exactly when it matters: a short query that
    lands inside someone else's burst sees heavy preemption and a calm-looking load average.

    Reported at `info` because a busy shared box is often a deliberate choice rather than a
    fault. The point is that the timing should not be trusted as a measurement of the plan.

    Args:
        _profile: The query profile (unused; the evidence is per-operator).
        ops: The measured operators.
        total_ms: Total wall time, to skip runs too short to judge.

    Returns:
        One insight when an operator was measurably fighting for cores, else `[]`.
    """
    if total_ms < _TRIVIAL_MS:
        return []
    contended = [
        op
        for op in ops
        if preemption_rate(
            int(op.get("invol_ctx_switches", 0)),
            float(op.get("elapsed_ms", 0.0)),
            int(op.get("threads", 0)),
        )
        >= CONTENDED_PREEMPTIONS_PER_CORE_SECOND
    ]
    if not contended:
        return []
    worst = max(
        contended,
        key=lambda op: preemption_rate(
            int(op.get("invol_ctx_switches", 0)),
            float(op.get("elapsed_ms", 0.0)),
            int(op.get("threads", 0)),
        ),
    )
    rate = preemption_rate(
        int(worst.get("invol_ctx_switches", 0)),
        float(worst.get("elapsed_ms", 0.0)),
        int(worst.get("threads", 0)),
    )
    kind = str(worst.get("kind", "?"))
    return [
        Insight(
            severity="info",
            rule="cpu-preempted",
            title=f"Operators evicted from the CPU ({rate:,.0f}/core-s)",
            evidence=(
                f"{len(contended)} operator(s) were preempted off their cores while running; "
                f"{kind} was worst at {rate:,.0f} involuntary context switches per core-second. "
                "Something else on this machine wanted the cores during the query, which the "
                "1-minute run queue can miss on a short run."
            ),
            action=(
                "Treat this run's timing as a lower bound on the plan's speed. Re-measure with "
                "the process on its own cpuset, or reduce execution.parallelism so the engine "
                "asks for cores it can actually hold."
            ),
            op=kind,
            detail={"preemptions_per_core_second": rate, "operators": len(contended)},
        )
    ]

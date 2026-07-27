"""How many cores the engine should ask for, given how many it is really getting.

The core count is the one resource Batcher sizes against without ever checking whether it got
it. Memory has a pressure monitor, a budget, and a spill path; CPU has `available_cpu_count()`,
which reports what the cgroup *permits* and says nothing about what the scheduler *delivers*.
Those diverge constantly in the deployments Batcher targets:

* two Ray workers land on one node and each fans out to the node's full core count, so both
  run at half speed while each believes it has the whole box;
* a container is permitted 8 cores by its cpuset but throttled below that by a CFS quota it
  keeps hitting, so a ninth thread only lengthens the queue;
* a co-tenant process takes half the machine, which no in-process API reports at all.

The failure mode is the same every time and it is not an error: threads are created, work is
distributed, and every thread runs slower than one thread would have. Past saturation, extra
workers cost context switches, cache evictions, and lock contention while adding no
throughput — so the same query gets slower the more parallel the engine tries to be, which
reads as "the engine is slow on this box" rather than "the engine asked for too much".

This policy is the counter-pressure. It takes the permitted budget and divides it by the
measured oversubscription, so the engine asks for the cores it can actually get. Carbonite's
lane, precisely: it protects against a resource the plan cannot have, without rewriting the
plan or deciding anything about what to compute.

**It only ever reduces.** A quiet machine measures no oversubscription and gets the permitted
budget unchanged, so the ordinary case is untouched and the ordinary cost is one read of a
`/proc` file per query.
"""

from __future__ import annotations

from batcher._internal.hardware import available_cpu_count, cpu_oversubscription

__all__ = ["effective_core_budget", "oversubscription_note"]

# Never cut fan-out below this fraction of the permitted budget, however bad the contention
# reading. A pathological measurement — a transient load spike, a PSI file reporting a
# neighbouring cgroup's stall, a load average still carrying a finished job's tail — must not
# be able to serialize a query. A quarter of the cores is a large enough cut to relieve real
# contention and a small enough floor that a bad reading is survivable.
MIN_BUDGET_FRACTION = 0.25

# Oversubscription below this is treated as noise and changes nothing. A perfectly idle box
# still reports a load average slightly above zero from the measurement itself, and PSI
# reports occasional single-digit-microsecond stalls on any live system. Acting on those would
# make fan-out jitter query to query for no reason.
CONTENTION_DEADBAND = 1.25


def effective_core_budget(configured: int = 0) -> int:
    """Cores to fan out across, reduced by measured contention. Never fewer than 1.

    `configured` wins when set, because an explicit `execution.parallelism` is a user
    instruction rather than an estimate, and silently overriding it would make the knob a lie.
    Otherwise the permitted budget is divided by the measured oversubscription, floored at
    `MIN_BUDGET_FRACTION` of the permitted count.

    Args:
        configured: An explicit parallelism setting, or `0` to derive one.

    Returns:
        The core count to size thread pools and task fan-out against, at least 1.
    """
    permitted = max(1, available_cpu_count())
    if configured > 0:
        return configured
    pressure = cpu_oversubscription()
    if pressure <= CONTENTION_DEADBAND:
        return permitted
    floor = max(1, int(permitted * MIN_BUDGET_FRACTION))
    return max(floor, int(permitted / pressure))


def oversubscription_note() -> str:
    """One line explaining a reduced budget, or `""` when nothing was reduced.

    Exists because a silently narrowed fan-out is indistinguishable from an engine that failed
    to parallelize, and the two have opposite fixes: move the workload, or fix the plan. This
    is what `EXPLAIN` and the decision log say instead of leaving the reader to guess.

    Returns:
        A short explanation, or `""` when the machine measured no contention.
    """
    pressure = cpu_oversubscription()
    if pressure <= CONTENTION_DEADBAND:
        return ""
    permitted = max(1, available_cpu_count())
    budget = effective_core_budget()
    if budget >= permitted:
        return ""
    return (
        f"cpu fan-out reduced {permitted} -> {budget} cores: the machine is "
        f"{pressure:.1f}x oversubscribed, so the extra threads would queue rather than run"
    )

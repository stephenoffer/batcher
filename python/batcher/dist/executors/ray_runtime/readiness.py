"""Bounded waits for a Ray cluster that is not ready yet.

Two waits with the same shape, and the same failure mode if they are unbounded. Connecting
to a head that is still starting: the driver and the head come up concurrently in every
orchestrated environment, so the first attach routinely fails against a cluster seconds
from ready. And waiting for the autoscaler to deliver capacity a query asked for, so the
query fills the cluster it triggered a scale-up for rather than the pre-scale one.

Both poll or retry against a deadline, both give up and let the caller degrade rather than
hang, and both are capped by the job's own lease (`config.deadline`) — because a wait that
outlives the process helps nobody, and on a leased allocation these are the longest things
between the query being submitted and any work happening.

Split from `scaling` (which *measures* the live cluster) and `autoscale_request` (which
*asks* for capacity) because this is the third side of the same concern: waiting for what
was asked for to arrive.
"""

from __future__ import annotations

import threading
import time

from batcher._internal.errors import BackendError
from batcher.config import active_config
from batcher.config.deadline import remaining_budget

__all__ = ["await_autoscale"]


def _cluster_topology() -> dict:
    """The live topology, resolved through the `scaling` module object.

    Deliberately not `from .scaling import cluster_topology`: the tests for these waits patch
    `scaling.cluster_topology` to script a capacity series, and a name bound at import time
    would not see that patch — the wait would silently poll the real cluster while the test
    believed it was driving one. Also breaks what would otherwise be an import cycle, since
    `scaling.clamp_workers` delegates here.
    """
    from batcher.dist.executors.ray_runtime import scaling

    return scaling.cluster_topology()


# --- Connecting to a head that is still coming up -----------------------------------


def _explicit_cluster_address() -> str | None:
    """The cluster address the *user* named, or `None` when one was merely detected.

    The distinction decides what an unreachable cluster means. A detected address (a
    KubeRay/Anyscale marker in the environment) is a hint, and falling back to a local Ray
    when it does not answer is a reasonable degradation. An address the user configured is
    an instruction, and quietly running single-node instead is a wrong answer rather than a
    degraded one — the job reports success having used one machine of the cluster they
    named, which is indistinguishable from working.
    """
    import os

    dc = active_config().distributed
    return dc.ray_address or os.environ.get("RAY_ADDRESS") or None


def _attach_with_retry(ray, **init_kwargs) -> bool:
    """Attach to a running cluster, retrying a not-yet-answering head. True when attached.

    The head and the driver come up concurrently in every orchestrated environment, so the
    first attach routinely fails against a cluster that is seconds from ready: a KubeRay
    driver pod is admitted before the head passes readiness, and a Slurm job's `ray start
    --head` races the step that runs the query. Retrying with exponential backoff turns
    that race into a pause instead of a silent single-node run.

    Bounded by `cluster_connect_timeout_s` and, under a lease, by the time actually left —
    waiting past the job's own deadline for a cluster to appear helps nobody. Returns False
    when the window is exhausted, leaving the caller to decide between raising (an explicit
    address) and falling back to local (a detected one).
    """
    dc = active_config().distributed
    budget = remaining_budget(max(0.0, dc.cluster_connect_timeout_s), reserve_s=dc.drain_lead_s)
    deadline = time.monotonic() + budget
    backoff = 0.5
    while True:
        try:
            ray.init(**init_kwargs)
            return True
        except ConnectionError:
            if ray.is_initialized():
                return True  # someone else won the race; we are attached either way
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(backoff, max(0.0, deadline - time.monotonic())))
            backoff = min(backoff * 2, 5.0)


def _connect_or_fall_back(ray, workers: int) -> None:
    """Retry the attach, then either raise (explicit address) or start a local Ray.

    Splitting the two cases is the point. Falling back to local for a *detected* address
    keeps a dev run inside a workspace whose cluster is down working, which is why the
    fallback exists. Doing the same for an address the user configured turns "connect to
    this cluster" into "run on this laptop" with no error anywhere — the job succeeds, on
    one machine, and nothing says the cluster was never reached.
    """
    from batcher.dist.executors.ray_runtime.lifecycle import _ray_init_kwargs

    address = _explicit_cluster_address()
    if _attach_with_retry(ray, **_ray_init_kwargs(workers)):
        return
    if address is not None:
        raise BackendError(
            f"could not connect to the Ray cluster at {address!r} within "
            f"{active_config().distributed.cluster_connect_timeout_s}s. "
            "Batcher will not silently run single-node against a cluster address you set: "
            "check the address and that the head is reachable, raise "
            "distributed.cluster_connect_timeout_s if the head is still starting, or pass "
            "distributed=False to run on this machine deliberately."
        )
    # Only a *detected* address gets here: the environment hinted at a cluster that turned
    # out not to be reachable. Degrade to a local single-node Ray rather than fail a job the
    # user never pointed at a specific cluster.
    ray.init(**_ray_init_kwargs(workers, force_local=True))


# --- Waiting for the autoscaler to deliver capacity ---------------------------------

# The capacity a wait *confirmed* the autoscaler won't exceed (set when a wait stalls below
# target): a later query asking for more skips the wait instead of re-discovering the same
# ceiling, so a fixed-at-max cluster pays the startup grace ONCE, not per cold query. A wait
# that grows the cluster lifts it (`_note_reached`), so real scale-up is never pinned stale.
_reachable_ceiling: float = float("inf")
_ceiling_lock = threading.Lock()


def _note_ceiling(best_cpus: int) -> None:
    """Record that the autoscaler stalled at `best_cpus` — the cluster will not exceed it."""
    global _reachable_ceiling
    with _ceiling_lock:
        _reachable_ceiling = min(_reachable_ceiling, float(best_cpus))


def _note_reached(cpus: int) -> None:
    """Lift a stale ceiling once capacity has climbed past it (the cluster grew/recovered)."""
    global _reachable_ceiling
    with _ceiling_lock:
        if cpus > _reachable_ceiling:
            _reachable_ceiling = float("inf")


def _reset_capacity_ceiling() -> None:
    """Forget the learned ceiling (tests; and any caller that wants a fresh probe)."""
    global _reachable_ceiling
    with _ceiling_lock:
        _reachable_ceiling = float("inf")


def await_autoscale(target_cpus: int, target_gpus: float = 0.0) -> None:
    """Block (bounded, growth-detected) until the autoscaler grows the cluster toward
    `target_cpus` cores (and `target_gpus` GPUs).

    Called *before* the fan-out is sized to the cluster, so a query that triggered a scale-up
    (`request_autoscale`) fills the SCALED-UP cluster rather than the pre-scale one — without
    it the worker-per-node fill reads the current (small) topology and the query never uses
    the nodes it asked for. A no-op when the wait is disabled, Ray is down, the cluster already
    covers the target, or a previous wait learned it will not reach the target
    (`_reachable_ceiling`) — so a fixed cluster pays the startup grace once, not per query.
    Pure scheduling — the result is identical whether it waits or not.
    """
    if active_config().distributed.autoscale_wait_s <= 0 or target_cpus <= 0:
        return
    import ray

    if not ray.is_initialized():
        return
    topo = _cluster_topology()
    avail = int(topo["cpus"])
    # Read current capacity BEFORE the ceiling short-circuit, so a cluster grown since the
    # ceiling was learned re-probes: covering the target returns satisfied (lifting the stale
    # ceiling); merely exceeding it drops the bound and waits for the rest.
    if avail >= target_cpus and float(topo["gpus"]) >= target_gpus:
        if target_gpus <= 0:
            _note_reached(avail)
        return
    with _ceiling_lock:
        ceiling = _reachable_ceiling
    if avail > ceiling:
        _note_reached(avail)  # capacity climbed past the old ceiling — it is stale
    elif target_cpus > ceiling and target_gpus <= 0:
        return  # a prior wait proved this is unreachable — don't re-discover it
    _await_autoscale(target_cpus, avail, target_gpus, float(topo["gpus"]))


def _await_autoscale(
    target_cpus: int, avail: int, target_gpus: float = 0.0, avail_gpus: float = 0.0
) -> int:
    """Wait (bounded) for the cluster to grow to `target_cpus` (and `target_gpus`), returning
    observed CPUs.

    Polls the live CPU/GPU counts every `autoscale_poll_s` until both cover their targets or
    `autoscale_wait_s` elapses, then returns the CPU count. A GPU stage waits for the GPUs
    too, not just the cores (else it clamps to the 0 GPUs visible before the GPU node boots).
    A no-op (returns `avail`) when the wait is disabled or the cluster already fits; stops
    early via the grace windows below when capacity goes flat.
    """
    dc = active_config().distributed
    if dc.autoscale_wait_s <= 0 or (avail >= target_cpus and avail_gpus >= target_gpus):
        return avail
    import time

    from batcher.config.deadline import remaining_budget

    # Under a wall-clock lease, wait only as long as the job will still be alive to use the
    # nodes — minus the drain lead, so the fleet that does arrive has time to publish and
    # migrate its output. This is the longest wait in the scheduling path (180 s by default
    # on an autoscaling cluster), so it is the one that most often consumes a short
    # allocation entirely: a Slurm job with 90 seconds left would otherwise spend all of it
    # waiting for capacity that arrives after the kill, and compute nothing. Shrinking it
    # only makes the poll loop give up sooner and run on the capacity already present, which
    # is exactly what it does for a stalled autoscaler.
    budget = remaining_budget(dc.autoscale_wait_s, reserve_s=dc.drain_lead_s)
    if budget <= 0:
        return avail  # no time to wait for nodes; run on what is here now
    deadline = time.monotonic() + budget
    poll = max(0.1, dc.autoscale_poll_s)
    # Give up early once capacity has been flat for the grace window — the autoscaler is done
    # (fixed cluster) or cannot satisfy the request (spot unavailable), so the rest of the
    # budget would block on nodes that will not arrive; any gain resets the window. Two
    # regimes: until the FIRST growth a short `startup_grace` applies — an infeasible request
    # (a fixed cluster already at max, the common case where a large aggregate's fan-out
    # exceeds the node count) grows zero from the start, and the query already runs on current
    # capacity, so it must not eat the full 90 s stall for nodes that never come. Once any
    # growth appears the cluster is genuinely scaling and the longer `autoscale_stall_s`
    # governs. `startup_grace` sits above a couple of polls so nodes registering within a few
    # seconds still cross into the growing regime.
    stall_grace = max(dc.autoscale_stall_s, poll * 2)
    startup_grace = max(dc.autoscale_startup_grace_s, poll * 2)
    best = (avail, avail_gpus)
    saw_growth = False
    reached = False
    stalled = False
    last_growth = time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(min(poll, max(0.0, deadline - time.monotonic())))
        topo = _cluster_topology()
        avail = int(topo["cpus"])
        avail_gpus = float(topo["gpus"])
        if avail >= target_cpus and avail_gpus >= target_gpus:
            reached = True
            break
        if (avail, avail_gpus) > best:
            best = (avail, avail_gpus)
            saw_growth = True
            last_growth = time.monotonic()
        elif time.monotonic() - last_growth >= (stall_grace if saw_growth else startup_grace):
            stalled = True
            break  # nothing is coming (never started, or grew then stopped)
    # A CPU-only wait that stalled below its target has learned a ceiling; one that reached
    # (or grew past a stale ceiling) lifts it. GPU waits don't participate — a 0-GPU snapshot
    # before a GPU node boots must not cap future GPU requests.
    #
    # A wait cut short by the *lease* has learned nothing about the cluster. It ran out of
    # time, which is a fact about this job, not about how far the autoscaler will go — and
    # recording it as a ceiling would tell every later query in the process that capacity it
    # never probed is unreachable. Only a genuine stall, or running the full requested
    # budget, is evidence about the autoscaler.
    truncated = budget < dc.autoscale_wait_s and not stalled
    if target_gpus <= 0:
        if reached or avail >= target_cpus:
            _note_reached(avail)
        elif not truncated:
            _note_ceiling(int(best[0]))
    return avail

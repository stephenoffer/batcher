"""The autoscaler request lifecycle: scale a cluster up for a query, reclaim after.

`ray.autoscaler.sdk.request_resources` sets a *sticky* floor — the autoscaler holds that
capacity until told otherwise — so an unmanaged request pins a cluster scaled-up forever
after one big query. This module owns the process-wide high-water floor across in-flight
query scopes and drops it back to zero when the last one ends.

Split from `scaling` (which measures and clamps against the live topology) because this is
the *write* side of the same concern: it asks the autoscaler for capacity, where `scaling`
observes what arrived.
"""

from __future__ import annotations

import contextlib
import math
import threading

# --- Autoscaler request lifecycle (scale up for a query, reclaim after) -------------
# `request_resources` sets a *sticky* floor: the autoscaler keeps that many cores until
# told otherwise. Left unmanaged, one big query pins the cluster scaled-up forever. We
# track a process-wide high-water floor across in-flight query scopes and reset it to 0
# the moment the last one ends, so the autoscaler reclaims the now-idle nodes. A
# running query's nodes are *busy* (tasks / persistent-fleet actors), so they are never
# reclaimed mid-query regardless of the floor — the floor only drives scale-*up* and
# keeps a node from being reclaimed in the brief gap before it picks up work.
_autoscale_lock = threading.Lock()
_autoscale_active = 0
_autoscale_floor = 0
_autoscale_gpu_floor = 0
# Custom-accelerator floor (TPU / neuron_cores / HPU / an operator's own resource), max
# per resource name across in-flight scopes — the same high-water rule as the CPU/GPU floors.
_autoscale_resources: tuple[tuple[str, float], ...] = ()


def _apply_autoscale_floor(
    cpus: int, gpus: int = 0, resources: tuple[tuple[str, float], ...] = ()
) -> None:
    with contextlib.suppress(Exception):
        from ray.autoscaler.sdk import request_resources

        if resources:
            # A custom accelerator needs bundles naming *that* resource: a `{"GPU": 1}`
            # bundle asks for GPU nodes, which a TPU cluster has none of, so the query would
            # wait out the autoscale window and then run on whatever was already up.
            request_resources(num_cpus=cpus, bundles=[{n: a} for n, a in resources])
        elif gpus > 0:
            # A GPU floor needs GPU *bundles* — `request_resources(num_cpus=)` alone never
            # triggers GPU-node scale-up, so a GPU query would hang or fall back to CPU
            # nodes it can't run on. One `{"GPU": 1}` bundle per requested GPU asks the
            # autoscaler for that many GPUs; the CPU floor rides alongside for the
            # relational stages. (Whole-GPU bundles — fractional packing is a scheduling
            # concern, not an autoscale-shape one.)
            request_resources(num_cpus=cpus, bundles=[{"GPU": 1}] * gpus)
        else:
            request_resources(num_cpus=cpus)


# Whether the drain hook that drops the floor on preemption has been registered. One per
# process: the hook is idempotent and the monitor fires it once, so re-registering per query
# would only grow the callback list.
_drain_release_armed = False


def _arm_drain_release() -> None:
    """Drop the autoscaler floor if this process is told it is going away. Idempotent.

    `release_autoscale` runs in a `finally`, which covers every way a query can *end* and
    none of the ways a job can be *killed*. That gap is the normal case in the environments
    this matters for: Slurm kills the allocation at its time limit, Kubernetes evicts the
    driver pod, a spot reclamation takes the node the driver was on. The floor lives in the
    autoscaler, not in this process, so it outlives the driver that set it — and the
    autoscaler holds that capacity until something tells it otherwise. A single killed job
    therefore pins a cluster at its scaled-up size indefinitely, which is a standing bill
    with nothing running against it.

    A drain notice arrives before the kill in exactly those cases (`SIGTERM` from Kubernetes
    or Slurm, `SIGUSR1` from a job submitted with `--signal`, or the lease deadline itself),
    so releasing there closes the gap. The driver's monitor is started here rather than at
    import: it costs a poll thread and a signal trap, and only a preemptible deployment
    needs either.

    Best-effort throughout. Failing to arm this must never fail the query it was called
    from — the floor leaking is a cost, not a correctness problem.
    """
    global _drain_release_armed
    if _drain_release_armed:
        return
    try:
        from batcher.carbonite.resilience import preemption_monitor
        from batcher.config import active_config

        if active_config().distributed.resilience != "spot":
            return  # a stable cluster is not preempted; don't pay for a poll thread
        monitor = preemption_monitor()
        monitor.on_drain(_release_all)
        monitor.start()
        _drain_release_armed = True
    except Exception as exc:  # pragma: no cover - arming is best-effort
        from batcher._internal.logging import note_suppressed

        note_suppressed("dist", "arm the autoscale-floor drain release", exc)


def _release_all() -> None:
    """Zero the autoscaler floor outright, whatever scopes are still nominally in flight.

    Unlike `release_autoscale` this does not decrement — it abandons. Called only when the
    process has been told it is going away, at which point every in-flight scope is about to
    stop existing and the floors they hold are exactly what must not survive them.
    """
    global _autoscale_active, _autoscale_floor, _autoscale_gpu_floor, _autoscale_resources
    with _autoscale_lock:
        _autoscale_active = 0
        _autoscale_floor = 0
        _autoscale_gpu_floor = 0
        _autoscale_resources = ()
        _apply_autoscale_floor(0, 0, ())


def _worth_scaling_up() -> bool:
    """Whether nodes requested now could still be used before this job is killed.

    Scaling up is not free and not instant. A cloud node takes a minute or more to boot,
    join, and be schedulable, so a job with less than that left asks for capacity that
    arrives after it is gone — and because the floor is sticky and lives in the autoscaler,
    the nodes then sit idle until something else drops it. That is a real bill for zero
    work, charged at exactly the moment the job has nothing to gain.

    The bar is the drain lead: if there is not even enough time left to migrate output
    before the kill, there is certainly not enough to boot a node and run work on it.
    Always true when no deadline is known, so an ordinary cluster is unchanged.
    """
    from batcher.config import active_config
    from batcher.config.deadline import seconds_remaining

    try:
        lead = float(active_config().distributed.drain_lead_s)
    except Exception:  # pragma: no cover - config is resolvable in practice
        return True
    remaining = seconds_remaining()
    # Stated directly rather than through `remaining_budget`, whose "no deadline returns the
    # request unchanged" convention would read a `drain_lead_s` of 0 as "no time left" and
    # disable scale-up on every cluster that turned the drain lead off.
    return remaining is None or remaining > lead


def request_autoscale(
    target_cpus: int,
    target_gpus: float = 0.0,
    target_resources: tuple[tuple[str, float], ...] = (),
) -> None:
    """Register a query scope wanting `target_cpus` cores (and `target_gpus` GPUs); maintain
    the high-water floor.

    The autoscaler is asked for the max over every in-flight scope, so concurrent
    queries compose and one scope never lowers the floor a live sibling still needs. A
    GPU query (`target_gpus > 0`) also lifts a GPU floor so the autoscaler provisions GPU
    nodes — not just cores, and `target_resources` does the same for an accelerator Ray
    names rather than counts as a GPU (TPU, neuron_cores, HPU). Balanced by exactly one
    `release_autoscale` at the scope's teardown — and, when the process is preemptible, by
    a drain hook that drops the floor if the job is killed before that teardown runs (see
    `_arm_drain_release`).
    """
    global _autoscale_active, _autoscale_floor, _autoscale_gpu_floor, _autoscale_resources
    _arm_drain_release()
    if not _worth_scaling_up():
        # Still count the scope so `release_autoscale` stays balanced; just don't raise a
        # floor whose nodes would boot after this job is dead.
        with _autoscale_lock:
            _autoscale_active += 1
        return
    with _autoscale_lock:
        _autoscale_active += 1
        _autoscale_floor = max(_autoscale_floor, target_cpus)
        _autoscale_gpu_floor = max(_autoscale_gpu_floor, math.ceil(target_gpus))
        merged = dict(_autoscale_resources)
        for name, amount in target_resources:
            merged[name] = max(merged.get(name, 0.0), amount)
        _autoscale_resources = tuple(sorted(merged.items()))
        _apply_autoscale_floor(_autoscale_floor, _autoscale_gpu_floor, _autoscale_resources)


def release_autoscale() -> None:
    """End one query scope; when the last one ends, drop the autoscaler floor (CPU and GPU)
    to 0 so it can reclaim the idle nodes the query scaled up (instead of pinning them
    forever)."""
    global _autoscale_active, _autoscale_floor, _autoscale_gpu_floor, _autoscale_resources
    with _autoscale_lock:
        _autoscale_active -= 1
        if _autoscale_active <= 0:
            _autoscale_active = 0
            _autoscale_floor = 0
            _autoscale_gpu_floor = 0
            _autoscale_resources = ()
            _apply_autoscale_floor(0, 0, ())

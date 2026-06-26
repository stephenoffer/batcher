"""Live cluster topology and the autoscaler request lifecycle.

Reads the cluster shape on demand (so it tracks autoscaler growth/shrink), clamps a
requested worker fan-out to schedulable capacity, and manages a process-wide
high-water autoscaler floor across in-flight query scopes (scale up for a query,
reclaim the idle nodes after the last scope ends).
"""

from __future__ import annotations

import contextlib
import math
import threading

from batcher.config import active_config


def cluster_topology() -> dict:
    """Live cluster shape: alive node count + total CPUs/GPUs. Ray must be up.

    Read on demand (not cached) so it stays correct as the autoscaler grows or
    shrinks the cluster — `ray.nodes()`/`ray.cluster_resources()` are cheap RPCs.
    """
    import ray

    nodes = [n for n in ray.nodes() if n.get("Alive", True)]
    resources = ray.cluster_resources()
    return {
        "nodes": max(1, len(nodes)),
        "cpus": float(resources.get("CPU", 0.0)),
        "gpus": float(resources.get("GPU", 0.0)),
    }


def clamp_workers(workers: int, num_cpus: float = 1.0) -> int:
    """Clamp the requested worker fan-out to what the cluster can actually schedule.

    Each worker asks for `num_cpus` cores, so the cluster fits `avail / num_cpus`
    workers — fractional requests pack *more* than one per core, whole-or-larger
    requests fewer. Creating more than fit over-subscribes the cluster (and makes the
    gang-scheduling placement group unsatisfiable). The query scope already asked the
    autoscaler to grow (`request_autoscale`); on a genuine autoscaling cluster
    (`distributed.autoscale_wait_s > 0`) we then *wait* — bounded — for the new nodes to
    arrive here, so the job runs on the scaled-up cluster instead of under-provisioned.
    With the wait off (the default) it clamps to current capacity. Always leaves at least
    one worker; a no-op when Ray reports no CPUs (test stubs).
    """
    import ray

    if not ray.is_initialized():
        return workers
    num_cpus = max(num_cpus, 1e-9)
    avail = int(cluster_topology()["cpus"])
    capacity = int(avail / num_cpus)
    if avail <= 0 or workers <= capacity:
        return workers
    # The query scope already asked the autoscaler for these cores (`request_autoscale`),
    # so here we only wait (bounded) for them to arrive, then clamp to what is schedulable.
    target_cpus = math.ceil(workers * num_cpus)
    avail_now = _await_autoscale(target_cpus, avail) or avail
    return max(1, min(workers, int(avail_now / num_cpus)))


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


def _apply_autoscale_floor(cpus: int) -> None:
    with contextlib.suppress(Exception):
        from ray.autoscaler.sdk import request_resources

        request_resources(num_cpus=cpus)


def request_autoscale(target_cpus: int) -> None:
    """Register a query scope wanting `target_cpus` cores; maintain the high-water floor.

    The autoscaler is asked for the max over every in-flight scope, so concurrent
    queries compose and one scope never lowers the floor a live sibling still needs.
    Balanced by exactly one `release_autoscale` at the scope's teardown.
    """
    global _autoscale_active, _autoscale_floor
    with _autoscale_lock:
        _autoscale_active += 1
        _autoscale_floor = max(_autoscale_floor, target_cpus)
        _apply_autoscale_floor(_autoscale_floor)


def release_autoscale() -> None:
    """End one query scope; when the last one ends, drop the autoscaler floor to 0 so it
    can reclaim the idle nodes the query scaled up (instead of pinning them forever)."""
    global _autoscale_active, _autoscale_floor
    with _autoscale_lock:
        _autoscale_active -= 1
        if _autoscale_active <= 0:
            _autoscale_active = 0
            _autoscale_floor = 0
            _apply_autoscale_floor(0)


def _await_autoscale(target_cpus: int, avail: int) -> int:
    """Wait (bounded) for the cluster to grow to `target_cpus`, returning observed CPUs.

    Polls the live CPU count every `autoscale_poll_s` until it covers `target_cpus` or
    `autoscale_wait_s` elapses, then returns it — so a query that triggered a scale-up
    runs on the bigger cluster. A no-op (returns `avail` immediately) when the wait is
    disabled or the cluster already fits, so a fixed cluster never blocks on a scale-up
    that cannot happen. Stops the instant capacity is sufficient.
    """
    dc = active_config().distributed
    if dc.autoscale_wait_s <= 0 or avail >= target_cpus:
        return avail
    import time

    deadline = time.monotonic() + dc.autoscale_wait_s
    poll = max(0.1, dc.autoscale_poll_s)
    while time.monotonic() < deadline:
        time.sleep(min(poll, max(0.0, deadline - time.monotonic())))
        avail = int(cluster_topology()["cpus"])
        if avail >= target_cpus:
            break
    return avail

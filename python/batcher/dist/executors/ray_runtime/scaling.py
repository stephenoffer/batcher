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


def node_classes() -> list[dict]:
    """Per-alive-node resource class: ``{"cpus", "gpus", "accelerator_type"}``.

    The explicit cluster-heterogeneity model the scheduler lacked: a node is a "GPU
    node" when it exposes a `GPU` resource, a "CPU-only node" otherwise. The accelerator
    type comes from Ray's default `ray.io/accelerator-type` node label when present.
    Read on demand (Ray must be up) so it tracks autoscaler growth/shrink; empty when the
    topology is unreadable (the caller then keeps its homogeneous defaults).
    """
    try:
        import ray

        out: list[dict] = []
        for n in ray.nodes():
            if not n.get("Alive", True):
                continue
            res = n.get("Resources", {})
            cpus = float(res.get("CPU", 0.0))
            if cpus <= 0:
                continue
            labels = n.get("Labels", {}) or {}
            out.append(
                {
                    "cpus": cpus,
                    "gpus": float(res.get("GPU", 0.0)),
                    "accelerator_type": labels.get("ray.io/accelerator-type"),
                }
            )
        return out
    except Exception:
        return []


def cpu_only_can_host(workers: int, num_cpus: float) -> bool:
    """Whether the cluster's **CPU-only** nodes alone can host `workers` x `num_cpus` cores.

    The gate for keeping a relational (CPU) fleet off GPU nodes on a heterogeneous
    cluster: only restrict the fleet to CPU-only nodes when those nodes have the capacity
    to run it — otherwise the restriction would under-provision (or fail to place) the
    query, so the fleet is left free to use every node (today's behavior). Returns False
    on a homogeneous cluster (no GPU nodes ⇒ nothing to keep off ⇒ no restriction needed)
    or unreadable topology.
    """
    classes = node_classes()
    if not classes or not any(c["gpus"] > 0 for c in classes):
        return False  # homogeneous / GPU-less → no restriction (use all nodes)
    cpu_only_cores = sum(c["cpus"] for c in classes if c["gpus"] <= 0)
    return cpu_only_cores >= workers * max(num_cpus, 1e-9)


def clamp_workers(workers: int, num_cpus: float = 1.0, num_gpus: float = 0.0) -> int:
    """Clamp the requested worker fan-out to what the cluster can actually schedule.

    Each worker asks for `num_cpus` cores (and, for a GPU stage, `num_gpus` GPUs), so the
    cluster fits `floor(cpus / num_cpus)` workers — bounded *also* by `floor(gpus /
    num_gpus)` when GPUs are requested, since a GPU stage cannot pack more workers than
    there are GPUs. Fractional requests pack more than one per device, whole-or-larger
    requests fewer. Creating more than fit over-subscribes the cluster (and makes the
    gang-scheduling placement group unsatisfiable). The query scope already asked the
    autoscaler to grow (`request_autoscale`); on a genuine autoscaling cluster
    (`distributed.autoscale_wait_s > 0`) we then *wait* — bounded — for the new nodes
    (CPU *and* GPU) to arrive, so the job runs on the scaled-up cluster instead of
    under-provisioned. With the wait off (the default) it clamps to current capacity.
    Always leaves at least one worker; a no-op when Ray reports no CPUs (test stubs).
    """
    import ray

    if not ray.is_initialized():
        return workers
    num_cpus = max(num_cpus, 1e-9)
    topo = cluster_topology()
    avail_cpus = int(topo["cpus"])
    capacity = int(avail_cpus / num_cpus)
    if num_gpus > 0:
        capacity = min(capacity, int(topo["gpus"] / num_gpus))
    if avail_cpus <= 0 or workers <= capacity:
        return workers
    # The query scope already asked the autoscaler for these resources
    # (`request_autoscale`); here we only wait (bounded) for them to arrive, then clamp
    # to what is schedulable. A GPU stage waits for the GPUs too, not just the cores.
    target_cpus = math.ceil(workers * num_cpus)
    target_gpus = workers * num_gpus
    topo_now = _await_autoscale(target_cpus, avail_cpus, target_gpus, float(topo["gpus"]))
    avail_now = topo_now or avail_cpus
    fit = int(avail_now / num_cpus)
    if num_gpus > 0:
        fit = min(fit, int(float(cluster_topology()["gpus"]) / num_gpus))
    return max(1, min(workers, fit))


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


def _apply_autoscale_floor(cpus: int, gpus: int = 0) -> None:
    with contextlib.suppress(Exception):
        from ray.autoscaler.sdk import request_resources

        if gpus > 0:
            # A GPU floor needs GPU *bundles* — `request_resources(num_cpus=)` alone never
            # triggers GPU-node scale-up, so a GPU query would hang or fall back to CPU
            # nodes it can't run on. One `{"GPU": 1}` bundle per requested GPU asks the
            # autoscaler for that many GPUs; the CPU floor rides alongside for the
            # relational stages. (Whole-GPU bundles — fractional packing is a scheduling
            # concern, not an autoscale-shape one.)
            request_resources(num_cpus=cpus, bundles=[{"GPU": 1}] * gpus)
        else:
            request_resources(num_cpus=cpus)


def request_autoscale(target_cpus: int, target_gpus: float = 0.0) -> None:
    """Register a query scope wanting `target_cpus` cores (and `target_gpus` GPUs); maintain
    the high-water floor.

    The autoscaler is asked for the max over every in-flight scope, so concurrent
    queries compose and one scope never lowers the floor a live sibling still needs. A
    GPU query (`target_gpus > 0`) also lifts a GPU floor so the autoscaler provisions GPU
    nodes — not just cores. Balanced by exactly one `release_autoscale` at the scope's
    teardown.
    """
    global _autoscale_active, _autoscale_floor, _autoscale_gpu_floor
    with _autoscale_lock:
        _autoscale_active += 1
        _autoscale_floor = max(_autoscale_floor, target_cpus)
        _autoscale_gpu_floor = max(_autoscale_gpu_floor, math.ceil(target_gpus))
        _apply_autoscale_floor(_autoscale_floor, _autoscale_gpu_floor)


def release_autoscale() -> None:
    """End one query scope; when the last one ends, drop the autoscaler floor (CPU and GPU)
    to 0 so it can reclaim the idle nodes the query scaled up (instead of pinning them
    forever)."""
    global _autoscale_active, _autoscale_floor, _autoscale_gpu_floor
    with _autoscale_lock:
        _autoscale_active -= 1
        if _autoscale_active <= 0:
            _autoscale_active = 0
            _autoscale_floor = 0
            _autoscale_gpu_floor = 0
            _apply_autoscale_floor(0, 0)


def _await_autoscale(
    target_cpus: int, avail: int, target_gpus: float = 0.0, avail_gpus: float = 0.0
) -> int:
    """Wait (bounded) for the cluster to grow to `target_cpus` (and `target_gpus`), returning
    observed CPUs.

    Polls the live CPU/GPU counts every `autoscale_poll_s` until both cover their targets
    or `autoscale_wait_s` elapses, then returns the CPU count — so a query that triggered a
    scale-up runs on the bigger cluster. A GPU stage waits for the GPUs to arrive too, not
    just the cores (otherwise it would clamp to the 0 GPUs visible before the GPU node is
    up). A no-op (returns `avail` immediately) when the wait is disabled or the cluster
    already fits, so a fixed cluster never blocks on a scale-up that cannot happen. Stops
    the instant capacity is sufficient.
    """
    dc = active_config().distributed
    if dc.autoscale_wait_s <= 0 or (avail >= target_cpus and avail_gpus >= target_gpus):
        return avail
    import time

    deadline = time.monotonic() + dc.autoscale_wait_s
    poll = max(0.1, dc.autoscale_poll_s)
    while time.monotonic() < deadline:
        time.sleep(min(poll, max(0.0, deadline - time.monotonic())))
        topo = cluster_topology()
        avail = int(topo["cpus"])
        avail_gpus = float(topo["gpus"])
        if avail >= target_cpus and avail_gpus >= target_gpus:
            break
    return avail

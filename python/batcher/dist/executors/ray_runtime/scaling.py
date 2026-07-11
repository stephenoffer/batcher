"""Live cluster topology and the autoscaler request lifecycle.

Reads the cluster shape on demand (so it tracks autoscaler growth/shrink), clamps a
requested worker fan-out to schedulable capacity, and manages a process-wide
high-water autoscaler floor across in-flight query scopes (scale up for a query,
reclaim the idle nodes after the last scope ends).
"""

from __future__ import annotations

import contextlib
import contextvars
import math
import threading

from batcher.config import active_config


class _Topology:
    """A one-shot snapshot of the live cluster shape (alive node records + resources)."""

    __slots__ = ("alive_nodes", "resources")

    def __init__(self, alive_nodes: list[dict], resources: dict) -> None:
        self.alive_nodes = alive_nodes
        self.resources = resources


# The topology snapshot in force for the current scheduling phase, if any. A distributed
# query reads the cluster shape from several places (transport choice, placement strategy,
# node-class selector, spread heuristic) — each an O(nodes) `ray.nodes()` RPC. Inside a
# `topology_scope()` they share ONE read. Ambient (a ContextVar) so it reaches those helpers
# without threading a snapshot through every call. Left unset (None) → every reader reads
# live, exactly as before.
_TOPOLOGY: contextvars.ContextVar[_Topology | None] = contextvars.ContextVar(
    "batcher_topology_snapshot", default=None
)


def _read_topology() -> _Topology:
    import ray

    return _Topology([n for n in ray.nodes() if n.get("Alive", True)], ray.cluster_resources())


@contextlib.contextmanager
def topology_scope():
    """Snapshot the live cluster shape once for the enclosed scheduling phase.

    Collapses the ~5 `ray.nodes()`/`cluster_resources()` reads a distributed query's
    placement/transport phase makes into a single one. MUST be entered only *after* the
    autoscale-wait and worker clamp have settled the cluster size — those poll the live
    topology and must never see a stale snapshot; within this scope the size is fixed, so a
    snapshot is faithful. A read failure inside falls back to live reads (the snapshot is
    best-effort). Nesting reuses the outer snapshot.
    """
    if _TOPOLOGY.get() is not None:
        yield  # already inside a scope — reuse it
        return
    try:
        snap = _read_topology()
    except Exception:
        yield  # topology unreadable → readers fall back to live
        return
    token = _TOPOLOGY.set(snap)
    try:
        yield
    finally:
        _TOPOLOGY.reset(token)


# The Ray head node's marker. Distributed WORKER actors are never placed on the head
# (it runs the GCS / dashboard / job supervisor — scheduling data operators there causes
# contention and instability), so every node-count that sizes the worker fan-out MUST
# exclude it too. When the head is the whole cluster it is kept (it has to run the work).
_HEAD_MARKER = "node:__internal_head__"


def _worker_eligible(nodes: list[dict]) -> list[dict]:
    """`nodes` minus the Ray head, matching worker placement — unless the head is the whole
    cluster (a single-node run must keep it). Sizing the fan-out from these keeps the worker
    count at what can actually be PLACED: counting the head made a data-heavy shuffle request
    one worker more than the schedulable node count, and the un-placeable actor hung the spawn
    (`ray.get` on its address never returned)."""
    non_head = [n for n in nodes if _HEAD_MARKER not in n.get("Resources", {})]
    return non_head or nodes


def _alive_nodes() -> list[dict]:
    """Worker-eligible alive node records (head excluded) — from the active snapshot if any,
    else a live `ray.nodes()`. Head-excluded so every fan-out sizing agrees with placement."""
    snap = _TOPOLOGY.get()
    if snap is not None:
        return _worker_eligible(snap.alive_nodes)
    import ray

    return _worker_eligible([n for n in ray.nodes() if n.get("Alive", True)])


def alive_node_count() -> int:
    """The number of alive nodes — snapshot-aware, so a placement/spread check does not
    trigger its own `ray.nodes()` RPC when a `topology_scope()` is active. Returns 0 on an
    unreadable topology (callers treat that as 'unknown')."""
    try:
        return len(_alive_nodes())
    except Exception:
        return 0


def shuffle_partitions(workers: int) -> int:
    """The number of shuffle partitions (reducers / hash buckets) for an all-to-all
    exchange over `workers` mappers — capped by `distributed.max_shuffle_partitions`.

    An exchange creates `mappers * reducers` streams; leaving the reducer count equal to the
    worker fan-out (one per node) makes it O(nodes^2), which collapses at 10k+ nodes. The
    reducer count only needs to balance keys and keep each reducer's state in memory, so it
    is capped: regular clusters (≤ the cap) are unchanged, huge clusters stay bounded.

    When prior runs have measured the shuffle families' real input volume, a learned reducer
    count (`learned_shuffle_fanout`) trims the fan-out for a shuffle whose measured data needs
    fewer, fuller buckets than one-per-worker — never above `workers`, so it only ever reduces
    the stream count. A cold store (no measured history) keeps the worker fan-out unchanged. Any
    reducer count is result-correct under the mergeable algebra, so this only affects scaling.
    Always at least 1; the cap is disabled when the config value is 0.
    """
    cap = active_config().distributed.max_shuffle_partitions
    n = max(1, workers)
    n = _learned_shuffle_fanout(n)
    return n if cap <= 0 else min(n, cap)


def _learned_shuffle_fanout(workers: int) -> int:
    """The learned reducer count for a shuffle over `workers` mappers, else `workers`.

    Best-effort read of the process-wide MetadataHub's measured shuffle-family input volume; any
    failure (no hub, cold store) returns `workers` unchanged."""
    try:
        from batcher.core import default_hub
        from batcher.dist.adaptive_sizing import learned_shuffle_fanout

        learned = learned_shuffle_fanout(default_hub(), None, workers)
        return learned if learned is not None else workers
    except Exception:  # pragma: no cover - learning is best-effort
        return workers


def cluster_topology() -> dict:
    """Live cluster shape: alive node count + total CPUs/GPUs. Ray must be up.

    Read on demand (not cached) so it stays correct as the autoscaler grows or shrinks the
    cluster — `ray.nodes()`/`ray.cluster_resources()` are cheap RPCs — unless a
    `topology_scope()` is active (the placement/transport phase), where every reader shares
    one snapshot.
    """
    snap = _TOPOLOGY.get()
    if snap is not None:
        nodes = _worker_eligible(snap.alive_nodes)
    else:
        import ray

        nodes = _worker_eligible([n for n in ray.nodes() if n.get("Alive", True)])
    # CPU/GPU/memory summed over the worker-eligible nodes (head excluded), so a CPU- or
    # memory-driven fit never counts a resource no worker will run on — consistent with the
    # head-excluded `nodes`. `min_node_memory` is the smallest worker node's RAM: the
    # SPREAD-safe per-worker memory ceiling (a grant sized above it would OOM the smallest
    # node it lands on), the hardware fact Carbonite sizes the distributed spill budget from.
    mem = [float(n.get("Resources", {}).get("memory", 0.0)) for n in nodes]
    return {
        "nodes": max(1, len(nodes)),
        "cpus": sum(float(n.get("Resources", {}).get("CPU", 0.0)) for n in nodes),
        "gpus": sum(float(n.get("Resources", {}).get("GPU", 0.0)) for n in nodes),
        "memory": sum(mem),
        "min_node_memory": min((m for m in mem if m > 0), default=0.0),
    }


def worker_node_memory_bytes() -> int:
    """The smallest worker node's RAM in bytes — the per-worker memory ceiling for a SPREAD
    fleet — or ``0`` when unknown (Ray down / no memory resource advertised).

    This is the hardware fact the distributed memory budget must respect: the driver may be a
    large box (e.g. a 197 GiB head) while workers are small (e.g. 34 GiB), so a budget sized
    from the driver's RAM would over-commit every worker. Sizing from the *worker* node keeps
    a distributed operator's spill threshold within the machine it actually runs on."""
    try:
        return int(cluster_topology().get("min_node_memory", 0.0))
    except Exception:
        return 0


def node_classes() -> list[dict]:
    """Per-alive-node resource class: ``{"cpus", "gpus", "accelerator_type"}``.

    The explicit cluster-heterogeneity model the scheduler lacked: a node is a "GPU
    node" when it exposes a `GPU` resource, a "CPU-only node" otherwise. The accelerator
    type comes from Ray's default `ray.io/accelerator-type` node label when present.
    Read on demand (Ray must be up) so it tracks autoscaler growth/shrink; empty when the
    topology is unreadable (the caller then keeps its homogeneous defaults).
    """
    try:
        out: list[dict] = []
        for n in _alive_nodes():
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


# A marker amount of the CPU-node custom resource — enough to *require* a labelled node
# without bounding how many tasks pack onto it (an affinity marker, not a capacity limit).
_CPU_NODE_EPS = 0.001


def node_class_selector(prefer_cpu_only: bool, workers: int, num_cpus: float) -> dict:
    """Ray resource fragment that HARD-restricts a fleet to CPU-only nodes, or ``{}``.

    Returns ``{"resources": {cpu_node_resource: _CPU_NODE_EPS}}`` — a requirement only
    nodes advertising the custom resource can satisfy — when the fleet asked to stay off
    GPU nodes (`prefer_cpu_only`), the config gate is on (`heterogeneous_node_isolation`),
    AND the live cluster's CPU-only nodes can actually host `workers x num_cpus` cores
    (`cpu_only_can_host`). Empty otherwise — a homogeneous / GPU-less cluster, an
    unreadable topology, the gate off, or CPU-only nodes too small — so the restriction
    can never make a query unschedulable (it then falls back to Ray's best-effort
    GPU-node avoidance, today's behavior).

    A custom resource is used rather than a node id / `NodeAffinitySchedulingStrategy`
    because it is a node *property* re-advertised when the autoscaler replaces a node, so
    the restriction survives spot churn; and because "GPU-absence" cannot be expressed as
    a soft node-label match. It is *additive* to Ray's soft `RAY_scheduler_avoid_gpu_nodes`
    — it makes the exclusion hard, so a CPU shuffle cannot steal an idle GPU node's cores.
    """
    if not prefer_cpu_only:
        return {}
    dc = active_config().distributed
    if not dc.heterogeneous_node_isolation:
        return {}
    if not cpu_only_can_host(workers, num_cpus):
        return {}
    return {"resources": {dc.cpu_node_resource: _CPU_NODE_EPS}}


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


def await_autoscale(target_cpus: int, target_gpus: float = 0.0) -> None:
    """Block (bounded, growth-detected) until the autoscaler grows the cluster toward
    `target_cpus` cores (and `target_gpus` GPUs).

    Called *before* the fan-out is sized to the cluster, so a query that triggered a
    scale-up (`request_autoscale`) fills the SCALED-UP cluster rather than the pre-scale
    one — the load-bearing step for out-of-the-box cluster saturation. Without it, the
    worker-per-node fill reads the current (small) topology and the query never uses the
    nodes it asked for; the wait inside `clamp_workers` can't fix that because the fill
    has already made `workers == capacity`.

    A no-op when the wait is disabled (`autoscale_wait_s <= 0`), Ray is down, or the
    cluster already covers the target. On a fixed cluster (or spot capacity the autoscaler
    cannot get) it returns quickly via `_await_autoscale`'s stall-window bail, so it never
    blocks the whole budget on nodes that will not arrive. Pure scheduling — the result is
    identical whether it waits or not.
    """
    if active_config().distributed.autoscale_wait_s <= 0 or target_cpus <= 0:
        return
    import ray

    if not ray.is_initialized():
        return
    topo = cluster_topology()
    _await_autoscale(target_cpus, int(topo["cpus"]), target_gpus, float(topo["gpus"]))


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
    already fits. Stops the instant capacity is sufficient, and — via the
    `autoscale_stall_s` grace window — also stops early once capacity has been flat that
    long (a fixed cluster, or spot capacity the autoscaler cannot get), so it never blocks
    the whole budget on nodes that will not arrive.
    """
    dc = active_config().distributed
    if dc.autoscale_wait_s <= 0 or (avail >= target_cpus and avail_gpus >= target_gpus):
        return avail
    import time

    deadline = time.monotonic() + dc.autoscale_wait_s
    poll = max(0.1, dc.autoscale_poll_s)
    # Give up early once capacity has been flat for the grace window: the autoscaler is
    # done (fixed cluster) or cannot satisfy the request (spot capacity unavailable), so
    # the rest of the budget would block on nodes that will not arrive. Any capacity gain
    # resets the window — a cluster that is still growing keeps its full wait.
    grace = max(dc.autoscale_stall_s, poll * 2)
    best = (avail, avail_gpus)
    last_growth = time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(min(poll, max(0.0, deadline - time.monotonic())))
        topo = cluster_topology()
        avail = int(topo["cpus"])
        avail_gpus = float(topo["gpus"])
        if avail >= target_cpus and avail_gpus >= target_gpus:
            break
        if (avail, avail_gpus) > best:
            best = (avail, avail_gpus)
            last_growth = time.monotonic()
        elif time.monotonic() - last_growth >= grace:
            break  # no new capacity for the grace window — nothing more is coming
    return avail

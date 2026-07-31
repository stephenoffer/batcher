"""What the live cluster is, and what of it a query may use.

Reads the cluster shape on demand (so it tracks autoscaler growth/shrink), narrows it to
the nodes a worker can actually be placed on *and kept* on — the head is excluded, and so
is anything Ray has marked for drain — and clamps a requested worker fan-out to that
schedulable capacity.

This is the *measuring* side. Asking the autoscaler for capacity lives in
`autoscale_request`, and waiting for what was asked to arrive lives in `readiness`.
"""

from __future__ import annotations

import contextlib
import contextvars
import math

from batcher._internal.accelerators import (
    accelerator_units,
    binding_gpu_memory_bytes,
    is_accelerator_node,
)
from batcher._internal.logging import note_suppressed
from batcher.config import active_config
from batcher.plan.resource import HardwareProfile


class _Topology:
    """A one-shot snapshot of the live cluster shape (alive node records + resources).

    Carries the drain list too, because it is part of "what is schedulable" and is read on
    the same paths: without it here, every `_worker_eligible` call inside a scope would make
    its own GCS round trip, which is the O(workers x nodes) cost this snapshot exists to
    remove.
    """

    __slots__ = ("alive_nodes", "draining", "resources")

    def __init__(
        self, alive_nodes: list[dict], resources: dict, draining: frozenset[str] = frozenset()
    ) -> None:
        self.alive_nodes = alive_nodes
        self.resources = resources
        self.draining = draining


# The topology snapshot in force for the current scheduling phase, if any. A distributed
# query reads the cluster shape from several places (transport choice, placement strategy,
# node-class selector, spread heuristic) — each an O(nodes) `ray.nodes()` RPC. Inside a
# `topology_scope()` they share ONE read. Ambient (a ContextVar) so it reaches those helpers
# without threading a snapshot through every call. Left unset (None) → every reader reads
# live, exactly as before.
_TOPOLOGY: contextvars.ContextVar[_Topology | None] = contextvars.ContextVar(
    "batcher_topology_snapshot", default=None
)


# How long a live drain read is reused. The list is polled from two places that run at very
# different rates: fan-out sizing reads it once per query, but the shuffle barrier polls it
# every `poll_seconds` (0.5 s) for the whole barrier, and an uncached read there is a GCS
# round trip twice a second for the length of a shuffle. A couple of seconds of staleness
# costs nothing — a drain notice precedes reclamation by tens of seconds at minimum — while
# an uncached read scales GCS load with barrier duration.
_DRAIN_TTL_S = 2.0
#: `(monotonic_deadline, node_ids)` for the last live read, or `None` before the first.
_drain_cache: tuple[float, frozenset[str]] | None = None


def _read_draining() -> frozenset[str]:
    """A TTL-cached read of Ray's drain list; empty on any failure.

    Deliberately unlocked: the worst a race does is two threads making the same GCS call and
    one overwriting the other's identical answer. A lock here would serialize every barrier
    poll in the process to protect against a benign duplicate read.
    """
    global _drain_cache
    import time

    cached = _drain_cache
    now = time.monotonic()
    if cached is not None and now < cached[0]:
        return cached[1]
    try:
        import ray._private.state as ray_state

        draining = frozenset(ray_state.state.get_draining_nodes() or ())
    except Exception as exc:
        note_suppressed("dist", "read draining nodes", exc)
        draining = frozenset()
    _drain_cache = (now + _DRAIN_TTL_S, draining)
    return draining


def _reset_drain_cache() -> None:
    """Drop the cached drain read (tests, and any caller wanting a fresh probe)."""
    global _drain_cache
    _drain_cache = None


def draining_node_ids() -> frozenset[str]:
    """Hex ids of nodes Ray has marked for drain, or empty when that cannot be read.

    A draining node is alive, advertises its full resources, and is going away — the
    autoscaler is scaling it in, KubeRay is evicting its pod, or a spot reclamation notice
    reached the node provider. Ray keeps reporting it as schedulable because tasks already
    on it must finish, but placing *new* work there loses that work minutes later.

    Every fan-out sizing in this module counted those nodes. On an autoscaling cluster mid
    scale-in, or a spot fleet mid churn, that means the fleet is sized to capacity already
    committed to disappearing: the placement group reserves bundles on a node being removed,
    the actors land, and the shuffle pays a recompute for output that was never going to
    survive. The drain list is the one signal separating "alive" from "alive and staying",
    and it arrives *before* the loss rather than after.

    Snapshot-aware: inside a `topology_scope()` this is the set read once for the whole
    scheduling phase, so a W-worker fleet costs one GCS round trip rather than W.

    Best-effort: `get_draining_nodes` is a private accessor, so a Ray version without it —
    or a GCS that will not answer — degrades to today's behavior (count every alive node)
    rather than failing a query over a scheduling refinement.
    """
    snap = _TOPOLOGY.get()
    if snap is not None:
        return snap.draining
    return _read_draining()


def _schedulable(nodes: list[dict]) -> list[dict]:
    """`nodes` minus those Ray is draining — unless that would leave nothing.

    A fleet must still be placeable on a cluster where *every* remaining node is draining:
    running on capacity that is going away beats not running at all, and the recovery path
    (recompute, replication, proactive migration) exists for exactly that case. So this only
    ever removes nodes when survivors remain.
    """
    draining = draining_node_ids()
    if not draining:
        return nodes
    staying = [n for n in nodes if n.get("NodeID") not in draining]
    return staying or nodes


def _read_topology() -> _Topology:
    import ray

    return _Topology(
        [n for n in ray.nodes() if n.get("Alive", True)],
        ray.cluster_resources(),
        _read_draining(),
    )


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
    """`nodes` minus the Ray head and minus anything Ray is draining, matching worker
    placement — unless that would leave nothing (a single-node run must keep the head; a
    fully-draining cluster must still run somewhere). Sizing the fan-out from these keeps the
    worker count at what can actually be PLACED *and kept*: counting the head made a
    data-heavy shuffle request one worker more than the schedulable node count, and the
    un-placeable actor hung the spawn (`ray.get` on its address never returned); counting a
    draining node places work that is lost when the node goes (see `_schedulable`)."""
    non_head = [n for n in nodes if _HEAD_MARKER not in n.get("Resources", {})]
    return _schedulable(non_head or nodes)


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
    except Exception as exc:  # pragma: no cover - learning is best-effort
        note_suppressed("dist", "read learned shuffle fan-out", exc)
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
    """Per-alive-node resource class:
    ``{"cpus", "gpus", "memory", "accelerators", "accelerator_type"}``.

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
                    # Per-node RAM, so a placement check can bound by the resource a
                    # bundle reserves alongside cores. `0.0` when the node advertises no
                    # `memory` resource, which callers read as "do not bound by memory"
                    # rather than "this node has none".
                    "memory": float(res.get("memory", 0.0)),
                    # Non-GPU accelerators (TPU / Trainium / Gaudi / NPU) Ray doesn't count as
                    # `GPU`; lets the CPU-fleet isolation treat a TPU node as an accelerator node.
                    "accelerators": accelerator_units(res),
                    "accelerator_type": labels.get("ray.io/accelerator-type"),
                }
            )
        return out
    except Exception as exc:
        note_suppressed("dist", "read node classes", exc)
        return []


def cluster_hardware_profile() -> HardwareProfile | None:
    """The `HardwareProfile` Kyber should plan against for a distributed run, or `None`.

    Built from live topology so the optimizer's cache/memory-sized thresholds track the
    *workers*, not the driver — which may be a fat head node next to small workers. Every
    field is the **binding** (weakest) worker so a plan sized against it is valid on every
    node it might land on: cores from the smallest worker, RAM from `worker_node_memory_bytes`
    (already the minimum), VRAM from the smallest device model any GPU node advertises, and L3
    cache from the smallest-cache node shape (probed from the workers — Ray's topology omits it,
    so this used to be `0` and every cluster query fell back to the config broadcast threshold).

    `gpu_count` is the cluster's **device** total, not the number of GPU-bearing nodes. Those
    differ on any multi-GPU node, and the figure is consumed as a device count (a whole-cluster
    VRAM budget is `one_gpu_gb * gpu_count`), so reporting nodes under-counted an 8-GPU box
    eightfold and refused work the cluster could hold.

    Returns `None` when the topology is unreadable (Ray down), so the caller falls back to the
    single-node local profile rather than a fabricated one.
    """
    classes = node_classes()
    if not classes:
        return None
    worker_count = len(classes)
    min_cores = min((int(c["cpus"]) for c in classes if c["cpus"] > 0), default=0)
    gpu_devices = int(sum(c["gpus"] for c in classes))
    from batcher.dist.executors.ray_runtime.hardware_probe import (
        cluster_l3_cache_bytes,
        warn_once_if_fleet_is_mixed,
    )

    warn_once_if_fleet_is_mixed()

    return HardwareProfile.for_cluster(
        cpu_cores=min_cores,
        memory_bytes=worker_node_memory_bytes(),
        worker_count=worker_count,
        gpu_count=gpu_devices,
        gpu_memory_bytes=binding_gpu_memory_bytes(classes),
        # The binding worker's L3, probed from the workers themselves — Ray's topology omits
        # cache, so this was left `0` and every cluster query fell back to the config broadcast
        # threshold. Cached per topology and best-effort, so an unprobeable cluster is unchanged.
        l3_cache_bytes=cluster_l3_cache_bytes(),
    )


def cpu_only_can_host(workers: int, num_cpus: float) -> bool:
    """Whether the cluster's **CPU-only** nodes alone can host `workers` x `num_cpus` cores.

    The gate for keeping a relational (CPU) fleet off accelerator nodes on a heterogeneous
    cluster: only restrict the fleet to CPU-only nodes when those nodes have the capacity
    to run it — otherwise the restriction would under-provision (or fail to place) the
    query, so the fleet is left free to use every node (today's behavior). Returns False
    on a homogeneous cluster (no accelerator nodes ⇒ nothing to keep off) or unreadable
    topology. Accelerator = GPU or custom accelerator (see `is_accelerator_node`).
    """
    classes = node_classes()
    if not classes or not any(is_accelerator_node(c) for c in classes):
        return False  # homogeneous / accelerator-less → no restriction (use all nodes)
    cpu_only_cores = sum(c["cpus"] for c in classes if not is_accelerator_node(c))
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


def clamp_workers(
    workers: int,
    num_cpus: float = 1.0,
    num_gpus: float = 0.0,
    *,
    memory_bytes: int = 0,
    cpu_only: bool = False,
) -> int:
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
    `memory_bytes` and `cpu_only` describe the rest of what the fleet's bundle reserves —
    the per-worker RAM grant, and whether the fleet is held to non-accelerator nodes. Both
    narrow what can host a worker, and omitting them overstates capacity in the one
    direction that hangs the job (see `capacity.placeable_workers`).

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
    # ...and by what a single node can *host* (see `capacity.placeable_workers`).
    from batcher.dist.executors.ray_runtime.capacity import placeable_workers
    from batcher.dist.executors.ray_runtime.readiness import _await_autoscale

    fits = placeable_workers(num_cpus, num_gpus, memory_bytes=memory_bytes, cpu_only=cpu_only)
    capacity = capacity if fits is None else min(capacity, fits)
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
    # Re-read after the wait: the grown cluster's shape decides what fits, not its totals.
    fits_now = placeable_workers(num_cpus, num_gpus, memory_bytes=memory_bytes, cpu_only=cpu_only)
    fit = fit if fits_now is None else min(fit, fits_now)
    return max(1, min(workers, fit))

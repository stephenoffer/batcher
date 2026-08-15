"""How many workers a cluster can actually *place*, as opposed to afford.

Fan-out sizing reads the cluster as totals — `floor(total_cpus / num_cpus)`. On a homogeneous
cluster that is the same number a per-node count would give, which is why the difference went
unnoticed. On a heterogeneous one it is not: four 8-core nodes total 32 cores, so the sum says
two `num_cpus=16` workers fit, while every individual node is too small to host even one. The
grant is uniform and placement is per node, so the sum is an upper bound that no arrangement
of nodes can reach.

That gap is not a rounding error. Ray schedules the fleet as a gang, so a fan-out sized above
what any node can host leaves the placement group permanently unsatisfiable — the job hangs
rather than failing, which is the failure mode `clamp_workers` exists to prevent.

Related: `dist.executor._cluster_fill_workers` slices nodes the same way when *choosing* a
fan-out. This module answers the complementary question — whether a fan-out already chosen can
be placed — and the two must keep using the same `floor(node_cores / num_cpus)` rule.

`describe_pending_demand` is that same question asked about a request Ray has *already*
refused to place, and phrased for a person: a task waiting on a busy cluster finishes
eventually, a task asking for more CPUs than any node has never runs, and `ray.wait` cannot
tell the two apart. Neither could the engine, which is why every stalled barrier used to
report the same "go run `ray status`" whichever it was.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher._internal.logging import get_logger, note_suppressed

__all__ = [
    "Demand",
    "describe_pending_demand",
    "free_cpus_by_node",
    "placeable_workers",
    "preferred_fleet_zone",
    "warn_once_if_allocation_is_wider_than_ray",
    "workers_per_node",
]


def free_cpus_by_node() -> dict[str, float] | None:
    """Cores each node has *unreserved* right now, or `None` when Ray will not say.

    `node_classes` reports nameplate capacity, which is the right figure for classifying a
    node and the wrong one for deciding what can be *placed* on it. A fleet reserves one
    worker per node holding that node's cores, so a single core held by anything else — a
    co-tenant job, another pipeline, a placement group the last query has not finished
    releasing — makes that whole bundle unplaceable. The gang then pends until the placement
    timeout and the query fails with `no distributed worker became available`, on a cluster
    that is almost entirely idle.

    Measured on this four-node fleet with one core busy per node: `4 bundles x 8 CPU` is
    unsatisfiable while `28 bundles x 1 CPU` places immediately, and every distributed query
    failed after three sixty-second waits until the grant was sized from these numbers
    instead of the nameplate.

    Snapshot-aware: inside a `scaling.topology_scope()` this is the figure read once for the
    whole scheduling phase. Without that it was the hole in the snapshot — `node_classes`
    reads it on every call, and the placement phase calls `node_classes` from five places, so
    a scope that had carefully collapsed its `ray.nodes()` reads still paid five GCS round
    trips for the free-CPU half of the same question.

    Returns:
        Node id -> free CPU count, or `None` when the per-node figures cannot be read. `None`
        means "assume nameplate", which is the behaviour every caller had before.
    """
    from batcher.dist.executors.ray_runtime.scaling import _TOPOLOGY

    snap = _TOPOLOGY.get()
    if snap is not None:
        return snap.free_cpus
    return _live_free_cpus_by_node()


def _live_free_cpus_by_node() -> dict[str, float] | None:
    """The unsnapshotted read behind `free_cpus_by_node`, and what fills the snapshot.

    Windowed by `scaling._LIVE_TTL_S` like the node list, and for the same measurement: this
    was seven `available_resources_per_node` round trips per distributed query, each O(nodes)
    in GCS work and deserialization, all asking the same question within a few milliseconds.
    See that constant for why the window is 50 ms and why that is safe.
    """
    import time

    from batcher.dist.executors.ray_runtime import scaling

    cached = scaling._free_cpus_cache
    now = time.monotonic()
    if cached is not None and now < cached[0]:
        return cached[1]
    try:
        from ray._private.state import available_resources_per_node

        free: dict[str, float] | None = {
            node_id: float(res.get("CPU", 0.0))
            for node_id, res in available_resources_per_node().items()
        }
    except Exception as exc:
        # A private Ray API, so a version that moves it must degrade rather than fail: the
        # caller falls back to nameplate sizing, which is what it did before this existed.
        note_suppressed("dist", "read per-node free CPU", exc)
        free = None
    scaling._free_cpus_cache = (now + scaling._LIVE_TTL_S, free)
    return free


def placeable_workers(
    num_cpus: float,
    num_gpus: float = 0.0,
    *,
    memory_bytes: int = 0,
    cpu_only: bool = False,
) -> int | None:
    """Workers that fit when each is placed on a single node, or `None` if unknown.

    Sums each node's own capacity rather than dividing the cluster's total, so a node too
    small to host one worker contributes zero instead of a fraction of one.

    The count must be taken over the nodes the fleet may actually land on, and against
    every resource its bundle reserves. Both were previously ignored, and each produces the
    same failure — a fan-out above what any arrangement of nodes can host, which leaves the
    gang-scheduling placement group permanently unsatisfiable and hangs the job:

    * `cpu_only` mirrors the node-class restriction (`scaling.node_class_selector`). When a
      relational fleet is held off accelerator nodes, those nodes' cores cannot host it, so
      counting them overstates capacity by exactly the GPU half of a mixed cluster.
    * `memory_bytes` is reserved by the bundle (`scheduling._bundle`), so a node with spare
      cores but no spare RAM hosts zero workers however many cores it has. On the memory
      -heavy shuffles this grant exists for, RAM binds well before cores do.

    Args:
        num_cpus: CPU shares each worker requests. Values at or below zero report unknown.
        num_gpus: GPUs each worker requests; `0` for a CPU-only fleet, which then bounds by
            cores alone.
        memory_bytes: Heap bytes each worker's bundle reserves; `0` skips the memory bound
            (the caller had no memory grant to place).
        cpu_only: Count only non-accelerator nodes, matching a fleet restricted to them.

    Returns:
        The number of placeable workers, or `None` when the topology is unreadable — the
        caller then keeps its total-based estimate rather than clamping on a guess.
    """
    if num_cpus <= 0:
        return None
    try:
        from batcher._internal.accelerators import is_accelerator_node
        from batcher.dist.executors.ray_runtime.scaling import node_classes

        nodes = node_classes()
    except Exception:
        return None
    if not nodes:
        return None
    if cpu_only:
        nodes = [n for n in nodes if not is_accelerator_node(n)]
        if not nodes:
            # The restriction is only ever applied when CPU-only nodes can host the fleet
            # (`cpu_only_can_host`), so an empty list here means the topology changed under
            # us. Report unknown rather than zero: zero would clamp the fan-out to one.
            return None
    demand = Demand(num_cpus=num_cpus, num_gpus=num_gpus, memory_bytes=memory_bytes)
    # Nameplate, because this answers what the cluster can host rather than what is free
    # right now — the fan-out is sized before the fleet is placed, and a co-tenant that
    # finishes in the meantime must not have shrunk it.
    total = sum(workers_per_node(node, demand, nameplate=True) for node in nodes)
    return total


@dataclass(frozen=True, slots=True)
class Demand:
    """What one pending task or placement-group bundle is asking for.

    Mirrors the fields `scheduling._bundle` reserves, so a diagnosis is made against the ask
    Ray was actually given rather than an approximation of it.

    * `num_cpus` — CPU shares one task/bundle requests.
    * `num_gpus` — GPUs one task/bundle requests; `0.0` for the CPU relational path.
    * `memory_bytes` — heap bytes reserved per task/bundle; `0` when nothing was granted.
    * `resources` — custom Ray resources (`TPU`, `neuron_cores`, an operator's own name).
    * `count` — how many are outstanding, for the message only.
    """

    num_cpus: float = 1.0
    num_gpus: float = 0.0
    memory_bytes: int = 0
    resources: tuple[tuple[str, float], ...] = ()
    count: int = 1

    @classmethod
    def from_envelope(cls, env, count: int | None = None) -> Demand:
        """The demand a `SchedulingEnvelope`'s per-task grant places on one node.

        Args:
            env: The `SchedulingEnvelope` in force, or `None` for Ray's implicit one CPU.
            count: Outstanding tasks; defaults to the envelope's fan-out.

        Returns:
            The `Demand` for one of that envelope's tasks.
        """
        if env is None:
            return cls(count=max(1, count or 1))
        return cls(
            num_cpus=float(env.num_cpus),
            num_gpus=float(env.num_gpus),
            memory_bytes=int(env.memory_bytes),
            resources=tuple(env.resources),
            count=max(1, count if count is not None else env.n_tasks),
        )


def _node_shortfall(node: dict, demand: Demand) -> str | None:
    """The resource `node`'s **nameplate** cannot cover for one `demand`, or `None` if it fits.

    Nameplate, not free: this answers "can this ever be placed", and a busy node is a
    scheduling delay rather than an impossibility. `placeable_workers` asks the
    complementary question against what is free right now.

    Returns the *binding* resource rather than a bare bool because that is the whole
    actionable content of the answer. Told only that a node cannot host the task, a reader
    compares the numbers they can see — cores against cores — and concludes the engine is
    wrong when the constraint was RAM.
    """
    if float(node.get("cpus", 0.0)) < demand.num_cpus:
        return f"CPU ({float(node.get('cpus', 0.0)):g} available, {demand.num_cpus:g} needed)"
    if demand.num_gpus > 0 and float(node.get("gpus", 0.0)) < demand.num_gpus:
        return f"GPU ({float(node.get('gpus', 0.0)):g} available, {demand.num_gpus:g} needed)"
    node_memory = float(node.get("memory", 0.0))
    # A node advertising no `memory` resource is not a node with no memory — Ray simply is
    # not tracking it — so an unreported figure cannot rule the node out. Same reading
    # `placeable_workers` takes, and for the same reason.
    if demand.memory_bytes > 0 and 0 < node_memory < demand.memory_bytes:
        return (
            f"memory ({node_memory / 1e9:.1f} GB available, "
            f"{demand.memory_bytes / 1e9:.1f} GB needed)"
        )
    return None


def _missing_custom_resources(demand: Demand) -> list[str]:
    """Custom resources the cluster advertises less of than one task needs.

    Checked against the cluster *total* rather than per node, because `node_classes`
    deliberately does not carry them — it classifies nodes, and an operator's own resource
    names are unbounded. That is the weaker test (it cannot see units spread thin across
    nodes), which is the right way to be wrong: it claims an impossibility only when the
    resource is genuinely absent.
    """
    if not demand.resources:
        return []
    try:
        import ray

        totals = dict(ray.cluster_resources())
    except Exception as exc:
        note_suppressed("dist", "read cluster resources for the demand diagnosis", exc)
        return []
    return [name for name, amount in demand.resources if totals.get(name, 0.0) < amount]


def _ask(demand: Demand) -> str:
    """The demand as a resource phrase a reader can match against `ray status`."""
    parts = [f"{demand.num_cpus:g} CPU"]
    if demand.num_gpus > 0:
        parts.append(f"{demand.num_gpus:g} GPU")
    if demand.memory_bytes > 0:
        parts.append(f"{demand.memory_bytes / 1e9:.1f} GB")
    parts.extend(f"{amount:g} {name}" for name, amount in demand.resources)
    return ", ".join(parts)


def describe_pending_demand(demand: Demand) -> str | None:
    """One sentence naming why `demand` has not been placed, or `None` if nothing is wrong.

    Three outcomes, in the order a reader needs them:

    * **Unsatisfiable** — no node in the cluster is large enough to host one of these, so
      waiting cannot help. Names what was asked and what the widest node holds, because the
      fix is always to change one of the two.
    * **Short** — the nodes that could host it do not have enough free capacity for what is
      outstanding. Reported as the three numbers rather than as a verdict, because the same
      shape means different things to the two callers: a gang needs every bundle at once, so
      it is why the group will not form, while a barrier's tasks queue happily and it is
      merely why nothing has started. Both callers ask only after a stall, and at that point
      the numbers are what a reader needs either way.
    * **`None`** — there is room for what is outstanding and the topology has no complaint.
      The wait is a slow task, not a scheduling problem, and manufacturing a diagnosis for it
      would train readers to skip the ones that mean something.

    Args:
        demand: The per-task or per-bundle ask that is pending.

    Returns:
        The diagnosis, or `None` when the topology reports nothing actionable.
    """
    from batcher.dist.executors.ray_runtime.scaling import node_classes

    try:
        nodes = node_classes()
    except Exception as exc:  # pragma: no cover - a diagnosis never fails its caller
        note_suppressed("dist", "read node classes for the demand diagnosis", exc)
        return None
    if not nodes:
        return None

    missing = _missing_custom_resources(demand)
    if missing:
        return (
            f"no node advertises {', '.join(missing)}: this stage asks for {_ask(demand)} per "
            f"task and the cluster has none of that resource, so waiting cannot schedule it"
        )

    shortfalls = {id(n): _node_shortfall(n, demand) for n in nodes}
    hosts = [n for n in nodes if shortfalls[id(n)] is None]
    if not hosts:
        # Report against the widest node rather than an arbitrary one: it is the node a
        # reader would compare the ask to, and the one whose shape has to change.
        widest = max(nodes, key=lambda n: float(n.get("cpus", 0.0)))
        return (
            f"no node can host one task: this stage asks for {_ask(demand)} per task and the "
            f"widest of {len(nodes)} node(s) is short on {shortfalls[id(widest)]}, so waiting "
            f"cannot schedule it"
        )

    free = sum(float(n.get("free_cpus", n.get("cpus", 0.0))) for n in hosts)
    if free < demand.num_cpus * demand.count:
        return (
            f"the cluster is short of free capacity: {demand.count} outstanding at "
            f"{_ask(demand)} each, and {len(hosts)} candidate node(s) have {free:g} CPU free "
            f"between them — another job, or an earlier stage's placement group, is holding "
            f"the rest"
        )
    return None


def preferred_fleet_zone(workers: int, demand: Demand) -> dict[str, str]:
    """A one-zone label selector for a shuffle fleet, or `{}` to place it anywhere.

    A shuffle moves nearly all of its bytes worker to worker, and on every cloud those bytes
    are billed and delayed differently depending on whether the two workers sit in the same
    availability zone. AWS charges $0.01/GB in *each* direction across zones and adds a
    round-trip; a fleet spread evenly over three zones sends roughly two thirds of its
    shuffle across that boundary. Nothing about the query requires it: the bundles are
    interchangeable, so a fleet that fits inside one zone can simply be placed inside one.

    The zone chosen is the one with the most free capacity among those that can host the
    *whole* fleet, so pinning never trades a cost saving for a placement that will not form.
    Everything else returns `{}` and the fleet is placed exactly as it was before:

    * a cluster in one zone, or one whose nodes carry no zone label — nothing to choose;
    * a fleet no single zone can host — spreading it is the only arrangement available;
    * an unreadable topology — a cost optimization must not act on a guess.

    Placement-only, so the result is identical either way. And it is applied to the bundles
    rather than to the tasks, which is what keeps the failure mode benign: a group that does
    not form within the timeout is abandoned and the stage falls back to default scheduling,
    where a zone pin on the tasks themselves would instead leave them pending forever.

    Args:
        workers: Bundles being reserved.
        demand: What one bundle asks for.

    Returns:
        `{label_key: zone}` to pin the fleet, or `{}` to leave placement unconstrained.
    """
    from batcher.dist.executors.ray_runtime.scaling import node_classes

    try:
        nodes = node_classes()
    except Exception as exc:  # pragma: no cover - a cost hint never fails a placement
        note_suppressed("dist", "read node classes for zone-aware placement", exc)
        return {}
    zoned = [n for n in nodes if n.get("zone")]
    if len(zoned) < 2 or len({n["zone"] for n in zoned}) < 2:
        return {}
    best: tuple[float, str, str] | None = None
    for zone in {n["zone"] for n in zoned}:
        members = [n for n in zoned if n["zone"] == zone]
        fits = sum(workers_per_node(n, demand) for n in members)
        if fits < workers:
            continue
        free = sum(float(n.get("free_cpus", n.get("cpus", 0.0))) for n in members)
        key = members[0].get("zone_label") or ""
        if key and (best is None or free > best[0]):
            best = (free, key, zone)
    return {best[1]: best[2]} if best is not None else {}


def workers_per_node(node: dict, demand: Demand, *, nameplate: bool = False) -> int:
    """How many workers of `demand` one node can host — the single per-node placement rule.

    Every question this module answers reduces to it: how wide a fan-out the cluster can
    place, which zone can hold a fleet, and whether the widest node can host one task at all.
    Stated three times it drifted three ways, and a per-node rule that disagrees with itself
    produces a fan-out no arrangement of nodes can satisfy — which hangs the job rather than
    failing it.

    Args:
        node: One `scaling.node_classes` record.
        demand: What one worker asks for.
        nameplate: Size against the node's full capacity rather than what is unreserved now.
            True when sizing a fan-out, which happens before the fleet is placed and must not
            shrink because a co-tenant was momentarily busy. False when deciding where a
            reservation can actually form.

    Returns:
        The worker count, never negative.
    """
    cpus = float(node.get("cpus", 0.0))
    if not nameplate:
        cpus = float(node.get("free_cpus", cpus))
    fits = int(cpus // max(demand.num_cpus, 1e-9))
    if demand.num_gpus > 0:
        fits = min(fits, int(float(node.get("gpus", 0.0)) // demand.num_gpus))
    node_memory = float(node.get("memory", 0.0))
    # Only bound by memory where the node actually reports it. A node advertising no `memory`
    # resource is not a node with no memory — Ray simply is not tracking it — and reading the
    # absent value as zero would place zero workers everywhere and collapse the fan-out to one.
    if demand.memory_bytes > 0 and node_memory > 0:
        fits = min(fits, int(node_memory // demand.memory_bytes))
    return max(0, fits)


# Set once the allocation-width notice has been given. The shape of a job does not change
# under it, and repeating the notice per query trains the reader to skip it.
_ALLOCATION_WARNED = False


def warn_once_if_allocation_is_wider_than_ray() -> None:
    """Say so, once, when a batch allocation spans more nodes than Ray is using.

    The silent failure this exists for: `srun -N 4 python job.py` — or the PBS and LSF
    equivalents — gives the job four nodes, and a bare `ray.init()` starts a *local*
    single-node Ray on whichever one the script happens to be on. The job runs, returns the
    right answer, and uses a quarter of the hardware it was billed for — with nothing anywhere
    to say so, because from Batcher's side
    a one-node cluster is a perfectly ordinary cluster.

    Batcher cannot fix it from here: bringing up Ray across a batch allocation is the
    launcher's job (`ray start` on each node, then `RAY_ADDRESS`). So this reports rather
    than acts, and reports only where the two figures actually disagree — a single-node
    allocation, an unscheduled process, and a Ray cluster that already spans the allocation
    all say nothing.
    """
    global _ALLOCATION_WARNED
    if _ALLOCATION_WARNED:
        return
    from batcher._internal.site import scheduler_job

    job = scheduler_job()
    if not job.multi_node:
        return
    try:
        import ray

        if not ray.is_initialized():
            return
        ray_nodes = len([n for n in ray.nodes() if n.get("Alive", True)])
    except Exception as exc:
        note_suppressed("dist", "compare the Ray cluster against the allocation", exc)
        return
    if ray_nodes >= len(job.nodes):
        return
    _ALLOCATION_WARNED = True
    get_logger("dist").warning(
        "this %s job holds %d nodes but Ray sees %d: the run will use one node's worth of "
        "the allocation. Start Ray across the allocation (`ray start --head` on one node, "
        "`ray start --address` on the rest) and point %s at it.",
        job.kind,
        len(job.nodes),
        ray_nodes,
        "RAY_ADDRESS",
    )

"""The distributed executor — the dispatcher.

Inspects a plan's shape and routes it to the matching distributed operator
(map_batches, aggregate, join, sort, distinct, window, union), each of which reuses
the engine's mergeable primitives so its result is identical to single-node
execution. Shapes that can't be distributed yet fall back to the multi-core
single-node engine.

The per-operator implementations live in the `executors` subpackage; plan
analysis in `executors.plan_analysis`, partitioning/post-breaker helpers in
`executors.partition_io`, and Ray lifecycle + fallback in `executors.ray_runtime`.
The internal helpers re-exported
here (`_apply_above`, `_empty_agg_table`, `_ensure_ray`, `_partition_source`,
`_relabel_single_source`, `_rmtree`) keep `from batcher.dist.executor import ...`
working for the Flight and spill paths.
"""

from __future__ import annotations

import dataclasses
import json
import math

import pyarrow as pa

from batcher._internal.hardware import available_cpu_count
from batcher._internal.logging import note_suppressed

# Re-exported (`X as X`) so the Flight + spill paths can keep importing these
# helpers from `batcher.dist.executor` after the split.
from batcher.dist.executors.partition_io import _apply_above as _apply_above
from batcher.dist.executors.partition_io import _partition_source as _partition_source
from batcher.dist.executors.partition_io import source_pushdown
from batcher.dist.executors.plan_analysis import _empty_agg_table as _empty_agg_table

# Used by the dispatcher below.
from batcher.dist.executors.plan_analysis import (
    _has_breaker,
    _is_linear_map_pipeline,
    _is_row_wise,
    _single_source,
    _split_at,
    empty_result_table,
    shuffle_branches,
)
from batcher.dist.executors.plan_analysis import _relabel_single_source as _relabel_single_source
from batcher.dist.executors.ray_runtime import _ensure_ray as _ensure_ray
from batcher.dist.executors.ray_runtime import _rmtree as _rmtree
from batcher.dist.executors.ray_runtime import (
    _single_node,
    alive_node_count,
    await_autoscale,
    clamp_workers,
    engine_config_json,
    node_class_selector,
    release_autoscale,
    request_autoscale,
    reset_scheduling_envelope,
    resolve_transport,
    set_scheduling_envelope,
    topology_scope,
    worker_node_memory_bytes,
)
from batcher.dist.executors.ray_runtime.trace import FanoutTrace
from batcher.dist.fleet.plan_id import with_query_shuffle_scope

# The *predicate* only, imported eagerly: `global_window.offsets` sees nothing but `plan`,
# where the executors beside it import this module back (for `_apply_above` and friends) and
# so must stay lazy at their call sites, exactly as `flight_window` does.
from batcher.dist.global_window.offsets import supports_ordered_bucket_offsets
from batcher.io.source import Source
from batcher.plan.expr_ir import Col
from batcher.plan.ir_specs import binary_task_ir
from batcher.plan.logical import (
    Aggregate,
    AsofJoin,
    Distinct,
    Join,
    Limit,
    LogicalPlan,
    RangeJoin,
    RowId,
    Sample,
    Scan,
    Sort,
    Union,
    Window,
)
from batcher.plan.resource import SchedulingEnvelope
from batcher.plan.visitor import scanned_source_ids

__all__ = ["execute_distributed", "resolve_worker_fanout"]


def resolve_worker_fanout(num_workers: int | None) -> int:
    """The worker fan-out for a distributed stage the caller did not size explicitly.

    `execute_distributed` derives its fan-out from the live cluster (`_cluster_fill_workers`
    — enough workers to fill every node's cores). Paths that bypass it and drive Ray tasks
    directly — notably the distributed **write** — must not invent their own constant: a
    hard-coded fan-out uses 4 workers on a 100-node cluster, stranding 96 nodes. This is
    that same sizing, exposed for those callers. An explicit `num_workers` always wins.
    """
    if num_workers is not None:
        return max(1, num_workers)
    _ensure_ray(available_cpu_count())
    fill = _cluster_fill_workers()
    return fill[0] if fill is not None else available_cpu_count()


@with_query_shuffle_scope
def execute_distributed(
    plan: LogicalPlan,
    sources: list[Source],
    num_workers: int | None = None,
    transport: str = "disk",
    envelope: SchedulingEnvelope | None = None,
    hub=None,
    *,
    materialize: bool = True,
    metrics_out=None,
):
    """Execute a plan across Ray workers, falling back to single-node when needed.

    `transport="flight"` shuffles aggregation partials over Arrow Flight between
    worker actors (object store bypassed) instead of the default disk Arrow-IPC
    files; the result is identical. `transport="auto"` (the surface default) is
    resolved here from cluster topology — Flight on a real multi-node cluster
    (where the disk shuffle's driver-local `work_dir` is unreachable cross-node),
    disk on a single node / shared filesystem.

    `envelope` is Carbonite's metadata-driven scheduling grant: its `n_tasks` sets
    the worker fan-out (replacing a blind `os.cpu_count()`) and its per-task
    resources are applied to every Ray task via `.options(...)` at wrap time. It is
    installed as the ambient grant for the duration of this call. `hub` lets the
    GPU map/inference path record measured utilization for next-run adaptation.

    `materialize=False` lets a stage keep its result partitioned on disk and return a
    `MaterializedSource` (over the disk-shuffle path, where it is supported) instead
    of collecting every reducer's output back to the driver — the adaptive executor
    scans that intermediate in place for the next stage. Shapes that don't support it
    still return a collected `pa.Table`, so the caller must handle either.
    """
    if envelope is not None and num_workers is None:
        workers = max(1, envelope.n_tasks)
    else:
        workers = num_workers or available_cpu_count()

    # Set the grant first so the up-front `_ensure_ray` wraps tasks with it; then ask
    # the autoscaler for the cores this query wants (released in the `finally` so a
    # one-off big job doesn't pin the cluster scaled-up), clamp the fan-out to
    # schedulable capacity, and pick the transport from the resulting topology.
    num_cpus, num_gpus = (envelope.num_cpus, envelope.num_gpus) if envelope else (1.0, 0.0)
    accel = tuple(envelope.resources) if envelope else ()
    token = set_scheduling_envelope(envelope)
    request_autoscale(
        math.ceil(workers * num_cpus),
        workers * num_gpus,
        # Scale the per-task accelerator ask by the fan-out, as the GPU floor is.
        tuple((name, amount * workers) for name, amount in accel),
    )
    try:
        _ensure_ray(workers)
        # Wait (bounded, growth-detected) for the autoscaler to bring the cluster toward
        # the cores this query asked for BEFORE sizing the fan-out — otherwise the
        # worker-per-node fill below reads the pre-scale (small) topology and the query
        # never uses the nodes it triggered (the wait inside `clamp_workers` can't help
        # once the fill has made `workers == capacity`). A fixed cluster / unavailable
        # capacity bails fast via the stall window; disabled by `autoscale_wait_s <= 0`.
        # Pure scheduling — the result is identical whether it waits or not.
        if num_workers is None:
            await_autoscale(math.ceil(workers * num_cpus), workers * num_gpus)
        # On a multi-node cluster, fan out to exactly ONE worker per node, each owning
        # that node's cores — the cluster-filling, evenly-distributed shape. This (a) uses
        # every node for any non-trivial query (max, even utilization — more workers than
        # nodes can't add CPU parallelism since cores are the limit, but each node-worker
        # saturates its cores via morsel parallelism + spill), (b) keeps shuffle fan-out
        # minimal (one bucket stream per node, not per core), and (c) gives the reused
        # session fleet a stable, adequate size independent of which query first spawned
        # it — the data-driven count would size the fleet to the first (maybe tiny) query
        # and then under-provision a later big one. An explicit `num_workers` overrides it;
        # a single node falls back to the data-driven `_even_cpu_share` path below.
        trace = FanoutTrace(workers)
        # A stage that holds a device is tiled by devices, not by cores: the core-shaped fill
        # below caps the fan-out at the node count, which on a multi-device node leaves most
        # of the fleet's accelerators idle. Falls back to the core fill whenever that is not
        # what this stage is (no device grant, no accelerator nodes, unreadable topology).
        fill = None
        by_device = False
        if num_workers is None:
            fill = _accelerator_fill_workers(num_gpus)
            by_device = fill is not None
            if fill is None:
                fill = _cluster_fill_workers()
        if fill is not None:
            desired, (workers, num_cpus) = workers, fill
            mem = (
                int(envelope.memory_bytes * max(1, desired) / workers)
                if envelope is not None and envelope.memory_bytes
                else (envelope.memory_bytes if envelope is not None else 0)
            )
            envelope = (
                dataclasses.replace(envelope, n_tasks=workers, num_cpus=num_cpus, memory_bytes=mem)
                if envelope is not None
                else SchedulingEnvelope(num_cpus=num_cpus, n_tasks=workers)
            )
            reset_scheduling_envelope(token)
            token = set_scheduling_envelope(envelope)
            trace.step(
                "accelerator_fill" if by_device else "cluster_fill",
                workers,
                (
                    f"one worker per {num_gpus:g}-device slice, {num_cpus:g} cores each"
                    if by_device
                    else f"one worker per {num_cpus:g}-core node slice"
                ),
            )
        elif num_workers is not None:
            trace.step("explicit", workers, "num_workers was passed by the caller")
        else:
            trace.step("single_node", workers, "one node, or topology unreadable")
        # Ray is up, so the live topology is readable: give each worker an EVEN SHARE of
        # the cluster's CPUs (capped at one node's cores), not the single core Carbonite's
        # per-operator `num_cpus` models — a `_FlightWorker` runs the multi-core executor
        # over a whole partition and a managed cgroup would pin it to 1 core, throttling
        # the scan ~Ncores×. MUST run after `_ensure_ray` (`ray.nodes()` is empty before).
        # Not when the fan-out was tiled by devices: that grant is already "an accelerator
        # node's cores divided among its accelerators", and raising it makes the workers
        # unplaceable. `_even_cpu_share` averages over every worker-eligible node, so on a
        # fleet with CPU-only nodes beside the accelerator ones it would hand each device
        # worker a share no GPU node can host — and the clamp below would then collapse the
        # fan-out to one worker per node, which is the behavior the device tiling exists to
        # replace.
        share = 0.0 if by_device else _even_cpu_share(workers)
        # The raise must not hand back the core the fill deliberately kept free.
        # `_cluster_fill_workers` thins its grant so the fleet leaves a schedulable core on
        # every node (`_headroom_grant`), while `_even_cpu_share` caps at `min(node cores)` —
        # the whole node on a homogeneous cluster — so an unconditional raise puts that core
        # straight back into the placement group and restores the deadlock the thinning exists
        # to prevent. Thinning the *share* rather than capping it at the fill's grant keeps the
        # raise doing its real job: when `_placeable_grant` has over-thinned against a busy
        # cluster, the share still pulls the grant back up, just never past the headroom.
        if fill is not None and not by_device and share > 0:
            share = _headroom_grant(share, _worker_node_cpus())
        if share > num_cpus:
            envelope = (
                dataclasses.replace(envelope, num_cpus=share)
                if envelope is not None
                else SchedulingEnvelope(num_cpus=share, n_tasks=workers)
            )
            num_cpus = share
            reset_scheduling_envelope(token)
            token = set_scheduling_envelope(envelope)
            trace.step("even_cpu_share", workers, f"raised the per-worker grant to {share:g} cores")
        # Clamp against everything the fleet's bundle reserves, not cores alone: the
        # per-worker RAM grant, and the node-class restriction when this (relational) fleet
        # is held off accelerator nodes. A clamp blind to either lets the fan-out exceed
        # what any arrangement of nodes can host, and a gang-scheduled placement group that
        # cannot be satisfied hangs rather than fails.
        #
        # `cpu_only` is read from `node_class_selector` rather than from the envelope's
        # *preference*, because the preference is not always honored: the restriction also
        # needs the config gate and enough CPU-only capacity. Asking the same function the
        # bundle asks is what keeps the clamp and the placement agreeing by construction,
        # instead of two rules that can disagree about which nodes are eligible.
        restricted = bool(
            envelope is not None
            and node_class_selector(envelope.prefer_cpu_only_nodes, workers, num_cpus)
        )
        clamped = clamp_workers(
            workers,
            num_cpus,
            num_gpus,
            memory_bytes=int(envelope.memory_bytes) if envelope is not None else 0,
            cpu_only=restricted,
        )
        # Carbonite sized the per-task memory hint against its *desired* fan-out;
        # once the cluster clamp reduces (or the data-driven want exceeds) it, each
        # real task holds a larger share. Rescale the soft memory hint to the actual
        # worker count and re-install the grant so `.options(memory=)` is honest.
        trace.step(
            "clamp",
            clamped,
            "bounded by schedulable capacity"
            + (", CPU-only nodes" if restricted else "")
            + (
                f", a {envelope.memory_bytes / 1e9:.1f} GB per-worker memory grant"
                if envelope is not None and envelope.memory_bytes
                else ""
            ),
        )
        trace.report()
        if envelope is not None and clamped != workers:
            envelope = _rescale_envelope(envelope, workers, clamped)
            reset_scheduling_envelope(token)
            token = set_scheduling_envelope(envelope)
            _ensure_ray(clamped)
        workers = clamped
        # Size the per-worker memory budget from the WORKER node's RAM (hardware metadata →
        # decision): a large driver (e.g. a 197 GiB head) must not hand a 34 GiB worker a
        # budget derived from the driver's memory. Under SPREAD a worker owns its node, so its
        # spill threshold is the node's RAM × soft_limit split across the workers packed on it
        # — enough to use the machine before spilling, bounded so it never OOMs the smallest
        # node. A tighter Carbonite estimate still wins; the topology having no memory info
        # leaves the grant untouched (today's behavior).
        sized = _size_worker_memory(envelope, workers, num_cpus)
        if sized is not envelope:
            envelope = sized
            reset_scheduling_envelope(token)
            token = set_scheduling_envelope(envelope)
        # The worker count and cluster size are now settled (autoscale-wait + clamp done),
        # so snapshot the topology once for the whole placement/transport/shuffle phase
        # instead of paying a fresh `ray.nodes()` RPC at each of its ~5 read sites — the
        # O(nodes)-per-read amplification that bites at thousands of nodes.
        with topology_scope():
            transport = resolve_transport(transport, workers)
            return _dispatch(
                plan,
                sources,
                workers,
                transport,
                hub,
                materialize=materialize,
                metrics_out=metrics_out,
            )
    finally:
        release_autoscale()  # let the autoscaler reclaim what this query scaled up
        reset_scheduling_envelope(token)


def _worker_node_cpus() -> list[float]:
    """CPU counts of the nodes eligible to run distributed workers.

    Delegates to `scaling.node_classes`, which is the one place that decides what
    "worker-eligible" means: the Ray head excluded (it runs the GCS / dashboard / job
    supervisor, and scheduling data operators there causes contention), anything Ray has
    marked for drain excluded, and neither exclusion allowed to empty the list.

    This used to re-derive that from `ray.nodes()` itself, and the copy had drifted in two
    ways that matter. It counted **draining** nodes, so the primary fan-out chooser
    (`_cluster_fill_workers`) sized the fleet onto capacity being reclaimed while
    `clamp_workers` — reading the same cluster through `scaling` — excluded it, and the two
    answers to "how many workers fit" disagreed. And it read the cluster directly, so it
    missed the `topology_scope()` snapshot and paid its own `ray.nodes()` round trip on
    every call. Sharing the definition is also what the layering asks for: two copies of a
    rule is the one way to get them out of step.

    Nameplate capacity, deliberately: this is the cluster's *shape*, and a node whose cores
    are momentarily all held is still a node the fleet will run on. Sizing the shape from what
    is free at this instant makes a busy 4-node cluster look like a 1-node one and collapses
    the fan-out. What free capacity legitimately bounds is the per-worker *grant*, which
    `_fill_grant` caps so the bundle is placeable.
    """
    from batcher.dist.executors.ray_runtime.scaling import node_classes

    return [c for node in node_classes() if (c := float(node["cpus"])) > 0]


def _accelerator_fill_workers(num_gpus: float) -> tuple[int, float] | None:
    """The device-filling fan-out for a stage that needs `num_gpus` accelerators per worker.

    `_cluster_fill_workers` tiles the fleet by *cores*, and its reasoning — "more workers than
    nodes cannot add CPU parallelism, since cores are the limit" — is exactly right for a
    relational query and exactly wrong for a stage holding a device. When each worker needs an
    accelerator, the limit is devices, and a core-shaped fill caps the fan-out at the node
    count however many devices a node holds. On the common four-devices-per-node shape that
    stranded three quarters of the fleet: a 16-GPU cluster ran its inference stage on 4.

    So tile by devices instead: each node hosts ``floor(node_devices / num_gpus)`` workers, and
    the per-worker core grant is the smallest such node's cores divided by the workers it will
    host, which keeps one worker placeable on every accelerator node the way the core-shaped
    grant does. Nodes with no device host nothing — they cannot run this stage at all.

    Args:
        num_gpus: Devices one worker holds. `0` or less means this is not an accelerator
            stage and the core-shaped fill is the right one.

    Returns:
        `(workers, num_cpus)`, or `None` when this is not an accelerator stage, the fleet has
        no devices, the topology is unreadable, or the answer is a single worker — in every
        one of those the caller's existing sizing is already correct.
    """
    if num_gpus <= 0:
        return None
    try:
        from batcher.dist.executors.ray_runtime.scaling import node_classes

        nodes = [
            (float(n["cpus"]), int(float(n["gpus"]) // num_gpus))
            for n in node_classes()
            if float(n.get("gpus") or 0.0) >= num_gpus and float(n["cpus"]) > 0
        ]
        hosts = [(cores, held) for cores, held in nodes if held > 0]
        if not hosts:
            return None
        workers = sum(held for _, held in hosts)
        if workers <= 1:
            return None
        # Floored to a whole core so the grant is a number Ray can actually reserve, and at
        # least one so a device-dense node cannot ask for a fractional-core worker.
        num_cpus = max(1.0, float(int(min(cores / held for cores, held in hosts))))
        return workers, num_cpus
    except Exception as exc:
        # Silence here is expensive: the caller falls back to the data-driven sizing, which on
        # a device-dense fleet is far narrower, and "the job used a quarter of the GPUs" is
        # then indistinguishable from a deliberate decision. `FanoutTrace` records the steps
        # that ran; this is the one that did not.
        note_suppressed("dist", "read the accelerator topology for the fan-out", exc)
        return None


def _cluster_fill_workers() -> tuple[int, float] | None:
    """The cluster-filling fan-out: enough `min`-core workers to fill EVERY node's cores.

    Returns `(workers, num_cpus)` on a genuine multi-node cluster. `num_cpus` = the smallest
    worker node's cores, so a worker is placeable on any node (SPREAD-safe); `workers` =
    ``Σ floor(node_cores / num_cpus)`` over the worker nodes — one worker per `num_cpus`-core
    slice. A **homogeneous** cluster reduces to one worker per node exactly as before; a
    **heterogeneous** one gives a node with k times the smallest node's cores k workers, so its
    extra cores are used instead of stranded idle under a uniform one-worker-per-node grant
    (a 64-core node next to 32-core nodes was running at half utilization). Any worker count
    is result-correct under the mergeable algebra, so this only affects saturation; the
    per-worker CPU grant stays `min`-sized so the SPREAD placement group still fits every
    node. Returns `None` on a single node or unreadable topology, so the caller keeps the
    data-driven `_even_cpu_share` sizing. Ray must already be initialized.
    """
    try:
        node_cpus = _worker_node_cpus()
        if len(node_cpus) <= 1:
            return None
        num_cpus = _headroom_grant(_fill_grant(node_cpus), node_cpus)
        workers = sum(max(1, int(c // num_cpus)) for c in node_cpus)
        return workers, num_cpus
    except Exception as exc:
        # Same reason as the accelerator fill above: a topology read that fails quietly halves
        # a large cluster's fan-out and leaves nothing to attribute it to.
        note_suppressed("dist", "read the cluster topology for the fan-out", exc)
        return None


def _headroom_grant(grant: float, node_cpus: list[float]) -> float:
    """`grant`, thinned so the fleet cannot reserve every core in the cluster.

    The fleet is a SPREAD placement group, and **not every distributed task runs inside it**:
    `executors.map._map_udf_task` is a plain Ray task (it takes a skew-adaptive, often
    sub-core `num_cpus`), and the hardware probe is another. A grant that tiles each node
    exactly — the homogeneous case, where a 16-core node hosts one 16-core worker — leaves
    those tasks nowhere to run.

    That is a *deadlock*, not a slowdown, and it is reachable through the ordinary path: the
    session fleet is cached across queries and `api.adaptive.staging` takes a query-scoped
    lease on it before the query's first stage, so a query whose map stage follows a
    predecessor's shuffle finds 100% of the cluster reserved and waits on a fleet only it
    could release. Observed on a 16 x 16-core cluster running TPC-H sf100: q1-q15 pass, then
    q16 hangs indefinitely with `256.0/256.0 CPU (256.0 reserved in placement groups)` and one
    `_map_udf_task` pending on `{'CPU': 0.5}` — while q16 run on its own finishes in 3.2s.

    Thinning **preserves the worker count**, exactly as `_placeable_grant` does: it returns the
    largest grant no bigger than `grant` that still tiles to as many workers while leaving a
    free core on every node that hosts one. On a homogeneous 16-core fleet that is 15 — one
    worker per node still, 1/16th of the cluster kept schedulable. `grant` is returned
    unchanged when no thinner grant preserves the fan-out, since a smaller fleet is the worse
    trade and the caller's timeout still surfaces a genuinely unplaceable one.
    """
    if grant <= 1:
        return grant  # already the thinnest possible; a 1-core grant strands nothing to give

    def workers_at(g: float) -> int:
        return sum(max(1, int(c // g)) for c in node_cpus)

    def leaves_headroom(g: float) -> bool:
        # Only nodes that actually host a worker need a spare core; one too small to tile is
        # already entirely free.
        return all(c - int(c // g) * g >= 1.0 for c in node_cpus if c >= g)

    if leaves_headroom(grant):
        return grant
    wanted = workers_at(grant)
    for candidate in range(int(grant) - 1, 0, -1):
        g = float(candidate)
        if workers_at(g) >= wanted and leaves_headroom(g):
            return g
    return grant


# How much of the best-achievable core occupancy a larger grant may give up before it stops
# being worth the fatter workers. At 0.9 a grant that strands a tenth of the cluster is
# rejected, which keeps a genuinely heterogeneous fleet (32 next to 64) tiled by the smaller
# node exactly as before, while letting one undersized node be skipped rather than obeyed.
_FILL_STRAND_TOLERANCE = 0.9


def _fill_grant(node_cpus: list[float]) -> float:
    """The per-worker core grant that leaves the fewest cores stranded.

    The grant used to be `min(node_cpus)`, chosen so a worker is placeable on every node.
    That is right on a cluster whose nodes differ by a factor of two, and pathological when
    one node is much smaller than the rest: a single 2-core utility node in a fleet of
    64-core machines pinned *every* worker to 2 cores, so each one ran its scan and fold on
    a thirty-second of the node it landed on. The smallest node was setting the shape of
    the whole cluster.

    Nothing actually requires a uniform grant to fit the smallest node. A grant that a node
    cannot host simply means that node hosts no workers, which costs its cores — and losing
    one small node's cores is obviously better than crippling every large node's.

    Maximizing cores occupied is *not* the objective, and getting that wrong is instructive:
    on `[2, 64, 64, 64]` the 2-core grant occupies 194 cores against 64's 192, so "most
    cores used" picks the pathology it was meant to avoid. A small grant always wins that
    contest, because it can fill every remainder. What it buys those two extra cores with is
    97 workers instead of 3 — a shuffle with 97 streams per stage and a scan that runs
    two-cores-wide on a 64-core box.

    So the rule prefers the **largest** grant that does not strand meaningful capacity: the
    biggest candidate whose core utilization is within `_FILL_STRAND_TOLERANCE` of the best
    any candidate achieves. Fat workers unless the cluster genuinely cannot be tiled by
    them.

    Worked through: `[2, 64, 64, 64]` gives 64 (192 of 194 cores, 1% stranded, three fat
    workers). `[32, 64]` keeps 32, because a 64-core grant would strand the 32-core node
    entirely — a third of the cluster, far past the tolerance. `[16, 32, 32]` keeps 16 for
    the same reason. A homogeneous cluster has one candidate and is unchanged.
    """
    candidates = sorted({max(1.0, float(int(c))) for c in node_cpus if c > 0}, reverse=True)
    if not candidates:
        return 1.0

    def occupied(grant: float) -> float:
        return sum(int(c // grant) * grant for c in node_cpus)

    best = max(occupied(g) for g in candidates)
    chosen = candidates[-1]
    if best > 0:
        # `candidates` is descending, so the first acceptable one is the largest.
        for grant in candidates:
            if occupied(grant) >= _FILL_STRAND_TOLERANCE * best:
                chosen = grant
                break
    return _placeable_grant(chosen, node_cpus)


def _placeable_grant(grant: float, node_cpus: list[float]) -> float:
    """`grant`, thinned until the gang it implies can actually be placed on free cores.

    The grant above is chosen from nameplate capacity, which is right for the cluster's
    *shape* and wrong for whether Ray can place the fleet. A gang is all-or-nothing: it needs
    one free block of `grant` cores per worker, so a grant the free capacity cannot tile
    leaves the placement group pending until the timeout and the query fails with `no
    distributed worker became available` on a cluster that is almost entirely idle. Measured
    here with a co-tenant holding a core or two per node: `4 bundles x 8 CPU` is unsatisfiable
    while the same four workers at 5 cores each place immediately.

    Thinning **preserves the worker count** — it returns the largest grant whose free-capacity
    tiling still yields as many workers as the nameplate tiling does. That is the whole
    difference between this and reading the cluster's shape from free capacity, which was
    tried first and is much worse: a node whose cores are momentarily all held drops out
    entirely, a busy four-node cluster looks like a one-node one, and the fan-out collapses
    silently to a single worker.

    Returns `grant` unchanged when the per-node figures cannot be read, when they do not line
    up with the node set, or when nothing is holding cores — the idle-cluster case, which is
    every single-tenant run and therefore the common one.
    """
    from batcher.dist.executors.ray_runtime.scaling import node_classes

    try:
        rows = node_classes()
        free = [float(n["free_cpus"]) for n in rows]
        nameplate = [float(n["cpus"]) for n in rows]
    except Exception as exc:
        note_suppressed("dist", "read free capacity for the worker grant", exc)
        return grant
    if len(free) != len(node_cpus) or not free:
        return grant

    def tiles(sizes: list[float], g: float) -> int:
        return sum(int(c // g) for c in sizes)

    wanted = tiles(nameplate, grant)
    if wanted <= 0 or tiles(free, grant) >= wanted:
        return grant  # already placeable — the idle-cluster path, and no-op
    for candidate in range(int(grant) - 1, 0, -1):
        if tiles(free, float(candidate)) >= wanted:
            return float(candidate)
    return grant  # nothing thin enough tiles it; the cluster is genuinely full


def _even_cpu_share(workers: int) -> float:
    """CPUs to grant each distributed worker so the fan-out isn't single-core-starved.

    Two hard constraints: the grant must be **placeable on every node** (capped at
    `min(node cores)`, since workers are SPREAD across nodes) and must not over-subscribe
    (capped at `floor(total / workers)`). Within those, hand each worker as many cores as
    possible (`>= 1`) so its parallel scan-read + fold use the node, not one cgroup-pinned
    core. The grant is deliberately *uniform* — skew is handled orthogonally (LPT-balanced
    splits in `_balance`; salted hot join keys in `join_par`). Returns 1.0 (historical
    default) when topology is unavailable.
    """
    try:
        node_cpus = _worker_node_cpus()
        if not node_cpus or workers <= 0:
            return 1.0
        # The placeability cap is the grant `_fill_grant` would choose, not `min(node_cpus)`.
        # Both answer "how wide may a uniform worker be", so they have to agree: with `min`
        # here, a single undersized node re-imposed the exact pinning `_fill_grant` exists to
        # avoid — the fill path would pick a 64-core grant and this would immediately cap it
        # back to 2. It is still a *cap* (`min` against the oversubscription bound below), so
        # this only ever raises the grant to what the cluster can actually host.
        placeable = _fill_grant(node_cpus)
        non_oversubscribing = float(sum(node_cpus) // workers)  # workers x grant <= cluster
        return max(1.0, min(placeable, non_oversubscribing))
    except Exception:
        return 1.0


def _max_workers_per_node(workers: int, num_cpus: float) -> int:
    """The most workers any one node will host, given a `num_cpus`-sized uniform grant.

    Mirrors how the fan-out is actually placed: a node with `c` cores takes `floor(c / num_cpus)`
    workers, which is the same slicing `_cluster_fill_workers` uses to *choose* the fan-out. Any
    memory budget derived from this is therefore valid on the node that packs the most workers,
    rather than on an imaginary average node that may host none of them.

    Falls back to the fleet average (`ceil(workers / nodes)`) when the topology is unreadable —
    the historical behavior, and the best available guess when node sizes are unknown.
    """
    try:
        node_cpus = _worker_node_cpus()
        if node_cpus and num_cpus > 0:
            return max(1, max(int(c // num_cpus) for c in node_cpus))
    except Exception as exc:
        note_suppressed("dist", "read worker node CPU topology", exc)
    return max(1, math.ceil(workers / max(1, alive_node_count())))


def _size_worker_memory(
    envelope: SchedulingEnvelope | None, workers: int, num_cpus: float
) -> SchedulingEnvelope | None:
    """Set the per-worker memory budget from the worker node's RAM (hardware-aware spill
    threshold). Returns `envelope` unchanged (same object) when the topology advertises no
    node memory, so a cluster that doesn't report it keeps today's behavior.

    A worker owns its node under SPREAD; when several workers pack one node they split its
    RAM. The budget is `min(node_mem * soft_limit / workers_per_node, Carbonite's estimate)`
    — Carbonite's tighter data-driven estimate still wins, but an unset (unbounded) or
    driver-oversized grant is clamped to what the worker machine can actually hold.

    `workers_per_node` is the **most** any single node hosts, not the fleet average. The two
    diverge exactly on the clusters this sizing exists for: `_cluster_fill_workers` gives a node
    with k times the smallest node's cores k workers *on purpose*, so a 128-core node beside
    three 32-core ones hosts 4 workers while the average is 2. Dividing the (already smallest)
    node RAM by the average then hands each of those 4 workers twice the memory its node can
    honour — an OOM on the busiest node in the fleet, and only on heterogeneous ones.
    """
    node_mem = worker_node_memory_bytes()
    if node_mem <= 0:
        return envelope
    per_node_workers = _max_workers_per_node(workers, num_cpus)
    from batcher.config import active_config

    budget = int(node_mem * active_config().memory.soft_limit / per_node_workers)
    if budget <= 0:
        return envelope
    if envelope is None:
        return SchedulingEnvelope(num_cpus=num_cpus, n_tasks=workers, memory_bytes=budget)
    current = envelope.memory_bytes
    new_mem = budget if current <= 0 else min(current, budget)
    if new_mem == current:
        return envelope
    return dataclasses.replace(envelope, memory_bytes=new_mem)


def _rescale_envelope(
    envelope: SchedulingEnvelope, desired: int, actual: int
) -> SchedulingEnvelope:
    """Rescale a scheduling grant from its desired fan-out to the actual one.

    The per-task memory hint was `peak // desired`; with `actual` tasks each holds
    `peak // actual`, so scale the hint by `desired / actual` (and update `n_tasks`).
    """
    actual = max(1, actual)
    memory_bytes = int(envelope.memory_bytes * desired / actual) if envelope.memory_bytes else 0
    return dataclasses.replace(envelope, n_tasks=actual, memory_bytes=memory_bytes)


def _is_splittable_source(source: Source) -> bool:
    """Whether `source` yields real per-chunk splits workers can read directly (Parquet
    row-groups, lakehouse fragments) — vs an in-memory/iterator source that returns one
    `WholeSourceSplit` and would have to be shipped to the workers. Only the former is
    worth distributing for a breaker-free scan."""
    from batcher.io.splits import WholeSourceSplit

    try:
        splits = source.splits()
    except Exception:
        return False
    return bool(splits) and not (len(splits) == 1 and isinstance(splits[0], WholeSourceSplit))


# Max `LIMIT k` for the shuffle-free distributed top-N (driver merges `workers x k` rows).
_TOPN_MAX_ROWS = 1_000_000


def _collapse_limits(lim: Limit) -> tuple[int, int, LogicalPlan]:
    """Fold a chain of nested `Limit`s into one `(n, offset, base)` over the chain's base.

    Kyber's limit pushdown leaves a `Limit` above the one it pushed into the scan, so the
    dispatcher routinely sees `Limit(Limit(scan))`; treating the inner `Limit` as an opaque
    (non-map) input would strand the whole shape on `_unsupported`.

    An outer `Limit(n, offset)` keeps rows `[offset, offset + n)` of its input, and the
    inner one already restricted its child to `[i_offset, i_offset + i_n)`. Composing:
    the child rows kept are `[i_offset + offset, ...)`, and at most `min(n, i_n - offset)`
    of them survive (clamped at 0 when the outer offset skips past the inner window).
    """
    n, offset = lim.n, lim.offset
    base = lim.input
    while isinstance(base, Limit):
        n = max(0, min(n, base.n - offset))
        offset = base.offset + offset
        base = base.input
    return n, offset, base


def _join_sides_are_map_only(join) -> bool:
    """Whether both join operands are a single source with a breaker-free (map-only) plan.

    Each side is shipped to every worker and re-run against that worker's partition, so a
    breaker on a side is evaluated per-partition: `limit(5).join(dim)` kept 5 rows on each of
    4 workers and returned 20. Such joins are `requires_staging` — the inner breaker runs as
    its own distributed stage first — so refusing here turns a wrong answer into the staged
    path (or, with `adaptive=False`, a loud error).
    """
    return (
        _single_source(join.left)
        and _single_source(join.right)
        and not _has_breaker(join.left)
        and not _has_breaker(join.right)
    )


def _fusable_join_aggregate(agg: Aggregate) -> bool:
    """Whether `agg` can be distributed by reusing the join's co-partitioning — no shuffle.

    **This is exchange elimination, and it is decided by the physical-property layer.** The
    shuffle join co-partitions both sides by the join key, so its output is hash-partitioned
    by that key (`kyber.properties.hash_partitioned_on`). An aggregate whose group keys are a
    *superset* of it therefore has every group entirely inside one bucket: each reducer joins
    *and* aggregates its own bucket, the union of the reducer outputs is the complete result,
    and no second shuffle is needed. Because no cross-bucket combine is needed, even a
    non-mergeable aggregate is correct here.

    Kyber owns the question "what distribution does this relation already have"; `dist`
    schedules against the answer rather than re-deriving it, so the two cannot drift.
    """
    from batcher.kyber.properties import PhysicalProperties, hash_partitioned_on, satisfies

    j = agg.input
    if not isinstance(j, Join) or not _join_sides_are_map_only(j):
        return False
    group_cols = tuple(gk.expr.name for gk in agg.group_keys if isinstance(gk.expr, Col))
    if not group_cols:
        return False
    partitioned_on = hash_partitioned_on(j)
    if not partitioned_on:
        return False  # nothing guaranteed (an outer join, or no equi-key): must re-shuffle
    # The group keys must *cover* the partitioning, so no group straddles two buckets: the
    # delivered partitioning must be a subset of the required grouping (see `satisfies` — the
    # containment runs the opposite way from ordering). Arguments in the natural
    # (delivered, required) order.
    return satisfies(
        PhysicalProperties(hash_partitioned_on=partitioned_on),
        PhysicalProperties(hash_partitioned_on=group_cols),
    )


def _partition_aligned_aggregate(agg: Aggregate, sources, workers: int, hub) -> tuple[str, ...]:
    """Whether `agg`'s groups are already whole inside one worker's read — no shuffle at all.

    **Exchange elimination against the layout a table already has on disk**, the storage-side
    twin of `_fusable_join_aggregate`. A partitioned table — a Hive directory tree, a Delta
    table — reads splits that each hold their partition columns constant, and the scheduler
    assigns the splits sharing a value together, so equal partition-column values are already
    co-located. An aggregate whose group keys *cover* those columns therefore has every group
    entirely inside one partition: each worker folds its own partitions to final groups and
    the driver concatenates them. No map barrier, no shuffle, no reduce barrier — and, because
    nothing is combined across partitions, a non-mergeable aggregate is correct here too.

    Kyber owns "what distribution does this relation already have"
    (`kyber.properties.clustered_on`); `dist` supplies the one part of the answer only it can
    see — what the split set actually guarantees — and schedules against the result.

    Three conditions, each load-bearing:

    * every split of the read declares the same clustering columns, verified rather than
      taken on trust (`io.splits.declared_clustering`), and the splits sharing a value are
      assigned together (`io.splits.group_by_clustering`), which is what makes the guarantee
      hold for a reader that splits per data *file* rather than per partition;
    * every clustering column is a group key, so no group straddles two partitions. Getting
      this backwards reports each half of a split group as a finished group: a wrong answer,
      at cluster scale, that every single-node test passes;
    * the grouping keeps enough of the read's parallelism to be worth the exchange it saves.
      A value cannot be in two places, so a partition is one assignable unit however many
      files it holds; `scan_clustering_for` prices that against what the shuffle would have
      had, from measurements rather than a guess.

    Args:
        agg: The aggregate under consideration.
        sources: The query's bound sources.
        workers: The fleet width.
        hub: The metadata hub (only to match the executor's partition count).

    Returns:
        The clustering columns to assign by, or an empty tuple when the aggregate must
        shuffle. The columns are what the executor needs -- it assigns whole groups of them
        -- so returning them rather than a bare yes/no is what keeps the decision and the
        assignment reading the same answer.
    """
    # Bare-column keys only, and a *subset* of the keys is enough: grouping by
    # `(day, year(ts))` refines grouping by `(day)`, so every one of its groups is still
    # inside one `day` directory. A computed key simply contributes nothing to the cover.
    group_cols = tuple(gk.expr.name for gk in agg.group_keys if isinstance(gk.expr, Col))
    if not group_cols:
        return ()  # a global aggregate has one group spanning every worker: it must combine
    return _keys_cover_the_layout(agg.input, group_cols, sources, workers, hub)


def _partition_aligned_distinct(distinct: Distinct, sources, workers: int, hub) -> tuple[str, ...]:
    """Whether `distinct`'s duplicate groups are already whole inside one worker's read.

    The same elimination as `_partition_aligned_aggregate`, for the same reason: a dedup is a
    group-by that keeps one row per group, so it needs no exchange exactly when no group
    straddles a worker. A whole-row `DISTINCT` groups on every column, which contains the
    clustering columns by definition, so any clustered layout aligns it; a `DISTINCT ON`
    aligns only when its keys cover the clustering.

    A `limit` is the one shape excluded outright: keeping `n` rows per partition and
    concatenating keeps `n x partitions` rows, which is the per-partition-limit bug the
    aggregate path documents, not a shuffle-free plan.

    Args:
        distinct: The dedup under consideration.
        sources: The query's bound sources.
        workers: The fleet width.
        hub: The metadata hub (only to match the executor's partition count).

    Returns:
        The clustering columns to assign by, or an empty tuple when the dedup must shuffle.
    """
    if distinct.limit is not None:
        return ()
    keys = distinct.keys or tuple(distinct.input.available_columns())
    if not keys:
        return ()
    return _keys_cover_the_layout(distinct.input, keys, sources, workers, hub)


def _partition_aligned_window(window: Window, sources, workers: int, hub) -> tuple[str, ...]:
    """Whether every window partition is already whole inside one worker's read.

    The window form of the same elimination. A window computes each partition independently
    -- a rank, a running total, a lag -- so the only thing the shuffle establishes is that a
    partition's rows are all on one worker. A table partitioned on disk by a column the
    window partitions by has already established it, and `ROW_NUMBER() OVER (PARTITION BY
    day ...)` over a directory-per-day table is a common enough shape to be worth the check.

    The frame and the ordering need no attention: both are *within* a partition, so a worker
    holding whole partitions computes them exactly. A `rank_limit` is likewise per-partition,
    unlike the `Distinct` limit that had to be excluded.

    Args:
        window: The window under consideration.
        sources: The query's bound sources.
        workers: The fleet width.
        hub: The metadata hub (only to match the executor's partition count).

    Returns:
        The clustering columns to assign by, or an empty tuple when the window must shuffle.
    """
    keys = tuple(k.name for k in window.partition_keys if isinstance(k, Col))
    if not keys:
        return ()  # one partition over all rows: nothing to co-locate it by
    return _keys_cover_the_layout(window.input, keys, sources, workers, hub)


def _partition_local_chain(node: LogicalPlan) -> bool:
    """Whether every operator between `node` and its scan computes *within* a clustering group.

    `clustered_on` answers a different question, and the difference is the trap. It says where
    rows *are* -- which worker holds them -- and by that measure a `Limit` is transparent: it
    removes rows, and removing one never moves another. That is a true statement about
    distribution and `hash_partitioned_on` makes the identical one.

    It is not the property a shuffle-free plan needs. That plan runs the whole sub-tree
    independently on each partition and concatenates, so it also needs every operator in the
    chain to *mean the same thing* applied per partition. `Limit` does not:
    `limit(100).group_by(k)` run per partition keeps a hundred rows on each of them, which is
    the per-partition-limit defect the shuffle path's own guard documents.

    `Distinct` does, which is what this predicate is for. A dedup only ever collapses rows that
    agree, and rows that agree on the clustering columns are on one worker, so a per-partition
    dedup of a clustered relation is already global. That makes `COUNT(DISTINCT x) GROUP BY
    day` over a day-partitioned table shuffle-free -- and it is a common enough query to be
    worth the distinction. A `Distinct` carrying a limit is a limit, and is refused as one.

    Args:
        node: The root of the chain, exclusive of the operator being scheduled.

    Returns:
        True when running the chain per partition computes what running it once would.
    """
    if isinstance(node, Scan):
        return True
    if isinstance(node, Distinct):
        return node.limit is None and _partition_local_chain(node.input)
    if _is_row_wise(node):
        return _partition_local_chain(node.input)
    return False


def _note_exchange_eliminated(operator: str, columns: tuple[str, ...]) -> None:
    """Report that an operator ran with no exchange because the layout already partitioned it.

    Published as a `Decision`, so it lands in `explain(analyze=True)` and the live job view
    beside Kyber's and Carbonite's, rather than only in a log nobody enabled. Without it the
    optimization is **unobservable from outside**: the shuffle path returns exactly the same
    rows, so the only visible difference is a wall-clock number, and a user has no way to tell
    whether their table's layout is being used or silently ignored.

    Never raises: this describes work already decided on.

    The task count the read runs at is already carried by the ordinary progress events, so
    what is added here is only the part nothing else can say: *why* there is no shuffle stage.

    Args:
        operator: The operator that skipped its exchange (`aggregate`, `distinct`, `window`).
        columns: The clustering columns the layout supplied.
    """
    try:
        from batcher._internal import events
        from batcher.plan.profile import Decision

        cols = ", ".join(columns)
        events.publish(
            events.DECISION,
            **Decision(
                subsystem="core",
                category="exchange",
                summary=f"{operator} needs no shuffle: the table is already partitioned by {cols}",
                detail={"operator": operator, "clustered_on": list(columns)},
            ).to_dict(),
        )
    except Exception as exc:  # pragma: no cover - observation must never fail a query
        note_suppressed("dist", "report the eliminated exchange", exc)


def _keys_cover_the_layout(
    node: LogicalPlan, keys: tuple[str, ...], sources, workers: int, hub
) -> tuple[str, ...]:
    """Whether grouping `node` by `keys` needs no exchange, given the layout it reads.

    The shared core of the two callers above: ask `dist` what the read's split set actually
    guarantees, ask Kyber to propagate that up to `node`, and check the containment.

    Args:
        node: The relation being grouped.
        keys: The grouping (or dedup) key columns, as bare column names.
        sources: The query's bound sources.
        workers: The fleet width.
        hub: The metadata hub (only to match the executor's partition count).

    Returns:
        The clustering columns to assign by, or an empty tuple when a group could straddle two
        partitions, or when the chain below `node` would not mean the same thing run per
        partition (`_partition_local_chain`).
    """
    from batcher.dist.executors.map import scan_clustering_for
    from batcher.kyber.properties import PhysicalProperties, clustered_on, satisfies

    ids = scanned_source_ids(node)
    if len(ids) != 1 or not _partition_local_chain(node):
        return ()
    sid = next(iter(ids))
    cols = scan_clustering_for(node, sources, workers, hub)
    if not cols:
        return ()
    # `cols` names the columns at the SCAN; `clustered_on` renames them through the
    # projections between the scan and `node`, and that renamed form is what the keys must
    # cover. The executor assigns by the scan-level names, since that is what the splits
    # declare, so the two are returned and consumed separately on purpose.
    ok = satisfies(
        PhysicalProperties(clustered_on=clustered_on(node, {sid: cols})),
        PhysicalProperties(hash_partitioned_on=tuple(keys)),
    )
    return cols if ok else ()


def _aggregate_over_join(agg: Aggregate) -> bool:
    """Whether `agg` is an aggregate over a join of two single sources (any join type or
    group keys) — the general case `_fusable_join_aggregate` does not cover.

    Distributed as partial-aggregate-per-reducer + a driver-side `combine_finalize` of the
    small partials (the mergeable two-phase): the join is aggregated *on the workers* and
    only group-cardinality-many partial rows reach the driver, instead of collecting the
    whole join to the head to aggregate it single-node (the 70s→~1s join fix).
    """
    j = agg.input
    return isinstance(j, Join) and _join_sides_are_map_only(j)


# A whole-relation window aggregate is the same computation as the equivalent GROUP BY
# aggregate over zero keys; only the SQL spelling of `avg` differs from the engine's `mean`.
_WINDOW_AGG_TO_AGG = {"sum": "sum", "avg": "mean", "min": "min", "max": "max", "count": "count"}


def _is_empty_relation(plan: LogicalPlan) -> bool:
    """Whether `plan` provably yields zero rows, so there is nothing to distribute.

    Kyber folds a predicate it can prove false against the source's statistics into
    `Limit(0)`. Every row-preserving or row-reducing operator above it still yields zero
    rows. The walk stops at an `Aggregate` (a zero-key aggregate over an empty relation
    returns exactly ONE row), a `Join` (an outer join emits the surviving side), and a
    `Union` (another branch may have rows) — none of those is empty just because an input is.
    """
    from batcher.plan.logical import Filter, Project, Unnest

    node = plan
    while True:
        if isinstance(node, Limit) and node.n == 0:
            return True
        if isinstance(node, (Filter, Project, Sort, Distinct, Limit, Window, Unnest)):
            node = node.input
            continue
        return False


def _is_broadcastable_global_window(window: Window) -> bool:
    """Whether `window` is `<agg>(x) OVER ()` — one scalar per function, over every row.

    With no PARTITION BY *and* no ORDER BY, an aggregate window's frame is the whole
    relation, so every row receives the same value. That is exactly a zero-key aggregate,
    which is mergeable — so it distributes, unlike an ordered global window (a running sum,
    `row_number()`, `lag`), which needs one global row order and has no distributed path.
    An explicit frame re-introduces per-row bounds, so it is excluded too.
    """
    return (
        not window.partition_keys
        and not window.order_keys
        and window.rank_limit is None
        and all(
            f.func in _WINDOW_AGG_TO_AGG and f.frame is None and f.input is not None
            for f in window.functions
        )
    )


def _distributed_global_window(
    above: list[LogicalPlan], window: Window, sources: list[Source], workers: int, transport: str
) -> pa.Table:
    """`<agg>(x) OVER ()` — aggregate the whole relation, then broadcast the scalars.

    Two distributed passes, both linear: a zero-key mergeable aggregate reduces the relation
    to one row on the driver (O(1) memory, not O(rows)), then a stateless map appends each
    result as a literal column. Collecting every row onto one node to compute the window —
    what a naive "one partition" implementation does — is precisely the cliff this avoids.
    """
    from batcher.plan.expr_ir import AggExpr, Lit
    from batcher.plan.logical import Aggregate, AggregateSpec, Project, Projection

    totals = _dispatch(
        Aggregate(
            input=window.input,
            group_keys=(),
            aggregates=tuple(
                AggregateSpec(alias=f.alias, agg=AggExpr(_WINDOW_AGG_TO_AGG[f.func], f.input))
                for f in window.functions
            ),
        ),
        sources,
        workers,
        transport,
    )
    if totals.num_rows == 0:
        # No input rows ⇒ no output rows; the window preserves its input's cardinality.
        result = empty_result_table(window, window.available_columns())
        return result if not above else _apply_above(above, result)

    scalars = totals.to_pydict()
    columns = window.input.available_columns()
    broadcast = Project(
        input=window.input,
        items=(
            *(Projection(alias=c, expr=Col(c)) for c in columns),
            *(Projection(alias=f.alias, expr=Lit(scalars[f.alias][0])) for f in window.functions),
        ),
    )
    result = _dispatch(broadcast, sources, workers, transport)
    return result if not above else _apply_above(above, result)


def _range_partitionable_sort_key(sort: Sort) -> bool:
    """Whether the leading sort key is a type the range partitioner accepts.

    The distributed sort routes rows against sampled quantile boundaries, in one of two
    orders. A numeric key is compared as `f64` (`bc_runtime::shuffle::range_partition_by_key`),
    and a temporal one through its order-preserving integer backing. A **string** key is
    compared byte-lexically (`range_partition_by_str_key`) — never as `f64`, because arrow
    would read `"12"` as `12.0` and order the buckets numerically, disagreeing with the
    single-node lexical sort.

    The string case used to be refused here, which was the honest answer while it had no
    routing: the alternative was a `RuntimeError` from inside a Ray task. But refusing is
    only harmless when the fallback can *run*. Once an earlier stage leaves its result on
    the workers, every source is splittable and `_unsupported` raises rather than falling
    back — so `ORDER BY <string column>` after any shuffle failed outright. That is four of
    the 22 TPC-H queries (q4, q9, q12, q22), each ending in a string `ORDER BY` over a
    materialized aggregate.

    `None` from either the schema or the inference means "not certain", and the sound answer
    there is to leave routing exactly as it was — this may only ever *withhold* distribution
    on a key it is sure about.
    """
    schema = sort.input.available_schema()
    if schema is None:
        return True
    from batcher.plan.types import infer_type

    dtype = infer_type(sort.keys[0].expr, schema)
    if dtype is None:
        return True
    return (
        pa.types.is_integer(dtype)
        or pa.types.is_floating(dtype)
        or pa.types.is_decimal(dtype)
        or pa.types.is_temporal(dtype)
        or pa.types.is_string(dtype)
        or pa.types.is_large_string(dtype)
    )


def _hoist_computed_sort_key(sort: Sort):
    """Rewrite `ORDER BY <expr>, …` so the LEADING key is a plain column.

    Returns `(sort', drop_key)` — a sort over a `Project` that materializes the computed
    leading key as a hidden column, plus the `Project` that drops it again — or `None` when
    the leading key is already a column (the common case, left byte-identical).

    The distributed sort range-partitions on the leading key's *values*, which it can only
    read from a column, so `df.sort(col("a") + col("b"))` had no distributed path at all.
    Only the leading key is hoisted: the rest are evaluated by each reducer's local sort,
    which needs no column. `hoist_computed_keys` owns the materialization itself, shared
    with the window's partition keys.
    """
    from batcher.plan.logical import SortKeySpec, hoist_computed_keys, project_columns

    key = sort.keys[0]
    hoisted = hoist_computed_keys(sort.input, [key.expr], prefix="__sort_key")
    if hoisted is None:
        return None
    with_key, (hidden,) = hoisted

    columns = sort.input.available_columns()
    rewritten = dataclasses.replace(
        sort,
        input=with_key,
        keys=(
            SortKeySpec(hidden, descending=key.descending, nulls_first=key.nulls_first),
            *sort.keys[1:],
        ),
    )
    return rewritten, project_columns(rewritten, columns)


def _hoist_computed_window_keys(window: Window):
    """Rewrite `PARTITION BY <expr>, …` so every partition key is a plain column.

    Returns `(window', drop_keys)` — the window over a `Project` that materializes each
    computed partition key as a hidden column, plus the `Project` that drops those columns
    again — or `None` when every partition key is already a column.

    The distributed window hash-shuffles rows by the partition keys' column *positions*
    (`executors/window.py` resolves each key with `cols.index(k.name)`), so a computed key
    such as `partition_by=[col("v") % 4]` could not be shuffled on and the whole query had
    no distributed path. This is the window's half of the same rewrite the sort already
    used, sharing `hoist_computed_keys` rather than restating it.

    The dropped set is the window's ORIGINAL output — its input columns plus the function
    aliases — so the hidden keys vanish and nothing else does.
    """
    from batcher.plan.logical import hoist_computed_keys, project_columns

    hoisted = hoist_computed_keys(window.input, window.partition_keys, prefix="__win_key")
    if hoisted is None:
        return None
    with_keys, keys = hoisted

    output = window.available_columns()
    rewritten = dataclasses.replace(window, input=with_keys, partition_keys=keys)
    return rewritten, project_columns(rewritten, output)


def _staged_aggregate_over_join(
    above: list[LogicalPlan],
    agg: Aggregate,
    sources: list[Source],
    workers: int,
    hub=None,
    metrics_out=None,
) -> pa.Table:
    """Disk-transport aggregate over an arbitrary join: shuffle the join, then the aggregate.

    The Flight path folds the partial aggregate into the join's reducers, so only partials
    cross the network — strictly better, and it stays the flight branch. The disk shuffle has
    no such fold, and collecting the whole join to the driver to aggregate it single-node is
    the exact cliff `_unsupported` exists to prevent. So run two distributed shuffles: the
    join keeps its result partitioned on disk (`materialize=False`), and the aggregate then
    treats that intermediate as an ordinary splittable source. Driver memory stays O(groups),
    and the result is the single-node one (both stages are mergeable).
    """
    from batcher.dist.executors.aggregate import _distributed_aggregate
    from batcher.dist.executors.join import _distributed_join
    from batcher.plan.logical import Scan
    from batcher.plan.schema import SchemaRef

    joined = _distributed_join([], agg.input, sources, workers, materialize=False, hub=hub)
    if isinstance(joined, pa.Table):
        # The join shape had no partitioned form and already collected; aggregate it here
        # rather than re-shuffling a table the driver is holding anyway.
        from batcher.io.source import InMemorySource

        intermediate: Source = InMemorySource(joined.to_batches())
        cleanup = None
    else:
        intermediate, cleanup = joined, joined.cleanup

    sid = len(sources)
    staged = dataclasses.replace(
        agg, input=Scan(source_id=sid, schema=SchemaRef(intermediate.schema()))
    )
    try:
        return _distributed_aggregate(
            above, staged, [*sources, intermediate], workers, hub, metrics_out=metrics_out
        )
    finally:
        if cleanup is not None:
            cleanup()


# Deduped rows below which aggregating them centrally beats a second distributed stage.
# The stage costs a map barrier, a shuffle and a reduce barrier — call it a second of fixed
# overhead — while the single-node engine folds a simple aggregate at roughly 10M rows/s, so
# a million rows is about where the two meet. Chosen from that arithmetic rather than
# measured on a cluster; it is a routing threshold, so being off by a factor costs time on
# one query shape and never an answer.
_STAGED_DISTINCT_ROWS = 1_000_000


def _too_small_to_restage(intermediate) -> bool:
    """Whether a partitioned intermediate is small enough to aggregate on this node.

    Reads the source's own exact `row_count` — a materialized shuffle output knows how many
    rows it wrote, so this is a lookup rather than an estimate. An intermediate that cannot
    say (or says nothing) is treated as large, because the cost of restaging something small
    is one wasted stage while the cost of centralizing something large is the serial fraction
    this function exists to avoid.
    """
    count = getattr(intermediate, "row_count", None)
    rows = count() if callable(count) else None
    return rows is not None and rows < _STAGED_DISTINCT_ROWS


def _staged_aggregate_over_distinct(
    above: list[LogicalPlan],
    agg: Aggregate,
    distinct: Distinct,
    sources: list[Source],
    workers: int,
    transport: str,
    hub=None,
    metrics_out=None,
) -> pa.Table:
    """Dedup distributed, then aggregate the deduped rows **distributed as well**.

    `COUNT(DISTINCT x)` lowers to an aggregate over a `Distinct`, and the dedup has to run
    first and globally — running the aggregate map-local would count a value once per
    partition it appears in. Doing the dedup distributed and then the *aggregate* on the
    driver is correct and was what happened, but it puts every distinct value through one
    node: the driver's work is Θ(distinct cardinality), independent of the cluster, so a
    15M-key `COUNT(DISTINCT)` gets no faster however many workers it is given. That is the
    definition of a serial fraction, and on this shape it is the whole query.

    Both halves are mergeable, so there is no reason for either to be central. The dedup
    keeps its result partitioned on disk (`materialize=False`) and the aggregate then treats
    that intermediate as an ordinary splittable source — the same two-stage shape
    `_staged_aggregate_over_join` uses, for the same reason. Driver memory becomes O(groups)
    and the count is summed across reducers.

    **A second stage is only worth its own overhead when there is something to distribute**,
    and the dedup has already answered that exactly: a partitioned intermediate carries a
    measured `row_count`. Below `_STAGED_DISTINCT_ROWS` the aggregate runs locally over it
    instead — a `COUNT(DISTINCT status)` over six values would otherwise pay a full second
    map/shuffle/reduce to count six rows, which is the shape most `COUNT(DISTINCT)` queries
    actually are. The staging is for the cardinality that motivated it, not for every query
    that spells the operator.

    Falls back to aggregating here for the same reason when the dedup had no partitioned
    form (an in-memory source, or a transport that materialized anyway): the driver is
    already holding those rows, so shuffling them back out to fold them is pure cost.
    """
    from batcher.dist.executors.aggregate import _distributed_aggregate
    from batcher.dist.executors.distinct import _distributed_distinct
    from batcher.plan.logical import Scan
    from batcher.plan.schema import SchemaRef

    # `hub=None`: this stage runs the dedup **alone**, which is a fragment of the query the
    # user asked for, and its measurements must not be learned as facts about the relation.
    # A keyed dedup pre-reduces on the map side, so its reducer sees one row per key and
    # emits one row per key — recorded, that reads as a `Distinct` which removes nothing,
    # and Kyber then drops the `Distinct` from the original query. Measured:
    # `distinct(subset=["k"]).agg(count())` returned 900, then 900 distributed, then
    # **40,000** single-node in the same process. The same `hub=None` guard the distributed
    # `LIMIT` and `sample` paths carry, for the same reason: a truncated plan's row count is
    # not the source's.
    deduped = _distributed_distinct(
        [], distinct, sources, workers, transport, materialize=False, hub=None
    )
    if isinstance(deduped, pa.Table):
        # No partitioned form to restage — the driver is already holding the rows, so
        # shuffling them back out to aggregate them would be pure cost.
        return _apply_above([*above, agg], deduped)

    intermediate: Source = deduped
    cleanup = deduped.cleanup
    sid = len(sources)
    staged = dataclasses.replace(
        agg, input=Scan(source_id=sid, schema=SchemaRef(intermediate.schema()))
    )
    staged_sources = [*sources, intermediate]
    try:
        if _too_small_to_restage(intermediate):
            # Read the intermediate's rows and aggregate them as an in-memory relation, the
            # way this path did before it was staged at all. Optimizing a *fresh plan over
            # the intermediate* instead looks equivalent and is not: the deduped rows have
            # one row per key by construction, so the column statistics learned from that
            # scan say the key is unique — and Kyber then drops the `Distinct` from the
            # ORIGINAL query on a later run. Measured: `distinct(subset=["k"]).agg(count())`
            # returned 900, then 900 distributed, then **40,000** single-node in the same
            # process. Core measures and Kyber decides; what must not happen is a fragment's
            # measurements being attributed to the relation it was derived from.
            from batcher.io.source import read_source

            batches = read_source(intermediate)
            table = pa.Table.from_batches(batches) if batches else _empty_agg_table(agg)
            return _apply_above([*above, agg], table)
        # Second stage on the transport the first ran on. The disk shuffle needs cluster-wide
        # scratch; routing a Flight query's second stage through it would demand a shared
        # mount the Flight path deliberately does not require.
        if transport == "flight":
            from batcher.dist.flight_aggregate import execute_aggregate_flight

            return execute_aggregate_flight(
                above, staged, staged_sources, workers, hub=hub, metrics_out=metrics_out
            )
        return _distributed_aggregate(
            above, staged, staged_sources, workers, hub, metrics_out=metrics_out
        )
    finally:
        cleanup()


def _dispatch(
    plan: LogicalPlan,
    sources: list[Source],
    workers: int,
    transport: str,
    hub=None,
    *,
    materialize: bool = True,
    metrics_out=None,
):
    # A provably-empty relation (Kyber folded a false predicate to `Limit(0)`) has no data
    # to distribute, so run it on one node: correct, instant, and not the perf cliff
    # `_unsupported` guards against. Without this, `filter(<provably false>)` under an
    # operator that is not a `_split_at` pass-through (a window) reached `_unsupported` and
    # raised, purely because the folded `Limit` reads as a pipeline breaker.
    if _is_empty_relation(plan):
        return _single_node(plan, sources)

    # Batch-inference / embedding pipelines (map_batches): distribute the linear
    # map chain across workers — the Ray Data competitor path.
    from batcher.core.udf import has_map_batches

    if has_map_batches(plan):
        if _is_linear_map_pipeline(plan) and _single_source(plan):
            # A linear chain with a stateless-CPU prefix feeding a GPU/load-once stage
            # streams the two with the stages overlapped (CPU prepares k+1 while the GPU
            # runs k), when enabled; otherwise it runs embarrassingly parallel. Any such
            # chain qualifies (incl. CPU→GPU→postprocess); shapes with no overlap to win
            # fall back to the non-overlapped map.
            from batcher.config import active_config

            if active_config().distributed.stream_inference:
                from batcher.dist.executors.plan_analysis import split_into_resource_stages

                if split_into_resource_stages(plan) is not None:
                    from batcher.dist.streaming import stream_distributed_pipeline

                    return stream_distributed_pipeline(plan, sources, workers, hub)
            from batcher.dist.executors.map import _distributed_map

            return _distributed_map(plan, sources, workers, hub)
        # An aggregate over a linear map/UDF pipeline: distribute the UDF across workers,
        # partial-aggregate on each, combine on the driver (the Ray Data map_batches→agg
        # shape). The UDF runs cluster-wide instead of single-node on the driver.
        agg_split = _split_at(plan, Aggregate)
        if agg_split is not None:
            above, agg = agg_split
            sub = agg.input
            if has_map_batches(sub) and _is_linear_map_pipeline(sub) and _single_source(sub):
                from batcher.dist.executors.map import _distributed_map_aggregate

                return _distributed_map_aggregate(above, agg, sources, workers)
        # Any other breaker over a map/UDF pipeline — `map_batches(...).sort(...)`,
        # `.distinct()`, `.limit()`, a partitioned window: run the pipeline as its own
        # distributed stage, land it on shared scratch, and dispatch the breaker over that.
        staged = _stage_map_prefix(plan, sources, workers, transport, hub)
        if staged is not None:
            return staged
        # Any other map+breaker shape has no distributed path yet.
        return _unsupported(plan, sources, "a map_batches/UDF pipeline feeding this operator")

    # Breaker-free scan/filter/project over a SPLITTABLE source (Parquet row-groups,
    # lakehouse fragments): fan the read out so each worker reads its own splits in
    # parallel — the distributed-scan case — instead of one node reading the whole
    # source. In-memory/iterator sources stay single-node (shipping them to workers
    # costs more than the parallel CPU saves). Reuses `_distributed_map`'s stateless
    # task path (no UDF/GPU ⇒ one task per partition).
    if _is_linear_map_pipeline(plan) and _single_source(plan):
        sid = next(iter(scanned_source_ids(plan)))
        if sid < len(sources) and _is_splittable_source(sources[sid]):
            from batcher.dist.executors.map import _distributed_map

            return _distributed_map(plan, sources, workers, hub)

    # A bare `LIMIT n OFFSET k` over a breaker-free single source (`df.limit(10)`,
    # `df.head()`, `df.filter(...).limit(10)` — the most common interactive shape, and
    # until now a hard failure on distributed data). Each worker keeps only the first
    # `k + n` rows of its OWN partition and the driver re-slices their concatenation.
    #
    # This is exact, not a sample: the global first `k + n` rows are a prefix of the
    # source, so every one of them lies in some partition's own first `k + n` rows; and
    # `_distributed_map` (with `preserve_order`) assembles partition results **by index**
    # over contiguous, source-ordered split runs, so their concatenation is the source's
    # row order and `slice(k, n)` selects precisely the rows the single-node engine returns.
    # `preserve_order` is required: the default load-balanced (`_balance`) split assignment
    # puts non-adjacent splits in one partition, so a partition's own first `k + n` rows
    # interleave rows from different parts of the source and the slice returns a different
    # row set than single-node. Only `workers x (k + n)` rows ever reach the driver, never
    # the whole source.
    #
    # `hub=None`: the per-worker plan is truncated, so its row count must not be learned
    # as the source's cardinality. A `map_batches` prefix returned above, so the pipeline
    # here is pure scan/filter/project/unnest.
    limit_split = _split_at(plan, Limit)
    if limit_split is not None:
        above, lim = limit_split
        n, offset, base = _collapse_limits(lim)
        if _single_source(base) and _is_linear_map_pipeline(base):
            sid = next(iter(scanned_source_ids(base)))
            if sid < len(sources) and _is_splittable_source(sources[sid]):
                from batcher.dist.executors.map import _distributed_map

                per_worker = Limit(input=base, n=offset + n, offset=0)
                table = _distributed_map(per_worker, sources, workers, None, preserve_order=True)
                table = table.slice(offset, n)
                return table if not above else _apply_above(above, table)

    # `with_row_index` — a single global counter, so a per-partition run would restart it at
    # zero on every worker. Instead run the (row-wise) input distributed and number the rows
    # on the driver: `_distributed_map` (with `preserve_order`) assembles partitions BY INDEX
    # over contiguous, source-ordered split runs, so their concatenation is the source's own
    # row order and the counter lands on exactly the rows single-node numbers. Without
    # `preserve_order` the default `_balance` split assignment scrambles that order, so row
    # `i` of the source gets a different index than single-node. `with_random` (position-keyed
    # hash) and `tail` (row index + filter) both lower to `RowId`, so all three distribute
    # through this one path.
    rowid_split = _split_at(plan, RowId)
    if rowid_split is not None:
        above, rowid = rowid_split
        if _single_source(rowid.input) and _is_linear_map_pipeline(rowid.input):
            sid = next(iter(scanned_source_ids(rowid.input)))
            if sid < len(sources) and _is_splittable_source(sources[sid]):
                from batcher.dist.executors.map import _distributed_map

                table = _distributed_map(rowid.input, sources, workers, None, preserve_order=True)
                index = pa.array(
                    range(rowid.offset, rowid.offset + table.num_rows), type=pa.int64()
                )
                # `RowId.available_columns()` puts the index FIRST, and the engine emits it
                # non-nullable (a counter is never null). Match both exactly, or the schema
                # differs from single-node and `pa.concat_tables` rejects the pair.
                field = pa.field(rowid.alias, pa.int64(), nullable=False)
                table = table.add_column(0, field, index)
                return table if not above else _apply_above(above, table)

    # A fixed-count `sample(n=...)` keeps the `n` smallest-hash rows of the WHOLE relation,
    # so — unlike the fraction form, which is a per-row predicate and rides the map path
    # above — running it per partition keeps `n` rows from EVERY partition. It is not
    # row-wise, so until now it reached `_unsupported` and raised on distributed data.
    #
    # It is, however, mergeable top-N: a row among the globally `n` smallest hashes is also
    # among its own partition's `n` smallest (its partition holds a subset of the rows, so
    # its rank there is no worse than its global rank), so the union of the per-partition
    # results *contains* the global answer, and re-applying the same operator to that union
    # selects exactly it. `bc_interp::ops::reshape::sample_n_batches` states that contract
    # and breaks hash ties by row *content*, so no `preserve_order` is needed here (unlike
    # the `Limit` path above): the result does not depend on how the input was split.
    #
    # `hub=None`: the per-worker plan is truncated to `n` rows, so its row count must not be
    # learned as the source's cardinality.
    sample_split = _split_at(plan, Sample)
    if sample_split is not None:
        above, sample = sample_split
        if sample.n is not None and _single_source(sample.input) and not _has_breaker(sample.input):
            sid = next(iter(scanned_source_ids(sample.input)))
            if sid < len(sources) and _is_splittable_source(sources[sid]):
                from batcher.dist.executors.map import _distributed_map

                partials = _distributed_map(sample, sources, workers, None)
                # `sample` innermost: the global n-smallest of the union of the partials,
                # then whatever the user stacked above it.
                return _apply_above([*above, sample], partials)

    agg_split = _split_at(plan, Aggregate)
    if agg_split is not None:
        above, agg = agg_split
        # The table's layout already partitions it by the group keys, so every group is
        # complete inside one partition and there is nothing to exchange: run the aggregate
        # itself as the per-partition plan and concatenate.
        #
        # Checked FIRST, ahead of the DISTINCT branches below, because the shape it most often
        # rescues is one of theirs. `COUNT(DISTINCT x) GROUP BY day` lowers to an aggregate
        # over a `Distinct`, and a dedup is partition-local over a clustered relation -- rows
        # that agree are already on one worker -- so the whole thing folds with no shuffle at
        # all. Reached only when `_partition_local_chain` holds, which is what keeps a `Limit`
        # in the chain out.
        #
        # `hub=None` deliberately: what comes back is one row per group, and learning that as
        # the *source's* cardinality would teach the optimizer that a thousand-partition table
        # holds a thousand rows. The distributed `LIMIT` path above withholds it for the same
        # reason.
        if _single_source(agg.input):
            aligned = _partition_aligned_aggregate(agg, sources, workers, hub)
            if aligned:
                from batcher.dist.executors.map import _distributed_map

                _note_exchange_eliminated("aggregate", aligned)
                table = _distributed_map(agg, sources, workers, None, cluster_by=aligned)
                return table if not above else _apply_above(above, table)
        # Aggregate over a DISTINCT (the `count_distinct → distinct + count` rewrite, or a
        # user `distinct().agg(...)`) must be caught BEFORE the map/shuffle aggregate path:
        # that path runs `agg.input` as a per-partition map prefix, but a DISTINCT has GLOBAL
        # semantics — run map-local, each partition dedups independently and the reducer sums
        # the per-partition counts, double-counting any value spanning two source partitions
        # (the COUNT(DISTINCT) overcount). Distribute the DISTINCT first (globally deduped),
        # then aggregate over its result — as a SECOND distributed stage, not on the driver:
        # the deduped rows are Θ(distinct cardinality) and folding them centrally is a serial
        # term no cluster size touches, which on a 15M-key COUNT(DISTINCT) is the whole query.
        # See `_staged_aggregate_over_distinct`.
        # Scoped to a direct `Distinct` input: OTHER breakers (nested aggregate / sort) keep
        # the map-local path, which is correct for the composable aggregates that dominate.
        # Whole-row only. A *keyed* dedup staged this way returns the right answer and then
        # poisons the next one: its reducer sees one row per key and emits one row per key,
        # which the learning loop reads as a `Distinct` that removes nothing, and Kyber drops
        # the `Distinct` from the original query on a later run in the same process —
        # measured at 900, 900 distributed, then **40,000**. Passing `hub=None` does not stop
        # it (the recording is via the ambient hub, not the parameter), and the shape this
        # staging exists for is `COUNT(DISTINCT x)`, which lowers to a whole-row dedup over a
        # projected column. So a keyed dedup keeps the path it had, and this is a narrowing
        # rather than a fix — the interaction is understood in effect but not in mechanism.
        if (
            isinstance(agg.input, Distinct)
            and not agg.input.keys
            and agg.input.limit is None
            and _single_source(agg.input)
            and not _has_breaker(agg.input.input)
        ):
            return _staged_aggregate_over_distinct(
                above,
                agg,
                agg.input,
                sources,
                workers,
                transport,
                hub=hub,
                metrics_out=metrics_out,
            )
        if (
            isinstance(agg.input, Distinct)
            and _single_source(agg.input)
            and not _has_breaker(agg.input.input)
        ):
            from batcher.dist.executors.distinct import _distributed_distinct

            return _distributed_distinct(
                [*above, agg],
                agg.input,
                sources,
                workers,
                transport,
                materialize=materialize,
                hub=hub,
                metrics_out=metrics_out,
            )
        # The map/shuffle aggregate path: run `agg.input` as the per-partition map prefix, then
        # partial-aggregate + shuffle + combine. Sound ONLY over a breaker-free prefix — a map
        # prefix is evaluated independently on every partition. A `Limit` prefix would then keep
        # `n` rows *per partition* (`limit(100).group_by(k).agg(count())` counted 4x on 4
        # workers), and a nested `Aggregate` would hand this one per-partition partial groups
        # (`max(sum per k)` read a per-partition max). Both returned wrong answers silently.
        # Such shapes are `requires_staging`, so the staged executor runs the inner breaker
        # first; here we refuse rather than compute. A join input has two sources, so
        # `_single_source` skips it to the join handlers below.
        if _single_source(agg.input) and not _has_breaker(agg.input):
            if transport == "flight":
                from batcher.dist.flight_aggregate import execute_aggregate_flight

                # `materialize=False` (when an adaptive-loop fleet is ambient) keeps the
                # result on the workers as a `FlightMaterializedSource` the next stage
                # reads in place — no driver collect; else it spawns + collects as before.
                return execute_aggregate_flight(
                    above,
                    agg,
                    sources,
                    workers,
                    materialize=materialize,
                    hub=hub,
                    metrics_out=metrics_out,
                )
            from batcher.dist.executors.aggregate import _distributed_aggregate

            return _distributed_aggregate(
                above, agg, sources, workers, hub, materialize=materialize, metrics_out=metrics_out
            )
        # An aggregate over a UNION ALL of map-only branches: every branch maps into the SAME
        # shuffle, so nothing is concatenated on the driver first. `union(...).group_by(...)`
        # and both set operators (`intersect`/`except_` lower to an aggregate over a union of
        # tagged branches) reach this. Without it they took `_distributed_union`, which runs
        # each branch to a driver table — the whole of both inputs through one node to answer
        # a query that reduces them. Disk shuffle only, as the ASOF join is, so it needs the
        # same cross-node scratch the disk transport always does.
        if shuffle_branches(agg.input) is not None and isinstance(agg.input, Union):
            _require_shared_scratch("aggregate over union")
            from batcher.dist.executors.aggregate import _distributed_aggregate

            return _distributed_aggregate(
                above, agg, sources, workers, hub, materialize=materialize, metrics_out=metrics_out
            )
        # Aggregate over an inner join grouped by ⊇ the join key: fold each reducer's bucket
        # to groups (exchange elimination) — full join never collects on head.
        if _fusable_join_aggregate(agg):
            if transport == "flight":
                from batcher.dist.flight_join import execute_join_flight

                return execute_join_flight(
                    above,
                    agg.input,
                    sources,
                    workers,
                    fused_agg=agg,
                    hub=hub,
                    metrics_out=metrics_out,
                )
            from batcher.dist.executors.join import _distributed_join_aggregate

            return _distributed_join_aggregate(above, agg, agg.input, sources, workers, hub)
        # General aggregate over a (non-key-aligned, or non-inner) join.
        if _aggregate_over_join(agg):
            if transport == "flight":
                # Fold the partial aggregate into the join's reducers, so only partials
                # cross the network and the join never collects on the head.
                from batcher.dist.flight_join import execute_join_flight

                return execute_join_flight(
                    above,
                    agg.input,
                    sources,
                    workers,
                    fused_agg=agg,
                    combine_partials=True,
                    hub=hub,
                    metrics_out=metrics_out,
                )
            return _staged_aggregate_over_join(above, agg, sources, workers, hub, metrics_out)

    join_split = _split_at(plan, Join)
    if join_split is not None:
        above, join = join_split
        if _join_sides_are_map_only(join):
            if transport == "flight":
                from batcher.dist.flight_join import execute_join_flight

                # `materialize=False` (an intermediate stage of a multi-join query) keeps
                # each reducer's joined bucket on its worker and returns a
                # `FlightMaterializedSource` the next stage reads in place — no driver
                # round-trip per join, which is what makes a 3+-table query scale.
                return execute_join_flight(
                    above,
                    join,
                    sources,
                    workers,
                    materialize=materialize,
                    hub=hub,
                    metrics_out=metrics_out,
                )
            from batcher.dist.executors.join import _distributed_join

            return _distributed_join(
                above, join, sources, workers, materialize=materialize, hub=hub
            )

    # ASOF join with `by` keys: co-partition both sides by the `by` keys (equal `by`
    # values hash together, so each bucket is an independent ASOF join). A KEYLESS ASOF has
    # no group to hash, so it range-partitions both sides on `on` instead and lends each
    # bucket the one row per direction that can match across a boundary.
    asof_split = _split_at(plan, AsofJoin)
    if asof_split is not None:
        above, asof = asof_split
        if _join_sides_are_map_only(asof):
            if asof.left_by:
                return _distributed_asof(above, asof, sources, workers)
            return _distributed_asof_keyless(above, asof, sources, workers)

    # RANGE (inequality) join: broadcast the build side, split the probe side. An
    # inequality has no equality to co-partition on — a hash shuffle would put `a.x` and the
    # `b.y` values it is less than in different buckets — so the shuffle every other join
    # uses is simply not available. Replicating the build side is: each probe task sees the
    # WHOLE right, so each left row's match set is computed in its own partition.
    #
    # Gated on the probe side reading a genuinely SPLITTABLE source, which the other join
    # paths do not need: an equi-join chooses broadcast-vs-shuffle from the planner's
    # size-based `strategy`, so it already has a cheap plan for a small input. A range join
    # has only the one strategy, so the size question has to be asked here — and without it
    # a 200-row-against-50-row interval join was partitioned to disk and handed to Ray tasks,
    # which costs orders of magnitude more than running it locally. `_unsupported` states
    # the rule this follows: with every source in memory "there is no distributed data to
    # speak of, so executing it on one node is the correct plan, not a fallback."
    range_split = _split_at(plan, RangeJoin)
    if range_split is not None:
        above, rj = range_split
        from batcher.dist.executors.join import _BROADCAST_SAFE

        probe_ids = scanned_source_ids(rj.left)
        if (
            rj.join_type in _BROADCAST_SAFE
            and _join_sides_are_map_only(rj)
            and all(i < len(sources) and _is_splittable_source(sources[i]) for i in probe_ids)
        ):
            return _distributed_range_join(above, rj, sources, workers, hub)

    # A top-level sort over a scannable input distributes via range partitioning on the
    # leading key. That key must be a plain COLUMN (the range partitioner splits on its
    # values); a computed one — `ORDER BY a + b`, `ORDER BY lower(name)` — is hoisted into a
    # hidden column first. Secondary keys may be any expression: only the leading key drives
    # the partitioning, the rest are evaluated by each reducer's local sort.
    sort_split = _split_at(plan, Sort)
    if sort_split is not None:
        above, sort = sort_split
        if (
            _single_source(sort.input)
            and sort.keys
            and not _has_breaker(sort.input)
            and _range_partitionable_sort_key(sort)
        ):
            hoisted = _hoist_computed_sort_key(sort)
            if hoisted is not None:
                sort, drop_key = hoisted
                above = [*above, drop_key]  # innermost: drops the hidden key from the result
            if transport == "flight":
                from batcher.dist.flight_sort import execute_sort_flight, execute_topn_flight

                # Small `ORDER BY ... LIMIT k` → mergeable top-N (no shuffle); else full sort.
                if sort.limit is not None and sort.limit <= _TOPN_MAX_ROWS:
                    return execute_topn_flight(above, sort, sources, workers)
                # `materialize=False` leaves each range bucket on the worker that sorted
                # it and hands back a `FlightMaterializedSource` over the handles in range
                # order. A sort is row-preserving, so collecting it is the whole relation
                # through the driver — the biggest driver term left in this dispatcher.
                return execute_sort_flight(
                    above,
                    sort,
                    sources,
                    workers,
                    hub=hub,
                    metrics_out=metrics_out,
                    materialize=materialize,
                )
            from batcher.dist.executors.sort import _distributed_sort, _distributed_topn

            # Small `ORDER BY ... LIMIT k` → mergeable top-N (no shuffle), as the Flight
            # branch above already does. Without this the strategy was picked by whichever
            # transport the topology resolved to rather than by the query: the same
            # `df.sort(...).limit(10)` exchanged every row on disk and no rows on Flight.
            if sort.limit is not None and sort.limit <= _TOPN_MAX_ROWS:
                return _distributed_topn(above, sort, sources, workers)
            return _distributed_sort(
                above, sort, sources, workers, hub, metrics_out, materialize=materialize
            )

    # DISTINCT over a breaker-free single source: dedup via the aggregate shuffle.
    distinct_split = _split_at(plan, Distinct)
    if distinct_split is not None:
        above, distinct = distinct_split
        if _single_source(distinct.input) and not _has_breaker(distinct.input):
            # The table's layout already groups the duplicates: every row that could be a
            # duplicate of another is in the same directory, hence on the same worker. Dedup
            # per partition and concatenate -- see `_partition_aligned_aggregate` for why the
            # hub is withheld (the output is one row per key, not the source's row count).
            aligned = _partition_aligned_distinct(distinct, sources, workers, hub)
            if aligned:
                from batcher.dist.executors.map import _distributed_map

                _note_exchange_eliminated("distinct", aligned)

                table = _distributed_map(distinct, sources, workers, None, cluster_by=aligned)
                return table if not above else _apply_above(above, table)
            from batcher.dist.executors.distinct import _distributed_distinct

            return _distributed_distinct(
                above,
                distinct,
                sources,
                workers,
                transport,
                materialize=materialize,
                hub=hub,
                metrics_out=metrics_out,
            )

    # A partitioned window over a breaker-free source: hash-shuffle rows by the partition
    # keys so each partition is computed whole on one reducer. A computed partition key
    # (`partition_by=[col("v") % 4]`) is materialized into a hidden column first, exactly
    # as the sort hoists a computed leading key, because the shuffle reads keys by column.
    # A window with NO partition keys has nothing to *hash* on, but it still has a seam: when
    # it is order-free it is a whole-relation aggregate broadcast, and when it is ordered it
    # splits by the order instead — range-partition into ordered buckets, window each, then
    # shift each by the prior buckets' contribution (`dist/global_window/`). Both keep the
    # ordered global window off the one-node cliff it used to raise on.
    window_split = _split_at(plan, Window)
    if window_split is not None:
        above, window = window_split
        if _single_source(window.input) and not _has_breaker(window.input):
            if _is_broadcastable_global_window(window):
                return _distributed_global_window(above, window, sources, workers, transport)
            if not window.partition_keys and supports_ordered_bucket_offsets(window):
                if transport == "flight":
                    from batcher.dist.global_window import execute_global_window_flight

                    return execute_global_window_flight(above, window, sources, workers)
                from batcher.dist.global_window import execute_global_window_disk

                return execute_global_window_disk(above, window, sources, workers, hub)
            if window.partition_keys:
                # The table's layout already holds each window partition whole on one worker,
                # so there is nothing to exchange: window each partition where it was read and
                # concatenate. Checked before the computed-key hoist, which only matters for a
                # shuffle that has to read keys by column.
                aligned = _partition_aligned_window(window, sources, workers, hub)
                if aligned:
                    from batcher.dist.executors.map import _distributed_map

                    _note_exchange_eliminated("window", aligned)

                    # The hub is passed here where the aggregate and dedup paths withhold it,
                    # and the difference is the operator's row arithmetic: a window emits one
                    # row per input row, so what comes back IS the source's (post-filter) row
                    # count and is the same measurement any other map pipeline records. An
                    # aggregate returns one row per group, which learned as a source
                    # cardinality would teach the optimizer that a thousand-partition table
                    # holds a thousand rows.
                    table = _distributed_map(window, sources, workers, hub, cluster_by=aligned)
                    return table if not above else _apply_above(above, table)
                hoisted = _hoist_computed_window_keys(window)
                if hoisted is not None:
                    window, drop_keys = hoisted
                    above = [*above, drop_keys]  # innermost: drops the hidden keys
                if transport == "flight":
                    from batcher.dist.flight_window import execute_window_flight

                    # `materialize=False` leaves each windowed bucket on its worker: a
                    # window emits a row per input row, so collecting it is the whole
                    # relation through the driver.
                    return execute_window_flight(
                        above,
                        window,
                        sources,
                        workers,
                        hub=hub,
                        metrics_out=metrics_out,
                        materialize=materialize,
                    )
                from batcher.dist.executors.window import _distributed_window

                return _distributed_window(
                    above, window, sources, workers, hub, metrics_out, materialize=materialize
                )

    # UNION: distribute each branch independently, then concatenate (+ dedup).
    union_split = _split_at(plan, Union)
    if union_split is not None:
        above, union = union_split
        from batcher.dist.executors.union import _distributed_union

        return _distributed_union(above, union, sources, workers, transport)

    # No distributed path matched this shape.
    return _unsupported(plan, sources, "an unsupported operator combination")


def _stage_map_prefix(
    plan: LogicalPlan, sources: list[Source], workers: int, transport: str, hub=None
):
    """Run a `map_batches` prefix as its own stage so the breaker above it can distribute.

    Returns the result table, or `None` when the shape does not qualify (the caller then
    reports the gap as before).

    A UDF pipeline distributes, and every relational breaker distributes, but nothing composed
    them: the map executors ship a *whole* sub-plan to each worker, which is only sound for a
    map-only one, so a `sort` / `distinct` / `limit` / partitioned `window` above a
    `map_batches` matched no branch and raised on splittable data. That is the shape of most
    batch-inference work once the model has run — read, infer, then order or deduplicate the
    output — and `_distributed_map_aggregate` covered only the aggregate case of it.

    The fix is a stage boundary, which is what a shuffle is anyway. Each worker writes its own
    partition of the UDF output to shared scratch and only file locators come back, so the
    post-inference rows never pass through the driver — the same two-phase shape as the
    distributed write, and the reason this is bounded in memory rather than a driver
    materialization. The breaker then sees an ordinary splittable Parquet source and takes the
    route it always had, so it needs no map-awareness of its own.

    Order is preserved where it is observable, because the scratch is read back as a source
    and the breaker above imposes its own order; a bare `limit` over a UDF is the one shape
    that would depend on the write order, and it is handled by the ordinary `Limit` path only
    when there is no UDF.

    The walk down to the map pipeline follows `plan.visitor.children` rather than a `.input`
    attribute, so it stops at the first node with more than one operand and hands that shape
    to `_stage_map_operands` — which stages each operand that is a UDF pipeline. Before that,
    `map_batches(...).join(other)` (and a union of two inference branches) had no distributed
    path at all: the walk hit a node with no `.input` and gave up, so the most ordinary shape
    in a batch-inference job — embed a table, then join the embeddings to something — raised
    on distributed data.
    """
    from batcher.core.udf import has_map_batches
    from batcher.plan.visitor import children, with_children

    # Walk down single-operand nodes to the map pipeline.
    chain: list[LogicalPlan] = []
    node = plan
    while not _is_linear_map_pipeline(node):
        kids = children(node)
        if len(kids) != 1:
            return _stage_map_operands(plan, sources, workers, transport, hub)
        chain.append(node)
        node = kids[0]
    if not chain or not has_map_batches(node) or not _single_source(node):
        return None
    if not _stageable_pipeline(node, sources):
        return None  # nothing to fan out; the single-node fallback is the right plan
    _require_shared_scratch("a map_batches pipeline feeding this operator")

    from batcher.dist.shuffle_io import distributed_work_dir

    work_dir = distributed_work_dir("batcher_mapstage_")
    try:
        landed = _land_map_stage(node, sources, workers, hub, work_dir, len(sources))
        if landed is None:
            # Every partition was empty, so no file carries a schema. The result is empty
            # whatever the breaker is, and the plan's own schema is the honest one to return.
            return empty_result_table(plan, plan.available_columns())
        scan, staged = landed
        # Re-root the chain on a scan of the staged output. `materialize=True`: the scratch is
        # removed below, so a partitioned intermediate pointing into it must not escape.
        rebuilt: LogicalPlan = scan
        for above in reversed(chain):
            rebuilt = with_children(above, [rebuilt])
        return _dispatch(rebuilt, [*sources, staged], workers, transport, hub, materialize=True)
    finally:
        _rmtree(work_dir)


def _stageable_pipeline(node: LogicalPlan, sources: list[Source]) -> bool:
    """Whether `node` is a UDF pipeline worth landing on scratch as its own stage.

    It has to read exactly one source, and that source has to be splittable — otherwise
    there is nothing to fan out and the single-node fallback is already the right plan.
    """
    if not _single_source(node):
        return False
    sid = next(iter(scanned_source_ids(node)))
    return sid < len(sources) and _is_splittable_source(sources[sid])


def _land_map_stage(
    node: LogicalPlan,
    sources: list[Source],
    workers: int,
    hub,
    work_dir: str,
    source_id: int,
) -> tuple[LogicalPlan, Source] | None:
    """Run `node` as its own distributed stage onto `work_dir`, and read it back as a scan.

    The single half of the staging both callers share: fan the UDF pipeline across the
    cluster with a Parquet `write_spec` so no worker ships rows to the driver, then re-open
    the scratch as an ordinary splittable source the next stage scans.

    Args:
        node: The single-source UDF pipeline to run.
        sources: The ambient source list `node`'s scans index into.
        workers: The fan-out for this stage.
        hub: The metadata hub, for the map stage's measured cardinalities.
        work_dir: Cluster-shared scratch the stage's output lands in. Caller-owned.
        source_id: The index the returned source will occupy in the rebuilt source list.

    Returns:
        The `(Scan, Source)` pair reading the staged output, or `None` when every partition
        was empty and no file carries a schema.
    """
    from batcher.dist.executors.map import _distributed_map
    from batcher.io.formats.structured.parquet import ParquetSource
    from batcher.plan.schema import SchemaRef

    _distributed_map(
        node,
        sources,
        workers,
        hub,
        write_spec={
            "fmt": "parquet",
            "sink_kwargs": None,
            "path": work_dir,
            "partition_by": None,
        },
    )
    staged = ParquetSource(work_dir)
    try:
        schema = SchemaRef.from_arrow(staged.schema())
    except Exception:
        return None
    return Scan(source_id, schema), staged


def _stage_map_operands(
    plan: LogicalPlan, sources: list[Source], workers: int, transport: str, hub=None
):
    """Stage each UDF operand of a multi-operand breaker, then dispatch the breaker over them.

    The two-sided twin of `_stage_map_prefix`. `map_batches(...).join(other)` — embed a
    table, then join the embeddings to something — and a union of two inference branches
    both bottom out at a node with more than one operand, which the single-input walk cannot
    rewrite. Here each operand that contains a UDF is run as its own distributed stage onto
    its own scratch dir and replaced by a scan of the result; operands with no UDF are left
    exactly as they are, so a join of an inference branch against a plain Parquet table
    stages only the branch. The breaker itself is then dispatched normally, which is what
    gives it the shuffle, broadcast, skew handling and spill every other join gets.

    This is a rewrite, not a second join implementation: after the staging, what
    `_dispatch` sees is the ordinary shape it already routes.

    Returns the result table, or `None` when the shape does not qualify — a UDF operand
    that is not a single-source linear pipeline, or one whose staged output turned out
    empty. An empty operand is declined rather than folded to an empty result because a
    breaker is not uniformly empty-preserving: an outer join with an empty right side still
    emits every left row, so returning "empty" there would be a wrong answer rather than a
    missing route.
    """
    from batcher.core.udf import has_map_batches
    from batcher.plan.visitor import children, with_children

    chain: list[LogicalPlan] = []
    node = plan
    while True:
        kids = children(node)
        if len(kids) != 1:
            break
        chain.append(node)
        node = kids[0]
    operands = children(node)
    if len(operands) < 2:
        return None
    if not any(has_map_batches(k) for k in operands):
        return None
    for k in operands:
        if has_map_batches(k) and not (
            _is_linear_map_pipeline(k) and _stageable_pipeline(k, sources)
        ):
            return None
    _require_shared_scratch("a map_batches pipeline feeding this operator")

    from batcher.dist.shuffle_io import distributed_work_dir

    work_dirs: list[str] = []
    try:
        staged_sources = list(sources)
        rebuilt_operands: list[LogicalPlan] = []
        for k in operands:
            if not has_map_batches(k):
                rebuilt_operands.append(k)
                continue
            work_dir = distributed_work_dir("batcher_mapstage_")
            work_dirs.append(work_dir)
            landed = _land_map_stage(k, staged_sources, workers, hub, work_dir, len(staged_sources))
            if landed is None:
                return None
            scan, staged = landed
            rebuilt_operands.append(scan)
            staged_sources.append(staged)
        rebuilt: LogicalPlan = with_children(node, rebuilt_operands)
        for above in reversed(chain):
            rebuilt = with_children(above, [rebuilt])
        return _dispatch(rebuilt, staged_sources, workers, transport, hub, materialize=True)
    finally:
        for work_dir in work_dirs:
            _rmtree(work_dir)


def _unsupported(plan: LogicalPlan, sources: list[Source], reason: str):
    """Either fail loudly (the silent-single-node antipattern) or run a legitimately
    single-node-only plan on one node.

    Silent single-node fallback for a plan that *should* be distributed is an
    antipattern: it masks a missing distributed path behind a quiet perf cliff (the whole
    job on one node) and an OOM risk. So when any input is a **splittable** storage source
    (real distributed data), raise loudly with the shape — the gap must be fixed, not
    hidden. When every source is in-memory / non-splittable there is no distributed data
    to speak of, so executing it on one node is the correct plan, not a fallback.

    The splittable check is scoped to the sources **this plan actually reads**
    (`scanned_source_ids`), not the whole ambient `sources` list: a later adaptive stage — e.g. a
    trailing `project` over an in-memory intermediate materialized by an earlier stage —
    reads only that in-memory source and is correctly a single-node local transform, even
    though the original splittable scan source is still present in `sources` (unused here).
    Without this scoping such a tail stage would wrongly raise as "unsupported."
    """
    read_ids = scanned_source_ids(plan)
    if any(_is_splittable_source(sources[i]) for i in read_ids if i < len(sources)):
        from batcher._internal.errors import PlanError
        from batcher.dist.executors.plan_analysis import requires_staging

        # A join over a multi-source operand HAS a distributed path — the staged one. The
        # caller reached here only by forcing `adaptive=False`, so say that rather than
        # implying the operator is missing.
        hint = (
            "this shape distributes stage by stage (a join whose operand spans two sources, "
            "or a pipeline breaker beneath another breaker); it was disabled by an explicit "
            '`adaptive=False`. Re-run with `adaptive=True` (or the default `"auto"`). '
            "Running it in one shot would evaluate the inner plan once per partition and "
            "return wrong values, so it is refused rather than computed."
            if requires_staging(plan)
            else "File/extend the distributed operator, or run with distributed=False "
            "to force single-node explicitly."
        )
        raise PlanError(
            "distributed execution has no path for this plan shape "
            f"({reason}); refusing to silently fall back to single-node on distributed "
            f"data. {hint}"
        )
    _warn_accelerator_stage_falls_back(plan, reason)
    return _single_node(plan, sources)


def _warn_accelerator_stage_falls_back(plan: LogicalPlan, reason: str) -> None:
    """Say so when the single-node fallback is about to run a device stage on the driver.

    The fallback above is correct for CPU work: with no splittable source there is no
    distributed data, so one node is the right plan rather than a perf cliff. That reasoning
    does not carry to a stage holding a device. The driver of a GPU cluster is routinely the
    one node *without* one — a CPU head node with GPU workers is the ordinary shape — so the
    same fallback silently runs the model on CPU, returning the right answer arbitrarily
    slower with nothing said, after the user asked for `distributed=True` and named a device.

    Measured on the 4xT4 cluster: `group_by(...).agg(...)` feeding
    `ds.ml.map_batches(Model, num_gpus=1)` under `collect(distributed=True)` ran every batch
    on the driver's CPU, while all four devices sat idle — because a shuffle beneath a
    `map_batches` is a shape the dispatcher has no one-shot path for and the intermediate is
    in-memory, so it landed here.

    A warning rather than a raise: the query is correct, and failing a pipeline that works
    today would be a worse trade than telling its author what it is costing them.
    """
    from batcher.plan.accelerator import plan_requests_accelerator

    if not plan_requests_accelerator(plan):
        return
    from batcher._internal.hardware.devices.presence import local_accelerator_present

    if local_accelerator_present() is not False:
        return
    import warnings

    from batcher._internal.errors import PerformanceWarning

    warnings.warn(
        "a stage of this query requested an accelerator, but the distributed executor has "
        f"no path for this plan shape ({reason}) and fell back to running it in this "
        "process, which has no accelerator — the model will run on CPU. Materializing the "
        "stage before the accelerator stage (for example `.collect()` then re-reading, or "
        "writing the intermediate out) lets the inference stage distribute on its own.",
        PerformanceWarning,
        stacklevel=3,
    )


# --- ASOF join (co-partition by the `by` keys) --------------------------------
# Lives here (not in the `executors` subpackage, which is at its file-count ceiling)
# alongside the dispatch that routes to it. An ASOF match only ever pairs rows that
# share a `by` group, and `partition_batches` hashes equal `by` values to the same
# bucket on both sides, so each bucket is an independent ASOF join whose union is the
# full result. It reuses the equi-join's generic map/reduce tasks verbatim — only the
# partition keys (`by`) and the reducer IR (`asof_join`) differ.


def _require_shared_scratch(op: str) -> None:
    """Refuse a disk-shuffle-only operator on a multi-node cluster with no shared scratch.

    The disk shuffle hands only *paths* between tasks, so `work_dir` must resolve on every
    node. Every other operator is steered off disk by `resolve_transport` (which picks
    Flight the moment the cluster spans more than one node), but `op` has no Flight path, so
    nothing protects it: a worker would open a driver-local `/tmp` path that does not exist
    on its own node — a `FileNotFoundError` at best, and silently missing rows if a
    same-named directory happens to exist there. Fail with the fix instead.
    """
    from batcher.dist.shuffle_io import shared_scratch_root

    if shared_scratch_root() is not None or alive_node_count() <= 1:
        return
    from batcher._internal.errors import PlanError

    raise PlanError(
        f"distributed `{op}` uses the disk shuffle, which needs a scratch directory every "
        "worker node can reach, and this cluster has no shared mount. Point "
        "`MemoryConfig.spill_dir` at a shared filesystem, or run it single-node "
        "(`distributed=False`)."
    )


def _asof_reducer_ir(asof: AsofJoin) -> dict:
    """IR for the per-bucket ASOF join of a left input (source 0) and right input
    (source 1). The node's own `shape_ir()` with the per-task scans substituted, so a new
    ASOF field crosses the cluster without anyone remembering to add it here."""
    return binary_task_ir(asof)


def _distributed_asof(
    above: list[LogicalPlan], asof: AsofJoin, sources: list[Source], workers: int
) -> pa.Table:
    """Co-partition both sides by the `by` keys and ASOF-join each bucket in parallel."""
    from batcher.carbonite.resilience import gather_with_backups
    from batcher.dist.executors.join import _join_map_task, _join_reduce_task
    from batcher.dist.executors.ray_runtime import speculation_policy
    from batcher.dist.shuffle_io import read_ipc

    _ensure_ray(workers)
    _require_shared_scratch("asof_join")
    cfg_json = engine_config_json()  # driver config → shipped to workers

    left_plan, left_sid = _relabel_single_source(asof.left)
    right_plan, right_sid = _relabel_single_source(asof.right)
    left_ir = json.dumps(left_plan.to_ir())
    right_ir = json.dumps(right_plan.to_ir())
    asof_ir = json.dumps(_asof_reducer_ir(asof))
    left_proj, left_pred = source_pushdown(left_plan, 0)
    right_proj, right_pred = source_pushdown(right_plan, 0)

    from batcher.dist.shuffle_io import distributed_work_dir

    work_dir = distributed_work_dir("batcher_asof_")
    try:
        left_parts = _partition_source(
            sources[left_sid], workers, work_dir, tag="L", projection=left_proj, predicate=left_pred
        )
        right_parts = _partition_source(
            sources[right_sid],
            workers,
            work_dir,
            tag="R",
            projection=right_proj,
            predicate=right_pred,
        )

        pol = speculation_policy()

        # Co-partition each side by its `by` keys with the generic join map task.
        def _left_map_for(i: int):
            return _join_map_task.remote(
                left_ir, list(asof.left_by), left_parts[i], workers, work_dir, "L", i, cfg_json
            )

        def _right_map_for(i: int):
            return _join_map_task.remote(
                right_ir, list(asof.right_by), right_parts[i], workers, work_dir, "R", i, cfg_json
            )

        left_paths = gather_with_backups(
            [_left_map_for(i) for i in range(len(left_parts))], _left_map_for, pol
        )  # [mapper][bucket]
        right_paths = gather_with_backups(
            [_right_map_for(i) for i in range(len(right_parts))], _right_map_for, pol
        )

        def _reduce_for(r: int):
            l_inputs = [paths[r] for paths in left_paths]
            r_inputs = [paths[r] for paths in right_paths]
            return _join_reduce_task.remote(asof_ir, l_inputs, r_inputs, work_dir, r, cfg_json)

        result_paths = gather_with_backups(
            [_reduce_for(r) for r in range(workers)], _reduce_for, pol
        )

        batches: list[pa.RecordBatch] = []
        # `_join_reduce_task` yields `(path, rows, ...)` — it grew a trailing metrics field —
        # so unpack only what this path needs rather than pinning the tuple's arity.
        for entry in result_paths:
            path = entry[0]
            if path is not None:
                batches.extend(read_ipc(path))
    finally:
        _rmtree(work_dir)

    if not batches:
        result = empty_result_table(asof, [o.alias for o in asof.output])
    else:
        result = pa.Table.from_batches(batches)
    return result if not above else _apply_above(above, result)


# --- Keyless ASOF join (range-partition both sides on the `on` key) ------------
# The `by`-keyed ASOF above co-partitions by hash, because an ASOF match only ever pairs
# rows inside one `by` group. A KEYLESS ASOF has no such group: any left row may match any
# right row, and which one it matches is decided by a global order on `on`. Hashing is
# therefore not merely unbalanced, it is wrong.
#
# Range partitioning is the shape that works, and it is the same one the distributed sort
# already uses: cut `on` into ordered intervals from a sampled quantile grid, and send both
# sides through the *same* boundaries. Bucket `r` then holds every left row and every right
# row whose key lies in interval `r`, so a match that lives inside the interval is already
# local.
#
# What is left is the match that does not. A left row in bucket `r` can match a right row in
# an earlier bucket (`backward`) or a later one (`forward`) — and the gap is unbounded, so no
# fixed overlap covers it. Exactly one row per direction does: intervals are ordered, so
# among all right rows below the bucket the only one that can ever be the backward match is
# the LARGEST, and among all above it the only forward candidate is the SMALLEST. Carrying
# those two rows into the bucket makes each reducer's ASOF exact, and costs O(buckets) rows
# rather than O(rows).
#
# The extremes are measured where the rows already are — inside the right side's range task,
# which has the bucket in hand — so the carry costs no extra pass over the data.


def _asof_range_task(
    map_ir,
    key_name,
    boundaries,
    n_buckets,
    part_path,
    work_dir,
    tag,
    mapper_id,
    engine_config,
    extremes=False,
    fallback_schema=None,
):
    """Range-partition one side of a keyless ASOF by its `on` key, into `n_buckets` files.

    The sort's `_range_task` does the same partitioning and is not reused, because the two
    tasks differ in what they *return*, not in how they partition: the shared per-row work
    is `bucketize`, which both call. A sort's version additionally carries the skew split and
    the descending / nulls-first ordering a `Sort` node owns; an ASOF key is always ascending
    and never split, and it needs the one thing a sort never asks for — each bucket's extreme
    rows, which is what lets a reducer see the match that landed in someone else's bucket.

    With `extremes`, a second tiny file is written holding at most two rows per bucket (the
    smallest and largest `on` in that bucket, from this mapper's rows), tagged with their
    bucket in a ``__bt_asof_bucket`` column. The driver folds those across mappers into the
    per-bucket carry. Measuring them here is what keeps the carry free: the bucket is already
    materialized in this task, so nothing re-reads it.

    `fallback_schema` is this side's statically known Arrow schema, used for the empty
    files a mapper that read no rows still has to publish.

    Returns `(bucket_paths, extremes_path_or_None, metrics_json)`.
    """
    import os as _os

    from batcher.dist.executors.partition_io import bucketize, read_partition
    from batcher.dist.executors.ray_runtime import execute_metered
    from batcher.dist.shuffle_io import write_ipc

    rows, metrics_json = execute_metered(map_ir, [read_partition(part_path)], engine_config)
    # A mapper that returned no batch at all has no schema of its own, and the empty bucket
    # files it publishes would then carry no columns — which the reducer reports as "join
    # over an empty input side (no input schema)" rather than as an empty join, and an ASOF
    # reducer meets an empty right bucket routinely (the right side is the sparse one, and
    # `_join_reduce_task` deliberately does not short-circuit a left-style join). The engine
    # returns a zero-row batch that still carries its schema, so this is a backstop rather
    # than the common path; the driver knows the schema statically, so the backstop is free.
    schema = rows[0].schema if rows else (fallback_schema or pa.schema([]))
    # Ascending, nulls last: an ASOF key is ordered one way only, and a null key matches
    # nothing on either side, so which end it lands on cannot change a row.
    buckets = bucketize(rows, key_name, boundaries, n_buckets, False, False)
    paths = []
    for r in range(n_buckets):
        path = _os.path.join(work_dir, f"{tag}m{mapper_id}_r{r}.arrow")
        # An empty bucket still gets a schema-only file so every mapper publishes exactly
        # `n_buckets` paths and the reducer can index by bucket.
        batches = buckets[r] or [pa.RecordBatch.from_pylist([], schema=schema)]
        write_ipc(batches, path)
        paths.append(path)
    if not extremes:
        return paths, None, metrics_json
    ext = _bucket_extremes(buckets, key_name)
    if ext is None:
        return paths, None, metrics_json
    ext_path = _os.path.join(work_dir, f"{tag}ext{mapper_id}.arrow")
    write_ipc(ext.to_batches(), ext_path)
    return paths, ext_path, metrics_json


#: The column a carry row's bucket index rides in, between the range task that measures it
#: and the driver that folds it. Prefixed so it cannot collide with a user column, and
#: dropped before the row is ever handed to the reducer.
_ASOF_BUCKET = "__bt_asof_bucket"


def _bucket_extremes(buckets: list[list[pa.RecordBatch]], key_name: str) -> pa.Table | None:
    """The smallest- and largest-key row of each non-empty bucket, tagged with its bucket.

    Two rows per bucket at most, so the result is O(buckets) regardless of how many rows the
    mapper held. Nulls are ignored (``min_max`` skips them), which is what we want: a null
    `on` key matches nothing, so a null row is never anyone's nearest match.

    **Which row of a tie group** is not a detail. When several rows share the extreme key,
    the engine's ASOF picks the one nearest the probe in sorted order — the LAST of the group
    for a backward match and the FIRST for a forward one (measured, not assumed:
    ``tests/unit/test_dist_asof_keyless.py`` pins it). So the max carries its group's last row
    and the min carries its group's first, and a bucket whose keys are all equal contributes
    both. Keeping an arbitrary member instead returned a neighbouring row's payload — the
    right key, the wrong value, and nothing to distinguish it from a correct answer.
    """
    import pyarrow.compute as pc

    keep: list[pa.Table] = []

    def _tag(row: pa.Table, r: int) -> pa.Table:
        return row.append_column(_ASOF_BUCKET, pa.array([r], pa.int32()))

    for r, bucket in enumerate(buckets):
        if not bucket:
            continue
        table = pa.Table.from_batches(bucket)
        column = table.column(key_name)
        span = pc.min_max(column)
        low, high = span["min"], span["max"]
        if not low.is_valid:
            continue  # every key in this bucket is null
        low_group = table.filter(pc.equal(column, low))
        keep.append(_tag(low_group.slice(0, 1), r))
        high_group = low_group if high == low else table.filter(pc.equal(column, high))
        if high != low or high_group.num_rows > 1:
            keep.append(_tag(high_group.slice(high_group.num_rows - 1, 1), r))
    if not keep:
        return None
    return pa.concat_tables(keep)


def _asof_carry_rows(
    extremes: list[pa.Table], n_buckets: int, direction: str, key_name: str
) -> list[pa.Table | None]:
    """Fold per-mapper bucket extremes into the row each reducer must be lent.

    `extremes[i]` is one mapper's tagged extreme rows. For every bucket this computes the
    single largest key strictly below it (`backward`), the single smallest strictly above it
    (`forward`), or both (`nearest`) — which is exactly the set of out-of-bucket rows that
    can win an ASOF match, because the buckets are ordered intervals.

    Args:
        extremes: The per-mapper extreme-row tables, `__bt_asof_bucket` still attached.
        n_buckets: How many range buckets the sides were partitioned into.
        direction: The ASOF direction, deciding which side's carry is needed.
        key_name: The right side's `on` column.

    Returns:
        One table per bucket holding its carry rows (tag column dropped), or `None` where
        the bucket needs none.
    """
    import pyarrow.compute as pc

    out: list[pa.Table | None] = [None] * n_buckets
    if not extremes:
        return out
    tagged = pa.concat_tables(extremes)
    if tagged.num_rows == 0:
        return out
    plain = tagged.drop_columns([_ASOF_BUCKET])
    bucket_of = tagged.column(_ASOF_BUCKET)

    def _pick(mask, best: str) -> pa.Table | None:
        """The one row of `plain` where `mask` holds whose key is the `best` one.

        Ties are broken the way the engine breaks them (see `_bucket_extremes`): the last
        candidate for a `max` (backward), the first for a `min` (forward). Candidates arrive
        in mapper order, which is the order the reducer concatenates the mappers' bucket
        files in, so the two agree.
        """
        rows = plain.filter(mask)
        if rows.num_rows == 0:
            return None
        span = pc.min_max(rows.column(key_name))[best]
        if not span.is_valid:
            return None
        tied = rows.filter(pc.equal(rows.column(key_name), span))
        return tied.slice(tied.num_rows - 1, 1) if best == "max" else tied.slice(0, 1)

    wants_backward = direction in ("backward", "nearest")
    wants_forward = direction in ("forward", "nearest")
    for r in range(n_buckets):
        parts = []
        if wants_backward:
            below = _pick(pc.less(bucket_of, r), "max")
            if below is not None:
                parts.append(below)
        if wants_forward:
            above = _pick(pc.greater(bucket_of, r), "min")
            if above is not None:
                parts.append(above)
        if parts:
            out[r] = pa.concat_tables(parts)
    return out


def _side_schema(plan: LogicalPlan) -> pa.Schema | None:
    """One side's statically known Arrow schema, or `None` when the plan cannot infer it.

    `LogicalPlan.available_schema` is the engine's own type analysis, so this asks it rather
    than reading a row — which is the point: it answers for a side that turns out to be
    empty, which is exactly when a mapper has no schema to publish.
    """
    schema = plan.available_schema()
    return schema.arrow if schema is not None else None


def _distributed_asof_keyless(
    above: list[LogicalPlan], asof: AsofJoin, sources: list[Source], workers: int
) -> pa.Table:
    """Range-partition both sides on `on` and ASOF-join each interval, with a carried row.

    The keyless twin of `_distributed_asof`. See the module comment above for why the
    boundaries plus one carry row per direction make each bucket's join exact.
    """
    import os

    from batcher.carbonite.resilience import gather_with_backups
    from batcher.dist.executors.join import _join_reduce_task
    from batcher.dist.executors.partition_io import merge_boundaries, sample_probs
    from batcher.dist.executors.ray_runtime import speculation_policy
    from batcher.dist.executors.sort import _sample_task
    from batcher.dist.shuffle_io import distributed_work_dir, read_ipc, write_ipc

    _ensure_ray(workers)
    _require_shared_scratch("keyless asof_join")
    cfg_json = engine_config_json()

    left_plan, left_sid = _relabel_single_source(asof.left)
    right_plan, right_sid = _relabel_single_source(asof.right)
    left_ir = json.dumps(left_plan.to_ir())
    right_ir = json.dumps(right_plan.to_ir())
    asof_ir = json.dumps(_asof_reducer_ir(asof))
    left_proj, left_pred = source_pushdown(left_plan, 0)
    right_proj, right_pred = source_pushdown(right_plan, 0)

    work_dir = distributed_work_dir("batcher_asofk_")
    try:
        left_parts = _partition_source(
            sources[left_sid], workers, work_dir, tag="L", projection=left_proj, predicate=left_pred
        )
        right_parts = _partition_source(
            sources[right_sid],
            workers,
            work_dir,
            tag="R",
            projection=right_proj,
            predicate=right_pred,
        )
        pol = speculation_policy()
        n_buckets = max(1, workers)

        # Sample the LEFT key, because the left side decides the output's row count: an ASOF
        # is left-style, so every left row emits exactly one output row and balancing the
        # left is balancing the reducers' work.
        probs = sample_probs(n_buckets, len(left_parts))

        def _sample_for(i: int):
            return _sample_task.remote(left_ir, asof.left_on, probs, left_parts[i], cfg_json)

        grids = gather_with_backups(
            [_sample_for(i) for i in range(len(left_parts))], _sample_for, pol
        )
        boundaries = merge_boundaries(grids, n_buckets)

        left_schema = _side_schema(left_plan)
        right_schema = _side_schema(right_plan)

        def _left_range_for(i: int):
            return _asof_range_task.remote(
                left_ir,
                asof.left_on,
                boundaries,
                n_buckets,
                left_parts[i],
                work_dir,
                "L",
                i,
                cfg_json,
                False,
                left_schema,
            )

        def _right_range_for(i: int):
            return _asof_range_task.remote(
                right_ir,
                asof.right_on,
                boundaries,
                n_buckets,
                right_parts[i],
                work_dir,
                "R",
                i,
                cfg_json,
                True,
                right_schema,
            )

        left_out = gather_with_backups(
            [_left_range_for(i) for i in range(len(left_parts))], _left_range_for, pol
        )
        right_out = gather_with_backups(
            [_right_range_for(i) for i in range(len(right_parts))], _right_range_for, pol
        )
        left_paths = [entry[0] for entry in left_out]
        right_paths = [entry[0] for entry in right_out]

        extremes = [
            pa.Table.from_batches(read_ipc(entry[1])) for entry in right_out if entry[1] is not None
        ]
        carries = _asof_carry_rows(
            [t for t in extremes if t.num_rows], n_buckets, asof.direction, asof.right_on
        )
        carry_paths: list[str | None] = [None] * n_buckets
        for r, carry in enumerate(carries):
            if carry is not None and carry.num_rows:
                carry_paths[r] = write_ipc(
                    carry.to_batches(), os.path.join(work_dir, f"Rcarry_{r}.arrow")
                )

        def _reduce_for(r: int):
            l_inputs = [paths[r] for paths in left_paths]
            r_inputs = [paths[r] for paths in right_paths]
            if carry_paths[r] is not None:
                r_inputs = [*r_inputs, carry_paths[r]]
            return _join_reduce_task.remote(asof_ir, l_inputs, r_inputs, work_dir, r, cfg_json)

        result_paths = gather_with_backups(
            [_reduce_for(r) for r in range(n_buckets)], _reduce_for, pol
        )

        # Buckets are ordered intervals, so concatenating them in bucket order returns the
        # rows in `on` order. That is a permutation of the single-node result rather than a
        # match for it — a single-node ASOF emits rows in LEFT INPUT order — exactly as the
        # `by`-keyed path's hash buckets already are, and as every distributed join is.
        batches: list[pa.RecordBatch] = []
        for entry in result_paths:
            path = entry[0]
            if path is not None:
                batches.extend(read_ipc(path))
    finally:
        _rmtree(work_dir)

    if not batches:
        result = empty_result_table(asof, [o.alias for o in asof.output])
    else:
        result = pa.Table.from_batches(batches)
    return result if not above else _apply_above(above, result)


# --- RANGE (inequality) join (broadcast the build side) ------------------------
# Also here rather than in the `executors` subpackage, which is at its file-count ceiling.
# Unlike every other join, an inequality has no equality to co-partition on: hashing `a.x`
# and the `b.y` values it is less than sends them to different buckets, so the shuffle is
# not merely slower, it is wrong. Broadcast is the shape that works, and the probe
# machinery is the equi-join's (`broadcast_probe_join`) with only the reducer IR differing.


def _range_join_reducer_ir(rj: RangeJoin) -> dict:
    """IR for the per-task range join of a left chunk (source 0) against the full right
    (source 1). The node's own `shape_ir()` with the per-task scans substituted, so a new
    field crosses the cluster without anyone remembering to add it here."""
    return binary_task_ir(rj)


def _distributed_range_join(
    above: list[LogicalPlan],
    rj: RangeJoin,
    sources: list[Source],
    workers: int,
    hub=None,
) -> pa.Table:
    """Broadcast the right side and range-join each partition of the left against it.

    Exact for the `_BROADCAST_SAFE` join types the dispatcher gates on: each probe task
    holds the whole right side, so a left row's match set is fully determined inside its
    own partition, and each left row is in exactly one partition. `right`/`full` are
    excluded because a right row's matched-ness spans partitions.

    When the right side does not fit the broadcast budget there is no second strategy to
    fall back to — the shuffle the equi-join would use does not exist for an inequality —
    so this raises with the actual fix rather than silently running the whole join on one
    node (`_unsupported`'s rule) or replicating an over-large side into an OOM.
    """
    from batcher.dist.executors.join import broadcast_probe_join

    result = broadcast_probe_join(
        above,
        rj,
        rj.left,
        rj.right,
        _range_join_reducer_ir(rj),
        sources,
        workers,
        hub=hub,
    )
    if result is not None:
        return result

    from batcher._internal.errors import PlanError

    raise PlanError(
        "distributed range (inequality) join requires one side small enough to broadcast, "
        "and this query's right side is empty or over the broadcast budget. An inequality "
        "has no join key to co-partition on, so there is no shuffle fallback. Filter or "
        "pre-aggregate the right side, raise `OptimizerConfig.broadcast_max_bytes`, or run "
        "it single-node (`distributed=False`)."
    )

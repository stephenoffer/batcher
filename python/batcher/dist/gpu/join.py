"""Run a translated join across every GPU, by splitting the probe side and broadcasting the build.

A join was the one relational shape still pinned to a single device, which is backwards: it is
the shape whose whole premise is that one input is large. The star-schema query it exists for —
a big fact table joined to a small dimension, then aggregated — could not use a second GPU no
matter how many the cluster had.

Splitting the **probe** side and giving every worker the whole **build** side is correct for
the join types whose output is driven by left rows (`plan.distribution.BROADCAST_SAFE_JOINS`).
Each shard emits the rows its own probe slice produces, and unioning them never duplicates a
build row. `right` and `full` must emit an unmatched build row exactly once, and every shard
sees the whole build side, so each would emit it — those keep the single-device path.

The build side is *read* by each worker rather than shipped to it. Reading a small dimension N
times from storage costs less than moving it through the object store N times, and it keeps
the rule that bulk Arrow does not travel as Ray objects. Whether the build side is small enough
for this at all is not decided here: the fan-out runs only when the planner already marked the
join `broadcast`, which is a cost decision Kyber owns and that the CPU join path reads the same
way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.io.source import Source

__all__ = ["sharded_gpu_join"]


def sharded_gpu_join(
    left: Source,
    right: Source,
    left_ops: list[dict],
    right_ops: list[dict],
    join_ir: dict,
    ops: list[dict],
    *,
    gpu_count: int,
    sharded: bool,
    broadcast: bool,
) -> pa.Table | None:
    """Run a translated join across the cluster's GPUs, or `None` when it does not apply.

    Args:
        left: The probe side's source; split across devices.
        right: The build side's source; read whole by every device.
        left_ops: The probe chain's operator IR.
        right_ops: The build chain's operator IR.
        join_ir: The join node's IR.
        ops: The operator chain above the join.
        gpu_count: The cluster's live device count.
        sharded: Whether the working set exceeds one device.
        broadcast: Kyber's verdict that this join's build side is small enough to replicate.

    Returns:
        The join's result, or `None` when the fan-out does not apply — a join type that is not
        broadcast-safe, a build side Kyber will not replicate, a chain above the join with no
        shardable split, an unsplittable probe side, or an unreadable cluster.
    """
    from batcher.plan.distribution import BROADCAST_SAFE_JOINS, ShardSplit, shard_plan

    if join_ir.get("join_type") not in BROADCAST_SAFE_JOINS:
        return None
    if not broadcast:
        # "Is the build side small enough" is Kyber's decision and it has already been made.
        # Re-deriving it here would give the two backends two different answers to one
        # question, and the wrong one is an out-of-memory on every device at once.
        return None
    # The chain ABOVE the join divides the same way any chain does; an empty one merges by
    # plain concatenation.
    above = shard_plan(ops) if ops else ShardSplit([], [], [])
    if above is None:
        return None

    from batcher.dist.gpu.aggregate import shard_descriptors
    from batcher.dist.gpu.dispatch import whole_source_descriptor

    # Which side is replicated is decided by **measuring both**, not by which one the plan put
    # on the right. See `_replicate_the_smaller_side` for why the plan's order does not answer
    # this and what it cost when this code assumed it did.
    swap = _replicate_the_smaller_side(left, right, join_ir)
    if swap:
        left, right = right, left
        left_ops, right_ops = right_ops, left_ops
        join_ir = _mirrored_join_ir(join_ir)

    build = whole_source_descriptor(right)
    if build is None:
        return None
    # The probe side reads only what its own pre-join chain names. Unnarrowed — which is how
    # this read — the fact table's every column crosses onto each device to be joined on two of
    # them. `left_ops` is the chain below the join, so it is exactly the leaf's want-set; what
    # the join and the chain above it need of the probe side is already inside those ops.
    from batcher.core.gpu_plan.pruning import chain_projection
    from batcher.dist.gpu.shards import narrowed_schema

    probe_projection = chain_projection(left_ops)
    probes = shard_descriptors(
        left, gpu_count, sharded=sharded, preserve_order=False, projection=probe_projection
    )
    if probes is None:
        return None

    build_bytes = _build_side_bytes(build, right)
    if not _replication_measures_up(build_bytes, left, probe_projection, gpu_count):
        return None

    shards = _run_join_shards(
        probes,
        build,
        left_ops,
        right_ops,
        join_ir,
        above.shard_ops,
        gpu_count=gpu_count,
        # Priced at the width the probe shards are actually read with, for the same reason the
        # aggregate fan-out is: packing asks how much of a device one shard holds, and a shard
        # read through a projection holds the projection.
        probe_schema=narrowed_schema(_schema_of(left), probe_projection),
        build_bytes=build_bytes,
    )
    if not shards:
        return None
    from batcher.dist.gpu.aggregate import fold_shards

    return fold_shards(shards, above)


def _replicate_the_smaller_side(left, right, join_ir: dict) -> bool:
    """Whether to mirror this join so the *smaller* relation is the one every device copies.

    The fan-out splits the probe side and gives every device the whole build side, and this
    module used to take the build side to be whatever the plan put on the **right**. Nothing
    guarantees that. Kyber reorders a join's inputs for its own costing, and the CPU hash join
    it reorders for picks its build side at runtime, so the plan's left/right carries no promise
    about size — it just happened to hold on the plans this path was written against.

    It stops holding as soon as a fleet learns anything, and the failure is silent and
    expensive. TPC-H q14 (`lineitem ⋈ part`) runs at **12.2x** against an empty statistics hub.
    Give the same query a hub with two of its own passes in it and Kyber emits the mirrored
    join — probe `part` (2 M rows), build `lineitem` (60 M) — so the fan-out tries to replicate
    **15.36 GB** onto every device, `_replication_measures_up` refuses it at a 4.79 GB budget,
    and the query silently leaves the accelerator. The learning loop made the device tier worse
    the longer the fleet ran, on the *forced* `backend="gpu"` path.

    Measuring both sides answers it for good, and it is strictly better than the old rule rather
    than a different guess: a plan that already had the small side on the right is unchanged.

    **Only `inner` mirrors.** `LEFT_DRIVEN_JOINS` also carries `left`, `semi` and `anti`, and
    all three are asymmetric by definition — their output is driven by left rows, so exchanging
    the sides changes the answer rather than the schedule. An inner join is commutative, and
    `_mirrored_join_ir` keeps even the column order identical.

    Args:
        left: The side the plan puts on the left, which the fan-out would split.
        right: The side the plan puts on the right, which every device would copy.
        join_ir: The join node's IR.

    Returns:
        True when the sides should be exchanged before the fan-out runs.
    """
    if join_ir.get("join_type") != "inner":
        return False
    try:
        return _relation_bytes(right) > _relation_bytes(left) > 0.0
    except Exception as exc:  # pragma: no cover - sizing must never break the join
        note_suppressed("dist", "size both join sides to pick the replicated one", exc)
        return False


def _relation_bytes(source) -> float:
    """A relation's decoded size, or `0.0` when it will not say.

    A zero on either side leaves the plan's own order standing, which is what this path did
    before it measured anything.
    """
    from batcher.dist.gpu.shards import source_bytes

    return float(source_bytes(source))


def _mirrored_join_ir(join_ir: dict) -> dict:
    """`join_ir` with its two sides exchanged, producing the identical output columns.

    The translator builds a join's result explicitly from `join_ir["output"]` — one entry per
    column carrying the `side` it comes from, its `name` there, and its `alias` — rather than
    from whatever order the underlying `merge` returns. So exchanging the sides is fully
    expressible: swap the key lists, and flip each output entry's `side`. Column identity,
    column order and column type all survive, which is what makes this a scheduling change and
    not a semantic one.
    """
    flip = {"left": "right", "right": "left"}
    return {
        **join_ir,
        "left_keys": list(join_ir["right_keys"]),
        "right_keys": list(join_ir["left_keys"]),
        "output": [{**o, "side": flip[o["side"]]} for o in join_ir["output"]],
    }


def _schema_of(source):
    """A source's schema, or `None` when it will not describe itself.

    Used only to price the probe shards for packing, so a source that declines costs the join
    its fractional share and nothing else — the tasks then ask for whole devices, as they did.
    """
    try:
        return source.schema()
    except Exception as exc:
        note_suppressed("dist", "read the probe schema for gpu join packing", exc)
        return None


def _build_side_bytes(build: dict, right) -> float:
    """How much device memory the replicated build side occupies in *each* task.

    A broadcast join hands every task the whole build side. Co-tenants on one device therefore
    hold one copy each, not one between them, and a packing decision that charged the build
    side once would put four tasks on a device that is about to hold four copies of it. This
    is the figure that keeps the join from being the one shape where packing OOMs a device.

    Returns:
        Bytes, `0.0` when the build side's size cannot be read — which makes the packing
        under-state the need, so the caller must keep the subdivision ladder behind it.
    """
    from batcher.dist.gpu.resources import descriptor_bytes

    schema = _schema_of(right)
    if schema is None:
        return 0.0
    from batcher.plan.types import schema_row_bytes

    return float(descriptor_bytes(build, schema_row_bytes(schema)))


def _run_join_shards(
    probes: list,
    build: dict,
    left_ops,
    right_ops,
    join_ir: dict,
    above_ops: list[dict],
    *,
    gpu_count: int = 0,
    probe_schema=None,
    build_bytes: float = 0.0,
) -> list:
    """Join every probe shard against the whole build side, recovering from a failed shard.

    Recovery is the aggregate path's, minus the CPU substitute: reconstructing a join shard
    through the engine would need a two-source plan whose second input is this build side, and
    a half-supported fallback is worse than an honest one. A failed shard is subdivided on the
    device, and a shard that still fails abandons the fan-out — the caller then runs the join
    as a single dispatch, or on the CPU engine.
    """
    import ray

    from batcher.carbonite.resilience import gather_with_backups
    from batcher.config import active_config
    from batcher.dist.executors.ray_runtime import speculation_policy
    from batcher.dist.gpu.resources import gpu_shard_options, shard_node_affinity
    from batcher.dist.gpu.shards import ShardReport, is_memory_failure, run_subdivided
    from batcher.dist.gpu.tasks import gpu_join_task, gpu_task_options

    dc = active_config().distributed
    opts, packing = gpu_shard_options(
        probes, probe_schema, gpu_count=gpu_count, resident_bytes=build_bytes
    )
    task = ray.remote(**opts)(gpu_join_task)
    # The probe shards prefer the node that last read them, for the same reason the aggregate
    # fan-out's do. The build side is read whole by every node either way.
    affinity = shard_node_affinity(probes)
    # A probe shard that did not fit its packed share is retried on a whole device: the share is
    # the thing that was just shown to be too small, and a join's retry also carries the whole
    # replicated build side, which is the part of the footprint subdividing the probe cannot
    # shrink. Identical to `task` when nothing was packed.
    retry_task = ray.remote(**gpu_task_options())(gpu_join_task) if packing.packed else task

    report = ShardReport("gpu-join", len(probes), packing=packing)

    def _launch(i: int):
        args = (probes[i], build, left_ops, right_ops, join_ir, above_ops)
        if affinity:
            return task.options(scheduling_strategy=affinity[i]).remote(*args)
        return task.remote(*args)

    def _on_failure(i: int, _ref, exc):
        if not is_memory_failure(exc) or dc.gpu_shard_subdivide <= 1:
            raise exc
        note_suppressed("dist", f"gpu join shard {i} did not fit; subdividing", exc)
        report.note_subdivided()
        return run_subdivided(
            probes[i],
            lambda d: ray.get(retry_task.remote(d, build, left_ops, right_ops, join_ir, above_ops)),
            parts=int(dc.gpu_shard_subdivide),
            rounds=int(dc.gpu_shard_subdivide_rounds),
            cause=exc,
        )

    refs = [_launch(i) for i in range(len(probes))]
    results = gather_with_backups(refs, _launch, speculation_policy(), on_failure=_on_failure)
    report.publish()
    return [t for t in results if t is not None and t.num_rows]


def _replication_measures_up(
    build_bytes: float, probe: Source, projection: list[str] | None, gpu_count: int
) -> bool:
    """Whether the *measured* build side still fits beside a probe shard on every device.

    Kyber decided to replicate from estimates, and an estimate of a join's inputs is what this
    engine's cardinality model is least reliable about — TPC-H q14 and q17 at sf10 estimate
    **one row** for relations of tens of millions. So the executor asks again with what it can
    count: `build_bytes` comes from the descriptor's footer row count and the source's own
    schema.

    It asks only about **fit**, and that restraint is the correction to a rule that was here
    first and was wrong. That rule compared the *aggregate bytes* replication reads
    (`build x devices`) against the probe side it splits — and aggregate bytes is not what a
    fan-out costs, because the devices read **concurrently**. Per device the shapes are
    `build + probe/N` against `build + probe`, so replicating is never slower than the single
    device it replaces; it is only ever *not faster*. Measured on six T4s at TPC-H sf10, the
    byte rule refused q14 — 0.48 GB replicated against a 1.68 GB probe — and turned a **12.4x**
    speedup into a decline.

    Whether the fan-out buys enough to be worth running at all is a different question, decided
    on the plan by `kyber.gpu.shape`. This is the one the executor is uniquely able to answer:
    the planner's estimate said it fits, and only the descriptor knows.

    This mirrors the CPU path, whose threshold documents that "the executor re-checks the
    *measured* build side against this same number before replicating it, so a planner
    under-estimate costs a fallback rather than a cluster-wide OOM". Measured on six T4s,
    TPC-H q12 at sf10 replicates **15.4 GB** against a 4.8 GB budget on the strength of an
    estimate, and took 24.1 s against the CPU engine's 1.3 s.

    Args:
        build_bytes: The measured size of the side every device would read.
        probe: The side that would be split across devices. Unused by the fit test and kept in
            the signature because the caller has it and the next question about this join is
            about the pair.
        projection: The columns the probe side is read with.
        gpu_count: Devices that would each hold a copy.

    Returns:
        True when the measured build side fits the device budget, and whenever it cannot be
        measured — an unmeasurable input is exactly where Kyber's estimate is all there is, and
        overriding a planner decision on no evidence would decline the fan-out this protects.
    """
    del probe, projection  # the fit question needs neither; see the docstring
    if max(1, int(gpu_count)) < 2 or build_bytes <= 0:
        return True
    from batcher.config import active_config

    budget = active_config().distributed.device_replication_bytes()
    if build_bytes <= budget:
        return True
    note_suppressed(
        "dist",
        "replicate this join's build side across the devices",
        ResourceWarning(
            f"{build_bytes / 1e9:.1f}GB measured is past the {budget / 1e9:.1f}GB device budget"
        ),
    )
    return False

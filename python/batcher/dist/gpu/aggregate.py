"""Run a translated GPU chain ending in an aggregate across every GPU in the cluster.

The single-device GPU path is bounded by one device's memory, which is the wrong ceiling for
the workloads a GPU is worth using for. This shards the chain instead: each device reads its
own slice of the source, runs the whole chain plus the *partial* stage of the mergeable
decomposition, and the small per-group partials are folded once at the end. Per-device memory
is then a function of the shard count rather than of the input, so the same query runs on data
many times larger than any one device.

Two properties make this safe rather than merely faster. The decomposition is expressed in the
plan IR (`core.gpu_plan.mergeable`), so partial and combine run through the same translator
every other operator does — the multi-device answer equals the single-device one by
construction. And a shard that cannot run on a device is recomputed by the **native CPU
engine** on a CPU worker, which produces the identical partial; losing a device costs that
shard's time and nothing else, where the older path abandoned the accelerated run entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed
from batcher.config import active_config
from batcher.dist.gpu.fabric import adaptive_shard_factor
from batcher.dist.gpu.shards import plan_shard_count, source_bytes

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.io.source import Source

__all__ = ["fold_shards", "merge_shards", "shard_descriptors", "sharded_gpu_aggregate"]


def sharded_gpu_aggregate(
    source: Source, ops: list[dict], *, gpu_count: int, sharded: bool
) -> pa.Table | None:
    """Run a translated chain across the GPUs, reducing per device and folding once.

    Args:
        source: The scan's source; must be splittable for the fan-out to be worth it.
        ops: The bottom-up operator IR chain. Its reducing prefix runs per device; anything
            above the reducer runs once on the folded result.
        gpu_count: The cluster's live device count.
        sharded: Whether the working set exceeds one device, so the chain must fan out.
            `False` still runs on a worker (which reads the source itself) but as one shard.

    Returns:
        The chain's result, or `None` when the fan-out does not apply — an unsplittable source,
        no shardable split, an unreadable cluster — so the caller can use the single-device
        dispatch or the CPU engine instead.
    """
    from batcher.core.gpu_plan.pruning import chain_predicate, chain_projection
    from batcher.plan.distribution import shard_plan

    split = shard_plan(ops)
    if split is None:
        return None

    # Narrow the read to the columns the chain names. `shard_descriptors` has taken a
    # projection all along and the *tree* fan-out passes one; this one did not, so the
    # commonest accelerated shape there is read every column of the fact table to answer a
    # three-column query. Derived from the whole `ops` chain rather than `split.shard_ops`:
    # what runs above the reducer sees only the reducer's output aliases, so the full chain is
    # both the correct question and the narrower answer.
    projection = chain_projection(ops)
    # ...and narrow it to the *rows* the chain keeps, from the same chain, because the CPU scan
    # path has taken a pushed predicate all along and this one passed `None`. It prunes nothing
    # on TPC-H — `lineitem` is written in orderkey order, so no row-group's bounds rule it out
    # (q6: 1961 splits either way) — and pays on a table clustered on what it filters.
    predicate = chain_predicate(ops)
    descriptors = shard_descriptors(
        source,
        gpu_count,
        sharded=sharded,
        preserve_order=split.ordered,
        projection=projection,
        predicate=predicate,
    )
    if descriptors is None:
        return None
    # The same projection prices the shards. Packing asks how much of a device one shard holds,
    # and a shard that reads three of sixteen columns holds three of them — pricing it at the
    # relation's full width packs fewer tasks onto each board than fit.
    partials = _run_shards(
        descriptors, split.shard_ops, _source_schema(source, projection), gpu_count
    )
    if not partials:
        return None
    return fold_shards(partials, split)


def _fleet_throughputs() -> tuple[float, ...]:
    """Rows per second per distinct device model in the fleet, from the learned statistics.

    Per *model* rather than per device: that is the granularity the learner keys on, and it is
    the one that matters — a fleet is uneven because it mixes an H100 with an L4, not because
    two H100s differ. Empty on a fleet whose models are unknown or unmeasured, which every
    consumer reads as "keep the configured shape".
    """
    try:
        from batcher.core.runtime import default_hub
        from batcher.dist.executors.ray_runtime.fabric import gpu_node_topology
        from batcher.kyber.gpu.adaptive import learned_device_throughput

        hub = default_hub()
        models = {node.accelerator_type for node in gpu_node_topology() if node.accelerator_type}
        rates = tuple(learned_device_throughput(hub, model) for model in sorted(models))
    except Exception as exc:  # a sizing hint must never fail a fan-out
        note_suppressed("dist", "read the fleet's measured device throughput", exc)
        return ()
    return tuple(r for r in rates if r > 0.0)


def shard_descriptors(
    source: Source,
    gpu_count: int,
    *,
    sharded: bool,
    preserve_order: bool,
    projection: list[str] | None = None,
    predicate: dict | None = None,
):
    """One partition descriptor per shard, or `None` when the source cannot be fanned out.

    A shard reads itself from storage, so the driver never materializes the source to hand it
    out. An in-memory source has no splits to describe, and is left to the caller's
    ship-the-table path.

    `preserve_order` is passed through for a row-local chain, whose merge is a concatenation in
    shard order: the shards have to *be* contiguous slices of the source for reassembling them
    to reproduce the single-node result. A fold does not care, and asking for ordering it does
    not need would only constrain how the source may be divided.

    `projection` narrows what each shard reads to the columns the plan actually uses. It is the
    difference between moving a fact table's sixteen columns onto a device and moving the four
    the query names — off storage, across the host link, and as resident device memory the shard
    is then sized against. `None` reads the relation as it is.

    `predicate` narrows the same read to the *rows* that can match, from the footer bounds a
    Parquet file already carries: a row-group ruled out never becomes a shard, so it is never
    opened, balanced or shipped. It prunes only — the chain's own `filter` re-checks every
    surviving row — so `None`, or a predicate a source cannot use, costs rows nobody wanted
    rather than rows somebody did.
    """
    # Only Ray is optional here, so only Ray's import is tolerated. The batcher imports are
    # deliberately NOT in the `try`: they were, and when `_scan_splits` stopped being re-exported
    # the `except Exception` turned a broken import into "this source cannot be fanned out". The
    # whole multi-device path disabled itself, correctly and silently, behind a debug note.
    from batcher.dist.executors.partition_io import partition_descriptors
    from batcher.dist.executors.partition_io._sources import _scan_splits
    from batcher.dist.executors.ray_runtime import _ensure_ray
    from batcher.io.splits import WholeSourceSplit

    try:
        import ray
    except ImportError as exc:  # the `[ray]` extra is not installed
        note_suppressed("dist", "import ray for the GPU fan-out", exc)
        return None
    if not ray.is_initialized() or gpu_count < 1:
        return None
    splits = _scan_splits(source, gpu_count, predicate)
    if len(splits) == 1 and isinstance(splits[0], WholeSourceSplit):
        return None
    if sharded:
        # Oversubscribe past the device count: each shard is then bounded (no single-device
        # OOM on a large source), work load-balances finely across a heterogeneous fleet, and
        # a preempted shard's retry is 1/N of the work. Ray runs at most `gpu_count`
        # single-device tasks at once, so the surplus pipelines behind them.
        # Divided more finely when the fleet's own measurements say its devices differ. Ray
        # runs one task per device at a time, so an equal number of shards each means the
        # stage ends when the slowest device ends and the fast ones idle from then on. A
        # uniform (or unmeasured) fleet keeps exactly the configured factor.
        #
        # ...and then bounded by how much data there actually is. The factor above says how
        # many shards a device *may* pipeline; it does not say that cutting them pays. Sized
        # from the fleet alone, a 6M-row scan on sixteen devices became 64 tasks of a hundred
        # thousand rows, each paying a worker dispatch, a cuDF first touch and a device
        # allocator setup: TPC-H q6 at sf1 measured **196 s** that way against 0.12 s on the
        # CPU engine, essentially all of it fixed cost. `plan_shard_count` keeps the fan-out
        # wide where the data is wide and collapses it to one shard where it is not.
        factor = adaptive_shard_factor(
            int(active_config().distributed.gpu_shard_oversubscribe), _fleet_throughputs()
        )
        n_shards = min(
            gpu_count * factor,
            plan_shard_count(source_bytes(source, projection), gpu_count, _device_bytes()),
        )
    else:
        n_shards = 1
    _ensure_ray(gpu_count)
    return partition_descriptors(
        source,
        n_shards,
        projection=projection,
        predicate=predicate,
        preserve_order=preserve_order,
    )


def _device_bytes() -> float:
    """One device's usable memory, or `0.0` when the cluster will not say.

    The *binding* device on a mixed fleet, since a shard sized for the largest one is a shard
    the smallest cannot hold. `0.0` leaves `plan_shard_count` to size on granularity alone,
    which is the bound that matters at small scale anyway.
    """
    from batcher.dist.executors.ray_runtime.accelerators import cluster_gpu_memory_gb

    try:
        gb = cluster_gpu_memory_gb()
    except Exception as exc:
        note_suppressed("dist", "read the fleet's device memory for shard sizing", exc)
        return 0.0
    return float(gb) * 1e9 if gb else 0.0


def _source_schema(source: Source, projection: list[str] | None = None):
    """The schema the shards are actually read with, or `None` when it cannot be read.

    Only ever used to price the shards for packing, so a source that will not describe itself
    costs the fan-out its fractional share and nothing else — the tasks then ask for whole
    devices, which is what they always did.

    `projection` narrows it to the columns the read will return, because that is the width a
    shard occupies on the device — see `shards.narrowed_schema`, which the join fan-out prices
    through as well.
    """
    try:
        schema = source.schema()
    except Exception as exc:
        note_suppressed("dist", "read the source schema for gpu shard packing", exc)
        return None
    from batcher.dist.gpu.shards import narrowed_schema

    return narrowed_schema(schema, projection)


def _run_shards(descriptors: list, shard_ops: list[dict], schema=None, gpu_count: int = 0) -> list:
    """Reduce every shard on a device, recovering from a failed one rather than the query.

    Uses the same straggler-backup barrier the CPU shuffle does: a shard is a pure function of
    its descriptor, so a duplicate of a slow one is safe and the barrier keeps whichever copy
    lands first. `on_failure` is what makes a bad shard local, on a two-rung ladder:

    * a shard that did not **fit** is subdivided and run on the device in pieces. The shard
      count was chosen from an estimate, and an estimate is wrong exactly where it matters — a
      skewed key, a wider row than the footer suggested, a neighbouring tenant on the device.
      Handing that shard, the largest piece of the work, to the slowest executor is the worst
      available answer, and subdividing is exact because the stage is mergeable.
    * anything else — a lost worker, an untranslatable expression — is recomputed by the native
      **CPU engine**, which produces the identical partial. A deterministic error fails the same
      way on a smaller shard, so it does not take the first rung.

    Those recomputations are *submitted* from the barrier and awaited after it, not awaited
    inside it. The case that matters is a spot reclamation taking several nodes at once, where
    resolving each recovery where it was noticed would run them one after another — turning the
    one event a fan-out most needs to absorb into the slowest possible response to it.

    A subdivision, by contrast, runs its pieces one at a time on purpose. The shard did not fit;
    running its pieces concurrently on the same device is how it would not fit again.
    """
    import ray

    from batcher.carbonite.resilience import gather_with_backups
    from batcher.dist.executors.ray_runtime import engine_config_json, speculation_policy
    from batcher.dist.gpu.resources import gpu_shard_options
    from batcher.dist.gpu.shards import ShardReport, is_memory_failure, run_subdivided
    from batcher.dist.gpu.tasks import cpu_shard_partial, gpu_shard_partial, gpu_task_options

    dc = active_config().distributed
    cfg_json = engine_config_json()
    # The fan-out already cut these shards small on purpose. Asking for a whole device per shard
    # then serializes them one per device, which is the cost the oversubscription was paying to
    # avoid. A shard that turns out not to fit its share falls into the subdivision ladder below,
    # so the packing can only make the run faster or make it subdivide — never make it fail.
    opts, packing = gpu_shard_options(descriptors, schema, gpu_count=gpu_count)
    gpu_task = ray.remote(**opts)(gpu_shard_partial)
    # The *retry* of a shard that did not fit goes back with a whole device. Retrying it on the
    # same share is the one combination with no argument for it: the share is the thing that was
    # just shown to be too small, and the pieces would be divided against it again. Un-packing
    # costs the co-tenancy for one shard's recovery and makes that recovery far more likely to
    # be the last one. Identical to `gpu_task` when nothing was packed, so an unpacked fan-out
    # builds no second handle.
    retry_task = ray.remote(**gpu_task_options())(gpu_shard_partial) if packing.packed else gpu_task
    cpu_task = ray.remote(max_retries=int(dc.task_max_retries))(cpu_shard_partial)

    report = ShardReport("gpu-chain", len(descriptors), packing=packing)

    def _launch(i: int):
        return gpu_task.remote(descriptors[i], shard_ops)

    def _on_failure(i: int, _ref, exc):
        if is_memory_failure(exc) and dc.gpu_shard_subdivide > 1:
            try:
                note_suppressed("dist", f"gpu shard {i} did not fit; subdividing", exc)
                report.note_subdivided()
                return run_subdivided(
                    descriptors[i],
                    lambda d: ray.get(retry_task.remote(d, shard_ops)),
                    parts=int(dc.gpu_shard_subdivide),
                    rounds=int(dc.gpu_shard_subdivide_rounds),
                    cause=exc,
                )
            except Exception as sub_exc:
                exc = sub_exc
        if not dc.gpu_shard_cpu_fallback:
            raise exc
        note_suppressed("dist", f"gpu shard {i}; recomputing on the CPU engine", exc)
        report.note_recovered()
        return _Recovering(cpu_task.remote(descriptors[i], shard_ops, cfg_json))

    refs = [_launch(i) for i in range(len(descriptors))]
    results = gather_with_backups(refs, _launch, speculation_policy(), on_failure=_on_failure)
    results = _await_recoveries(results)
    report.publish()
    return [t for t in results if t is not None and t.num_rows]


class _Recovering:
    """A shard whose CPU recomputation is in flight, standing in for its eventual result.

    Returned from the barrier's failure hook so the barrier does not block on it. Every
    recovery is therefore already running by the time the last one is awaited.
    """

    __slots__ = ("ref",)

    def __init__(self, ref) -> None:
        self.ref = ref


def _await_recoveries(results: list) -> list:
    """Resolve the in-flight recomputations, in one wait rather than one wait each."""
    pending = [i for i, r in enumerate(results) if isinstance(r, _Recovering)]
    if not pending:
        return results
    import ray

    out = list(results)
    for i, value in zip(pending, ray.get([results[i].ref for i in pending]), strict=True):
        out[i] = value
    return out


def merge_shards(partials: list, ops: list[dict]) -> pa.Table:
    """Combine the shards' results, then run whatever sat above the reducer.

    For a folded chain this runs on one row per group (or per distinct row, or per top-N entry)
    per shard — small by construction, which is the whole point of reducing before merging. For
    a row-local chain `ops` is empty and this is the concatenation itself, in shard order.

    Using the translator's own kernels keeps both halves of the algebra in one implementation.
    """
    import pyarrow as pa

    combined = pa.concat_tables(partials)
    if not ops:
        return combined
    import pandas as pd

    from batcher.core.gpu_plan import DfBackend
    from batcher.core.gpu_plan.execute import run_chain

    be = DfBackend(pd)
    return be.to_arrow(run_chain(combined, ops, be))


def fold_shards(partials: list, split) -> pa.Table:
    """Merge the shards' partials without ever holding all of them at once.

    "Small by construction" is true of one partial and false of a thousand of them. The fan-out
    bounds *device* memory by dividing the input; the merge then concatenates every shard's
    output on the **driver** before a single row is combined, so a group-by over a million
    groups fanned across a thousand shards materializes a billion rows in one process. That is
    the same failure the sharding was built to avoid, moved to the host — and it arrives
    precisely on the large multi-GPU clusters the fan-out exists for, because the shard count
    grows with the fleet.

    Folding in waves fixes it: combine `gpu_merge_wave` partials, keep the one result, drop the
    wave. Peak driver memory becomes a function of the wave size and the number of distinct
    groups, not of the shard count. It is *exact* rather than approximate because the fold is
    associative and commutative over its own output — see `plan.distribution.recombine`, which
    is the form that reads what the fold just wrote.

    Args:
        partials: Every shard's Arrow partial.
        split: The `ShardSplit` the fan-out was built from.

    Returns:
        The merged result, identical to `merge_shards(partials, merge_ops + tail_ops)` for any
        wave size.
    """
    from batcher.config import active_config

    tail = [*split.merge_ops, *split.tail_ops]
    wave = max(0, int(active_config().distributed.gpu_merge_wave))
    if not split.foldable or wave < 2 or len(partials) <= wave:
        return merge_shards(partials, tail)

    # First wave: the combine reads the partial stage's private columns, so each wave of raw
    # partials goes through `fold_ops`. Every wave after that reads results, so they go through
    # `refold_ops`. Reversing the two is the one way this could be wrong, and it fails loudly
    # (a missing column) rather than quietly.
    folded = [merge_shards(chunk, split.fold_ops) for chunk in _waves(partials, wave)]
    while len(folded) > 1:
        folded = [merge_shards(chunk, split.refold_ops) for chunk in _waves(folded, wave)]
    return merge_shards(folded, [*split.finalize_ops, *split.tail_ops])


def _waves(items: list, size: int) -> list[list]:
    """`items` in contiguous groups of at most `size`."""
    return [items[i : i + size] for i in range(0, len(items), size)]

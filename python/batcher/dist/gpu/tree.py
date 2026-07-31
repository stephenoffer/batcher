"""Fan a whole plan tree out across the cluster's GPUs: split one leaf, replicate the rest.

The single-scan fan-out divides the one relation a chain reads. A tree reads several, and only
one of them can be divided: a worker holding a slice of the fact table has to see the *whole* of
every dimension it joins to, or it produces rows for the matches its slice happened to contain
and silently loses the others.

So the shape is: split the largest leaf that `plan.distribution.shardable_leaves` says may be
split, give every worker a whole-relation descriptor for the others, and fold the shards'
partials through the same mergeable algebra a chain uses. Every worker reads its own inputs
straight from storage — the replicated leaves included — so nothing bulk travels as a Ray
object and the driver never holds a row.

Replication is what bounds this, and the bound is checked before the fan-out rather than
discovered as a device out-of-memory on every worker at once. A dimension small enough to sit
beside a shard of the fact table is the star-schema case this is built for; one that is not is
a big-to-big join, which wants a partitioning exchange instead, and until there is one the
honest answer is to decline and let the spillable CPU engine have it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed
from batcher.config import active_config

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.io.source import Source

__all__ = ["leaf_bytes", "plan_tree_fanout", "sharded_gpu_tree"]


def leaf_bytes(source: Source, projection: list[str] | None) -> float:
    """Roughly how many bytes of device memory this leaf's rows would occupy.

    `shards.source_bytes` does the arithmetic — one statement of how a relation is priced, so
    the shard count and the replication bound cannot size the same table differently. This adds
    the one thing a *tree* needs from it: an unmeasurable relation reads as infinitely large.

    That asymmetry is deliberate. Replicating a leaf needs an upper bound on its size, and
    "unknown" is not one; splitting a leaf needs no bound at all, since the shards are sized by
    the splits themselves. Treating unknown as large therefore disqualifies a leaf from being
    replicated while leaving it eligible to be the one that is split, which is the safe way
    round: the failure it avoids is every device in the fleet running out of memory at once.

    Args:
        source: The leaf's source.
        projection: The columns the leaf reads, or `None` for all of them.

    Returns:
        The estimated byte size, or `float("inf")` when the source cannot report a row count.
    """
    from batcher.dist.gpu.shards import source_bytes

    measured = source_bytes(source, projection)
    return measured if measured > 0 else float("inf")


def plan_tree_fanout(spec: dict, sources: list, *, replicate_budget_bytes: float):
    """Choose which leaf to split, or `None` when no fan-out over this tree is safe.

    Args:
        spec: A GPU plan-tree spec from `gpu_tree_spec`.
        sources: The query's sources, indexed by a leaf's `source_id`.
        replicate_budget_bytes: How many bytes of replicated leaves one device may hold beside
            its own shard. A tree whose replicated side exceeds this declines.

    Returns:
        `(pruned_spec, split_leaf, projections)` — the tree narrowed to the columns it reads,
        the leaf index to divide, and each leaf's column list — or `None` when no leaf may be
        split, or when replicating the others would not fit a device.
    """
    from batcher.core.gpu_plan.pruning import prune_tree
    from batcher.core.gpu_plan.tree import tree_leaves
    from batcher.plan.distribution import shardable_leaves

    leaves = tree_leaves(spec)
    candidates = shardable_leaves(spec)
    if not candidates:
        return None
    spec, projections = prune_tree(spec)
    sizes = {
        leaf["leaf"]: leaf_bytes(sources[leaf["source_id"]], projections.get(leaf["leaf"]))
        for leaf in leaves
    }
    # The biggest splittable leaf. Splitting the small side of a star schema would replicate the
    # fact table, which is the one arrangement guaranteed not to fit.
    split = max(candidates, key=lambda leaf: (sizes[leaf], -leaf))
    replicated = sum(size for leaf, size in sizes.items() if leaf != split)
    if replicated > replicate_budget_bytes:
        note_suppressed(
            "dist",
            "fan a gpu plan tree out",
            MemoryError(
                f"replicating {replicated / (1 << 30):.2f} GiB of non-split leaves exceeds the "
                f"{replicate_budget_bytes / (1 << 30):.2f} GiB per-device budget"
            ),
        )
        return None
    return spec, split, projections


def sharded_gpu_tree(
    spec: dict,
    sources: list,
    *,
    gpu_count: int,
    sharded: bool,
    replicate_budget_bytes: float,
) -> pa.Table | None:
    """Run a whole plan tree across the cluster's GPUs, or `None` when the fan-out cannot apply.

    Args:
        spec: A GPU plan-tree spec from `gpu_tree_spec`.
        sources: The query's sources, indexed by a leaf's `source_id`.
        gpu_count: The cluster's live device count.
        sharded: Whether the working set exceeds one device, so the split leaf must divide
            further than one shard per device.
        replicate_budget_bytes: Per-device budget for the leaves that are replicated.

    Returns:
        The tree's result, or `None` for any reason the fan-out does not apply — no splittable
        leaf, a replicated side too large, an unsplittable source, an unreadable cluster — so
        the caller can fall back to a single dispatch or to the CPU engine.
    """
    from batcher.plan.distribution import ShardSplit, shard_plan

    chosen = plan_tree_fanout(spec, sources, replicate_budget_bytes=replicate_budget_bytes)
    if chosen is None:
        return None
    spec, split_leaf, projections = chosen
    root_ops = spec["ops"]
    # The tree's root chain divides exactly as a linear chain's does; an empty one merges by
    # plain concatenation, which for a join tree is the only merge there is.
    split = shard_plan(root_ops) if root_ops else ShardSplit([], [], [], ordered=True)
    if split is None:
        return None

    descriptors = _leaf_descriptors(spec, sources, projections, split_leaf, gpu_count, sharded)
    if descriptors is None:
        return None
    shard_spec = {**spec, "ops": split.shard_ops}
    partials = _run_tree_shards(descriptors, shard_spec, split_leaf)
    if not partials:
        return None
    from batcher.dist.gpu.aggregate import fold_shards

    return fold_shards(partials, split)


def _leaf_descriptors(
    spec: dict,
    sources: list,
    projections: dict,
    split_leaf: int,
    gpu_count: int,
    sharded: bool,
) -> list[list[dict]] | None:
    """One descriptor list per shard, positional by leaf index.

    The split leaf contributes a different descriptor to each shard; every other leaf contributes
    the same whole-relation descriptor to all of them. Built once on the driver so a shard's
    task argument is a small manifest rather than a relation.
    """
    from batcher.core.gpu_plan.tree import tree_leaves
    from batcher.dist.gpu.aggregate import shard_descriptors
    from batcher.dist.gpu.dispatch import whole_source_descriptor

    leaves = tree_leaves(spec)
    shards = shard_descriptors(
        sources[leaves[split_leaf]["source_id"]],
        gpu_count,
        sharded=sharded,
        preserve_order=False,
        projection=projections.get(split_leaf),
    )
    if not shards:
        return None
    whole: dict[int, dict] = {}
    for leaf in leaves:
        index = leaf["leaf"]
        if index == split_leaf:
            continue
        descriptor = whole_source_descriptor(
            sources[leaf["source_id"]], projection=projections.get(index)
        )
        if descriptor is None:
            # An in-memory leaf has no manifest to describe. Shipping it to every worker would
            # be the driver staging a relation, which is the thing this path exists to avoid.
            return None
        whole[index] = descriptor
    return [
        [shard if i == split_leaf else whole[i] for i in range(len(leaves))] for shard in shards
    ]


def _run_tree_shards(descriptors: list[list[dict]], spec: dict, split_leaf: int) -> list:
    """Run every shard's tree on a device, subdividing one that does not fit rather than losing it.

    There is no CPU substitute here, and deliberately so: reconstructing one shard of a
    multi-way join through the engine means handing it a plan whose leaves are this shard and
    those whole relations, which is a second lowering of the same tree and a second chance to
    disagree with it. A shard that still will not fit after subdividing abandons the fan-out,
    and the caller falls back to a path that is one implementation rather than one and a half.
    """
    import ray

    from batcher.carbonite.resilience import gather_with_backups
    from batcher.dist.executors.ray_runtime import speculation_policy
    from batcher.dist.gpu.shards import ShardReport, is_memory_failure, run_subdivided
    from batcher.dist.gpu.tasks import gpu_task_options, gpu_tree_task

    dc = active_config().distributed
    task = ray.remote(**gpu_task_options())(gpu_tree_task)
    report = ShardReport("gpu-tree", len(descriptors))

    def _launch(i: int):
        return task.remote(descriptors[i], spec)

    def _on_failure(i: int, _ref, exc):
        if not is_memory_failure(exc) or dc.gpu_shard_subdivide <= 1:
            raise exc
        note_suppressed("dist", f"gpu tree shard {i} did not fit; subdividing", exc)
        report.note_subdivided()
        # Only the split leaf divides. The replicated ones are what every worker must see whole,
        # so a subdivision that also cut them would change the answer rather than the memory.
        return run_subdivided(
            descriptors[i],
            lambda d: ray.get(task.remote(d, spec)),
            parts=int(dc.gpu_shard_subdivide),
            rounds=int(dc.gpu_shard_subdivide_rounds),
            split=_split_leaf_only(split_leaf),
            cause=exc,
        )

    refs = [_launch(i) for i in range(len(descriptors))]
    results = gather_with_backups(refs, _launch, speculation_policy(), on_failure=_on_failure)
    report.publish()
    return [t for t in results if t is not None and t.num_rows]


def _split_leaf_only(split_leaf: int):
    """A subdivider that cuts only the leaf the fan-out split, leaving the replicated ones whole.

    Subdividing a replicated leaf would not reduce the shard's memory — every worker needs the
    whole of it — and would change the answer, because the join above it would then see a slice
    of a relation it was promised in full.
    """
    from batcher.dist.gpu.shards import split_descriptor

    def divide(current: list[dict], parts: int) -> list[list[dict]]:
        pieces = split_descriptor(current[split_leaf], parts)
        if len(pieces) == 1:
            return [current]
        return [[p if i == split_leaf else d for i, d in enumerate(current)] for p in pieces]

    return divide

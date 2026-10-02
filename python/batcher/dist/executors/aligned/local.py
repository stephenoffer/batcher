"""Broadcasts each node reads for itself: a large input that no filter shrinks.

A broadcast subtree is normally evaluated once and its result shipped to every node
(`hoist.hoist_broadcasts`). That is the right trade when a filter shrinks it -- the fleet
reads the source in parallel and ships a sliver -- and the wrong one when nothing does.
TPC-H q10 at SF100 broadcasts all 15M rows of `customer`: the fleet read them, the driver
gathered and packed them, and every node fetched them back through the driver's one link,
~17 s around 0.4 s of work per unit. The source is 2 GB each node can read from storage
alongside every other node.

So such a subtree travels as a recipe (`LocalBroadcast`): its plan and the read of each
source it scans. The first unit stream on a node to need it reads and evaluates it, and
every unit on that node joins that one result (`resolve_local`).
"""

from __future__ import annotations

import dataclasses
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa

from batcher.config.env import env_int
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan, Scan

__all__ = [
    "LOCAL_BROADCAST_BYTES",
    "LOCAL_FILTERED_BYTES",
    "LocalBroadcast",
    "local_broadcast",
    "on_node_schema",
    "reduces",
    "resolve_local",
]

#: Projected source bytes past which an unfiltered broadcast is read on each node instead.
#: Below it, the driver's round trip is cheap enough not to matter; above it, it is strictly
#: more work than each node's own read: the fleet reads the source, the driver gathers and
#: packs it, and every node fetches it back -- 1.3 s of TPC-H q14's `part` at SF100, which
#: decodes to 725 MB and which the footers, the size compared here, put at 198 MB.
LOCAL_BROADCAST_BYTES = env_int("BATCHER_ALIGNED_LOCAL_BYTES", 128 << 20, floor=1)
#: Projected source bytes up to which any broadcast, filtered or not, is read on each node.
#: The driver reads one serially before any unit runs -- TPC-H q20's `part` and `partsupp`
#: at SF10, 0.4-0.5 s -- where each node's own read overlaps its first unit's
#: (`run_units_here`). Past it a filter's saving on the wire is worth the driver's read.
LOCAL_FILTERED_BYTES = env_int("BATCHER_ALIGNED_LOCAL_FILTERED_BYTES", 256 << 20, floor=0)

# Concurrent split reads per node: each is latency-bound on its own requests.
_READERS = 16


@dataclasses.dataclass(frozen=True)
class LocalBroadcast:
    """A broadcast subtree to evaluate on each node: its plan and its sources' reads."""

    key: str  #: unique to one cut's run, so a node never serves a stale result
    ir: str  #: the subtree's plan, over the query's source ids and the held broadcasts'
    reads: dict[int, tuple[list, list[str] | None, dict | None]]  #: sid -> splits, pushdown
    empties: dict[int, list[pa.RecordBatch]]  #: a schema-only batch per source it reads
    empty: list[pa.RecordBatch]  #: the subtree's own schema-only result
    cfg_json: str


#: A column-to-literal comparison is judged to keep nearly every row when the files whose
#: footer range it overlaps hold this share of the source's bytes.
_KEEPS_MOST = 0.9

_BOUNDS = {"ge": "hi", "gt": "hi", "le": "lo", "lt": "lo"}


def reduces(node: LogicalPlan, sources: list[Source] | None = None) -> bool:
    """Whether `node` filters or groups its input, so its result may be far smaller.

    A predicate of only `IS NOT NULL` terms (which join planning adds everywhere) keeps
    nearly every row, so it does not count; nor, given `sources`, does a comparison of a
    scanned column with a literal that nearly every file's footer range satisfies. That is
    the key range Kyber derives from a join's other side: TPC-H q10 reads `customer` under
    `c_custkey BETWEEN 1 AND 14999999` at SF100, which is every customer.
    """
    from batcher.plan.logical import Aggregate, Filter
    from batcher.plan.visitor import walk

    def real(ir: dict, below: LogicalPlan) -> bool:
        if ir.get("e") == "is_not_null":
            return False
        if ir.get("e") == "binary" and ir.get("op") == "and":
            return real(ir["left"], below) or real(ir["right"], below)
        return sources is None or not _keeps_most(ir, below, sources)

    return any(
        (isinstance(n, Filter) and real(n.predicate.to_ir(), n.input))
        or (isinstance(n, Aggregate) and n.group_keys)
        for n in walk(node)
    )


def _keeps_most(ir: dict, below: LogicalPlan, sources: list[Source]) -> bool:
    """Whether comparison `ir` of a column with a literal keeps nearly all rows of `below`."""
    from batcher.dist.executors.aligned.units import source_key_bounds

    side = _BOUNDS.get(ir.get("op", "")) if ir.get("e") == "binary" else None
    left, right = ir.get("left", {}), ir.get("right", {})
    if side is None or left.get("e") != "col" or right.get("e") != "lit":
        return False
    origin = _scan_column(below, left["name"])
    if origin is None or origin[0] >= len(sources):
        return False
    value = next(iter((right.get("value") or {}).values()), None)
    bounds = source_key_bounds(sources[origin[0]], origin[1])
    if not isinstance(value, int | float) or not bounds:
        return False
    try:
        kept = sum(
            b.nbytes
            for b in bounds
            if (getattr(b, side) >= value if side == "hi" else getattr(b, side) <= value)
        )
    except TypeError:  # a bound of another type than the literal
        return False
    return kept >= _KEEPS_MOST * sum(b.nbytes for b in bounds)


def _scan_column(node: LogicalPlan, name: str) -> tuple[int, str] | None:
    """The scan, and its column, that `name` of `node` is read from, through plain columns."""
    from batcher.plan.expr_ir import Col
    from batcher.plan.logical import Filter, Project

    while not isinstance(node, Scan):
        if isinstance(node, Filter):
            node = node.input
        elif isinstance(node, Project):
            expr = {item.alias: item.expr for item in node.items}.get(name)
            if not isinstance(expr, Col):
                return None
            node, name = node.input, expr.name
        else:
            return None
    return node.source_id, name


def on_node_schema(
    node: LogicalPlan, scanned: set[int], sources: list[Source], budget: int
) -> pa.Schema | None:
    """The result schema of a subtree each node should read for itself, else None.

    One whose sources are large and that nothing filters: shipped from the driver, its
    result would cross the driver's link once per node at its full size. `budget` bounds
    the sources' projected bytes, as it bounds any held broadcast. The schema comes from
    running the subtree over no rows, which is what the placeholder scan needs.
    """
    from batcher.dist.executors.aligned.units import projected_bytes
    from batcher.dist.executors.partition_io import (
        empty_descriptor,
        read_partition_descriptor,
        source_pushdown,
    )
    from batcher.dist.executors.ray_runtime import _single_node
    from batcher.io.source import InMemorySource
    from batcher.plan.visitor import transform_up

    if any(s >= len(sources) for s in scanned):
        return None  # an earlier cut's result, which the driver already holds
    sizes = [projected_bytes(sources[s], source_pushdown(node, s)[0]) for s in scanned]
    if any(size is None for size in sizes):
        return None
    # Small enough that every node reading it beats one serial read on the driver, filtered
    # or not; otherwise only an unfiltered one, past what the driver ships cheaply.
    small = sum(sizes) <= LOCAL_FILTERED_BYTES
    large = LOCAL_BROADCAST_BYTES < sum(sizes) <= budget and not reduces(node, sources)
    if not (small or large):
        return None
    order = sorted(scanned)

    def renumber(n: LogicalPlan) -> LogicalPlan:
        if isinstance(n, Scan):
            return dataclasses.replace(n, source_id=order.index(n.source_id))
        return n

    empties = [
        InMemorySource(
            read_partition_descriptor(empty_descriptor(sources[s], source_pushdown(node, s)[0]))
        )
        for s in order
    ]
    return _single_node(transform_up(node, renumber), empties).schema


def local_broadcast(
    node: LogicalPlan, sources: list[Source], empty: pa.Schema, key: str, cfg_json: str
) -> LocalBroadcast:
    """The recipe that evaluates `node` on a node, reading its sources whole."""
    from batcher.dist.executors.partition_io import (
        empty_descriptor,
        read_partition_descriptor,
        source_pushdown,
    )
    from batcher.io.source import plan_splits
    from batcher.plan.visitor import scanned_source_ids

    reads, empties = {}, {}
    for sid in scanned_source_ids(node):
        if sid >= len(sources):
            continue  # a held broadcast, shipped alongside
        projection, predicate = source_pushdown(node, sid)
        splits = list(plan_splits(sources[sid], predicate=predicate, projection=projection))
        reads[sid] = (splits, projection, predicate)
        empties[sid] = read_partition_descriptor(empty_descriptor(sources[sid], projection))
    return LocalBroadcast(
        key=key,
        ir=json.dumps(node.to_ir()),
        reads=reads,
        empties=empties,
        empty=[pa.RecordBatch.from_pylist([], schema=empty)],
        cfg_json=cfg_json,
    )


# One result per recipe, per process: the fleet actor serving a node's units keeps it for
# every unit it runs. Only the current run's recipes are kept, so a node holds one query's.
_RESULTS: dict[str, list[pa.RecordBatch]] = {}
_LOCK = threading.Lock()


def resolve_local(held: dict[int, object]) -> dict[int, object]:
    """`held` with each `LocalBroadcast` replaced by its result, evaluated once per process."""
    recipes = {sid: v for sid, v in held.items() if isinstance(v, LocalBroadcast)}
    if not recipes:
        return held
    tables = {sid: v for sid, v in held.items() if not isinstance(v, LocalBroadcast)}
    # Held while evaluating: a second stream on this node waits for the first's result
    # rather than reading the same source again.
    with _LOCK:
        live = {r.key for r in recipes.values()}
        for stale in [k for k in _RESULTS if k not in live]:
            del _RESULTS[stale]
        for recipe in recipes.values():
            if recipe.key not in _RESULTS:
                _RESULTS[recipe.key] = _evaluate(recipe, tables)
        return {**tables, **{sid: _RESULTS[r.key] for sid, r in recipes.items()}}


def _evaluate(recipe: LocalBroadcast, tables: dict[int, object]) -> list[pa.RecordBatch]:
    from batcher.dist.executors.ray_runtime import execute_metered

    width = max([*recipe.reads, *tables]) + 1
    inputs: list[list] = [[] for _ in range(width)]
    for sid, batches in _read_all(recipe).items():
        inputs[sid] = batches or recipe.empties[sid]
    for sid, batches in tables.items():
        inputs[sid] = batches
    rows, _metrics = execute_metered(recipe.ir, inputs, recipe.cfg_json)
    return [b for b in rows if b.num_rows] or recipe.empty


def _read_all(recipe: LocalBroadcast) -> dict[int, list[pa.RecordBatch]]:
    """Every source the recipe reads, its splits fetched concurrently."""
    from batcher.dist.executors.partition_io import read_partition_descriptor

    jobs = [
        (sid, {"splits": [split], "projection": projection, "predicate": predicate})
        for sid, (splits, projection, predicate) in recipe.reads.items()
        for split in splits
    ]
    out: dict[int, list[pa.RecordBatch]] = {sid: [] for sid in recipe.reads}
    with ThreadPoolExecutor(max_workers=max(1, min(_READERS, len(jobs)))) as pool:
        for (sid, _desc), batches in zip(
            jobs, pool.map(lambda job: read_partition_descriptor(job[1]), jobs), strict=True
        ):
            out[sid].extend(b for b in batches if b.num_rows)
    return out
